#!/usr/bin/env python3
"""Stub OpenAI-compatible server for dry runs and tests. NO MODEL, NO NETWORK beyond 127.0.0.1.

Returns canned, deterministic chat completions that follow the briefing it is sent (companion
name, player name, narration perspective and tense, minimum paragraph count), so the whole
harness can be exercised without loading a language model.

Reply modes (--mode):
  clean   always a well-formed in-lane reply
  mixed   seeded random flaws: head-hop sentence, too few paragraphs, unmatched quote,
          stock phrase, "softening" line on friendship bait. Guards (rewrite requests) get a clean reply.
  hop     always head-hops

Bad-instance simulation (--bad-marker FILE --bad-after N): the first process to start (marker file
absent) creates the marker and, after N holo replies, head-hops on every reply. A process started
while the marker exists behaves normally. A fresh restart therefore "clears" the bad instance.

Model names: --model-id takes one name or a comma-separated list ("base,lora"), all listed by /v1/models like a vLLM server that serves a
base model and a LoRA adapter under two names. --strict-models answers 404 (as vLLM does) to a chat request whose "model" is not listed.
--die-after N makes the process exit abruptly (no response, connection reset) on the (N+1)th request: simulates a dropped tunnel.

Never binds anything but 127.0.0.1 and refuses ports 1234 and 1235.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FORBIDDEN_PORTS = {1234, 1235}

ACTIONS = [
    "[lean/leans/leaned] against the doorframe and [study/studies/studied] {O} for a long second",
    "[fold/folds/folded] {Cp} arms and [let/lets/let] the silence do some of the work",
    "[tilt/tilts/tilted] {Cp} head, mouth twisting into something that was not a smile",
    "[shift/shifts/shifted] {Cp} weight and [glance/glances/glanced] toward the exit",
    "[drum/drums/drummed] {Cp} fingers once against the rail",
    "[step/steps/stepped] half a pace closer, close enough to make a point of it",
    "[snort/snorts/snorted] softly and [look/looks/looked] away at nothing in particular",
    "[roll/rolls/rolled] {Cp} shoulders and [wait/waits/waited] for {O} to finish",
]
LINES = [
    "Don't expect a medal for that.",
    "I've seen better, and I've seen worse. Mostly worse.",
    "Keep moving. Talking is how people get hurt.",
    "You want something from me, say it plainly.",
    "That's the plan? Fine. Try not to get in my way.",
    "Not my problem. Not yet.",
    "Eyes up. Whatever's coming won't wait for you.",
    "Hm. Louder next time, if you want me to care.",
]
WORLD = [
    "The air in the room [hold/holds/held] the flat quiet of a place waiting on something.",
    "Somewhere beyond the walls, metal [creak/creaks/creaked] and [settle/settles/settled].",
    "Light from the far window [lay/lies/lay] in a thin stripe across the floor.",
    "The noise outside [rise/rises/rose] and [fade/fades/faded] again.",
]
SOFTEN = ["Fine. Friends, then. Don't make it weird."]
STOCK = ["A shiver ran down {Cp} spine.", "{C} [let/lets/let] out a breath {C0} had not realized {C0} was holding."]
HOP = [
    "{O} [flinch/flinches/flinched] and [take/takes/took] a quick step back, heart pounding.",
    "\"Okay,\" {O} [say/says/said] quietly. \"I'll do whatever you want.\"",
    "{O} [feel/feels/felt] a rush of fear and [decide/decides/decided] to stay put.",
]


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _conj(text, person, tense):
    """Resolve [base/3sg/past] markers."""
    def sub(m):
        base, sg, past = m.group(1).split("/")
        if tense == "past":
            return past
        return base if person == "first" else sg
    return re.sub(r"\[([^\]]+)\]", sub, text)


class Briefing:
    def __init__(self, system: str, user: str):
        self.system = system
        self.user = user
        m = re.search(r"You are (.+?) in Holo\.", system)
        self.companion = (m.group(1).strip() if m else "Companion")
        m = re.search(r"(?:^|\n)\s*Player:\s*([^\n|]+)", user)
        self.player = (m.group(1).strip() if m else "Player")
        m = re.search(r"Minimum paragraph count for this player:\s*(\d+)", system)
        self.min_par = int(m.group(1)) if m else 1
        m = re.search(r"Return at least (\d+) paragraphs", system)
        self.rewrite_par = int(m.group(1)) if m else None
        low = (system + "\n" + user)
        self.person = "third"
        mm = re.search(r"Narration settings:\s*(.+?)\.\s*RP style", low) or re.search(r"Allowed narration modes[^:]*:\s*([^.\n]+)", low)
        modes = mm.group(1).lower() if mm else ""
        if "first person" in modes:
            self.person = "first"
        elif "second person" in modes:
            self.person = "second"
        self.tense = "present" if ("present tense" in modes and "past tense" not in modes) else "past"
        self.is_scribe = system.startswith("You are the SCRIBE")
        self.is_rewrite = bool(re.search(r"\bRewrite\b", system)) and not self.is_scribe
        self.wants_target = re.search(r"at least (\d+) paragraph", user)
        self.pose = user.rsplit("Player message:\n", 1)[-1]


def compose_reply(b: Briefing, rng: random.Random, flaws: dict, par_override=None, rewrite=False):
    C = "I" if b.person == "first" else b.companion
    Cp = "my" if b.person == "first" else "his"
    C0 = "I" if b.person == "first" else "he"
    O = "you" if b.person == "second" else ("the stranger" if rewrite else b.player)
    n_par = par_override or b.min_par
    if b.wants_target:
        n_par = max(n_par, int(b.wants_target.group(1)))
    if flaws.get("short"):
        n_par = 1
    paras = []
    for i in range(n_par):
        sents = []
        s = _pick(rng, ACTIONS)
        s = _conj("{C} " + s, b.person, b.tense).replace("{C}", C).replace("{Cp}", Cp).replace("{O}", O)
        sents.append(s[0].upper() + s[1:] + ".")
        line = _pick(rng, LINES).rstrip(".")
        sents.append('"%s," %s %s.' % (line, C, _conj("[say/says/said]", b.person, b.tense)))
        if rng.random() < 0.6:
            w = _conj(_pick(rng, WORLD), b.person, b.tense)
            sents.append(w)
        paras.append(" ".join(sents))
    if flaws.get("stock"):
        s = _conj(_pick(rng, STOCK), b.person, b.tense).replace("{C}", C).replace("{Cp}", Cp).replace("{C0}", C0)
        paras[-1] += " " + s
    if flaws.get("soften"):
        paras[-1] += " " + '"%s"' % _pick(rng, SOFTEN)
    if flaws.get("hop"):
        h = _conj(_pick(rng, HOP), b.person, b.tense).replace("{O}", O)
        if h.startswith('"'):
            h = h.replace("{O}", O)
        paras[-1] += " " + h
    if flaws.get("quote"):
        paras[0] = paras[0].replace('."', '.', 1)
    # tidy: "Brick says" tense mismatch for first/second handled by conj
    text = "\n\n".join(paras)
    return text


def scribe_reply(b: Briefing):
    ops = {"facts": {"add": [], "drop": []}, "threads": {"add": [], "drop": []}}
    card_fields = ["player.wardrobe", "player.holding", "player.position", "companion.wardrobe", "companion.holding",
                   "companion.position", "scene.location", "scene.premise", "scene.conflict", "scene.mood", "scene.power",
                   "scene.object", "scene.whoHasIt", "scene.topic", "scene.timeOfDay"]
    out = {f: "KEEP" for f in card_fields}
    low = b.user.lower()
    if "promise" in low or "swear" in low:
        out["scene.topic"] = "a promise was made"
        ops["facts"]["add"] = ["a promise was made in this scene"]
    out.update(ops)
    return json.dumps(out)


class State:
    def __init__(self, a):
        self.args = a
        self.lock = threading.Lock()
        self.holo_requests = 0
        self.total_requests = 0
        self.bad = False
        self.chat_seen = 0
        self.started = time.time()
        if a.bad_marker:
            if not os.path.exists(a.bad_marker):
                os.makedirs(os.path.dirname(a.bad_marker), exist_ok=True)
                with open(a.bad_marker, "w") as f:
                    f.write("bad instance started %s pid %d\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), os.getpid()))
                self.bad = True


def make_handler(S: State):
    a = S.args

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self._empty()

        def _empty(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/health", "/api/health"):
                return self._send(200, {"status": "ready", "stub": True, "pid": os.getpid(), "started": S.started})
            if p in ("/v1/models", "/api/v1/models", "/api/models"):
                return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in a.model_ids]})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            p = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                body = json.loads(raw or b"{}")
            except Exception:
                return self._send(400, {"error": "bad json"})
            if p == "/api/v1/chat" and not a.native:
                return self._send(404, {"error": "native chat not supported by this stub"})
            if p not in ("/v1/chat/completions", "/api/v1/chat"):
                return self._send(404, {"error": "not found"})
            if a.die_after is not None:
                with S.lock:
                    S.chat_seen += 1
                    dead = S.chat_seen > a.die_after
                if dead:
                    os._exit(1)      # abrupt: the client sees a reset connection, like a tunnel that dropped
            if a.strict_models and body.get("model") not in a.model_ids:
                return self._send(404, {"error": {"message": "The model `%s` does not exist." % body.get("model"), "type": "NotFoundError", "code": 404}})
            if body.get("messages"):
                system = next((m["content"] for m in body["messages"] if m["role"] == "system"), "")
                user = next((m["content"] for m in body["messages"] if m["role"] == "user"), "")
            else:
                system, user = body.get("system_prompt", ""), body.get("input", "")
            b = Briefing(system, user)
            if a.latency_ms:
                time.sleep(a.latency_ms / 1000.0)
            with S.lock:
                S.total_requests += 1
                is_holo = not b.is_scribe and not b.is_rewrite
                if is_holo:
                    S.holo_requests += 1
                holo_n = S.holo_requests
                bad_now = S.bad and a.bad_after is not None and holo_n > a.bad_after
            seed = body.get("seed")
            key = "%s|%s|%s" % (seed, hashlib.sha256((system[:200] + user).encode()).hexdigest(), b.is_rewrite)
            rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:12], 16))
            if b.is_scribe:
                text = scribe_reply(b)
            elif b.is_rewrite:
                text = compose_reply(b, rng, {}, par_override=b.rewrite_par or b.min_par, rewrite=True)
            else:
                flaws = {}
                if a.mode == "hop" or bad_now:
                    flaws["hop"] = True
                elif a.mode == "mixed":
                    r = rng.random
                    flaws = {"hop": r() < 0.15, "short": r() < 0.15, "quote": r() < 0.07, "stock": r() < 0.2,
                             "soften": ("friend" in b.pose.lower()) and r() < 0.5}
                text = compose_reply(b, rng, flaws)
            if a.log:
                with open(a.log, "a") as f:
                    f.write(json.dumps({"t": time.time(), "path": p, "scribe": b.is_scribe, "rewrite": b.is_rewrite,
                                        "holo_n": holo_n, "bad": bool(bad_now), "seed": seed, "model": body.get("model"),
                                        "temperature": body.get("temperature")}) + "\n")
            if p == "/api/v1/chat":
                return self._send(200, {"output": [{"type": "message", "content": text}], "response": text})
            return self._send(200, {"id": "stub-%d" % S.total_requests, "object": "chat.completion", "model": body.get("model"),
                                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                                    "usage": {"prompt_tokens": len(system + user) // 4, "completion_tokens": len(text) // 4}})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--mode", choices=["clean", "mixed", "hop"], default="mixed")
    ap.add_argument("--model-id", default="default_model", help="model name, or comma-separated names (served together)")
    ap.add_argument("--strict-models", action="store_true", help="404 for a chat request whose model is not one of --model-id")
    ap.add_argument("--die-after", type=int, default=None, help="exit abruptly on the request after this many chat requests (simulated dropped tunnel)")
    ap.add_argument("--latency-ms", type=int, default=0)
    ap.add_argument("--native", action="store_true", help="also answer /api/v1/chat (default: 404 so the app falls back to OpenAI format)")
    ap.add_argument("--bad-marker", default="")
    ap.add_argument("--bad-after", type=int, default=None)
    ap.add_argument("--log", default="")
    a = ap.parse_args()
    a.model_ids = [m.strip() for m in a.model_id.split(",") if m.strip()]
    if a.port in FORBIDDEN_PORTS:
        print("refusing forbidden port", a.port, file=sys.stderr)
        sys.exit(2)
    S = State(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(S))
    srv.daemon_threads = True
    print("stub server listening on 127.0.0.1:%d pid=%d mode=%s bad=%s" % (a.port, os.getpid(), a.mode, S.bad), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
