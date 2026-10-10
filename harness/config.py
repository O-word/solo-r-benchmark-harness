"""Matrix config loading, validation and expansion into sessions (stdlib only; JSON, no YAML dependency)."""
import itertools
import os
import random

from .util import EXTERNAL_PORTS, FORBIDDEN_PORTS, HARNESS_ROOT, REAL_PORT, STUB_PORT, TUNNEL_PORT, load_json, seed_from

PERSPECTIVES = ("first", "second", "third")
LEVELS = (1, 2, 3)


class ConfigError(Exception):
    pass


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(HARNESS_ROOT, path)


def load_config(path):
    cfg = load_json(path)
    validate(cfg)
    return cfg


def validate(cfg):
    def need(cond, msg):
        if not cond:
            raise ConfigError(msg)

    need(cfg.get("name"), "config needs a name")
    need(cfg.get("mode") in ("stub", "real", "external"), "mode must be 'stub', 'real' or 'external'")
    for p in cfg.get("perspectives", []):
        need(p in PERSPECTIVES, "bad perspective %r" % p)
    for l in cfg.get("levels", []):
        need(int(l) in LEVELS, "bad level %r" % l)
    need(cfg.get("tense") in ("past", "present"), "tense must be past or present")
    need(int(cfg.get("turns", 0)) >= 1, "turns must be >= 1")
    need(cfg.get("configurations"), "need at least one configuration")
    ids = [c["id"] for c in cfg["configurations"]]
    need(len(ids) == len(set(ids)), "configuration ids must be unique")
    for c in cfg["configurations"]:
        srv = c.get("server", {})
        port = int(srv.get("port", 0))
        need(port not in FORBIDDEN_PORTS, "configuration %s uses forbidden port %s (the owner's play servers)" % (c["id"], port))
        if cfg["mode"] == "stub":
            need(srv.get("kind") == "stub" and port == STUB_PORT, "stub mode: configuration %s must use the stub server on port %d" % (c["id"], STUB_PORT))
        elif cfg["mode"] == "external":
            # An EXTERNAL server is already running somewhere else and reached through a local tunnel port. The harness starts nothing.
            need(srv.get("kind") == "external", "external mode: configuration %s must use server kind 'external'" % c["id"])
            need(port in EXTERNAL_PORTS, "external mode: configuration %s must use the tunnel port %d (or the stub port %d, tests only), not %s" % (c["id"], TUNNEL_PORT, STUB_PORT, port))
            need(srv.get("host", "127.0.0.1") == "127.0.0.1", "external server host must be 127.0.0.1 (the tunnel's local end)")
            need("command" not in srv, "external server must not have a command (the harness never starts it)")
            need(str(c.get("model_name") or "").strip() and not any(ch.isspace() for ch in str(c["model_name"])), "external configuration %s needs a model_name (no spaces), e.g. 'base' or 'lora'" % c["id"])
            need(c.get("force_model", True), "external configurations always send the exact model_name (force_model cannot be false)")
        else:
            need(srv.get("kind") == "real" and port == REAL_PORT, "real mode: configuration %s must use kind 'real' on port %d" % (c["id"], REAL_PORT))
        if "model_name" in c:
            need(isinstance(c["model_name"], str) and c["model_name"].strip(), "model_name must be a non-empty string")
        need(c.get("instance_policy", "fresh_per_session") in ("fresh_per_session", "long_running"), "bad instance_policy")
        if cfg["mode"] == "real":
            toks = [str(t) for t in srv.get("command", [])]
            need("--host" in toks and toks[toks.index("--host") + 1] == "127.0.0.1", "real server command must bind --host 127.0.0.1")
            need("--port" in toks and toks[toks.index("--port") + 1] in ("{port}", str(REAL_PORT)), "real server command must use --port {port} (1236)")
        need(isinstance(c.get("guards", True), bool), "guards must be true/false")
    for e in cfg.get("extra_sessions", []):
        need(e["configuration"] in ids, "extra session references unknown configuration %r" % e["configuration"])
    return cfg


def load_scenario(sid):
    return load_json(os.path.join(HARNESS_ROOT, "scenarios", sid + ".json"))


def load_companion(cid):
    return load_json(os.path.join(HARNESS_ROOT, "companions", cid + ".json"))


def expand_sessions(cfg):
    """Return the ordered session list.

    Order (cfg["order"], default "configuration_major"): every session of the first configuration, then every session
    of the next (so only one model instance is ever needed at a time and the BASE run finishes before the LORA run
    starts). Inside a configuration the cells are shuffled with the run seed so machine drift is not confounded with
    a cell. "interleaved" shuffles across configurations instead. Extra sessions (incident / lifecycle demos) follow.
    Optional configurations (e.g. 'guards off' variants) are skipped unless cfg["include_optional"] is true.

    Order "paired_alternating" (the DEFAULT when every configuration uses an external server, which can serve all of them at once
    without restarts): cells are shuffled with the run seed and every cell is run once per configuration back to back, with the
    configuration that goes first rotating from cell to cell (BASE,LORA / LORA,BASE / ...). Time of day, server warm-up and
    tunnel conditions then fall evenly on every configuration instead of one finishing before the other starts.

    Seeds do NOT include the configuration id: the same cell gets the same poses and the same sampling seeds in
    every configuration, so configurations are compared on identical input.
    """
    rng = random.Random(cfg["seed"])
    include_opt = bool(cfg.get("include_optional"))
    mx = [c for c in cfg["configurations"] if c.get("matrix", True) and (include_opt or not c.get("optional"))]
    cells = [(persp, int(lvl), scen, rep) for persp, lvl, scen, rep in itertools.product(
        cfg["perspectives"], cfg["levels"], cfg["scenarios"], range(1, int(cfg.get("replicates", 1)) + 1))]
    sessions = []
    default_order = "paired_alternating" if mx and all(c["server"].get("kind") == "external" for c in mx) else "configuration_major"
    order = cfg.get("order", default_order)
    if order == "paired_alternating":
        cc = list(cells)
        rng.shuffle(cc)
        pairs = []
        for k, cell in enumerate(cc):
            rot = k % len(mx) if mx else 0
            pairs += [(c, cell) for c in (mx[rot:] + mx[:rot])]
    elif order == "interleaved":
        allx = [(c, cell) for c in mx for cell in cells]
        rng.shuffle(allx)
        pairs = allx
    else:
        pairs = []
        for c in mx:
            cc = list(cells)
            rng.shuffle(cc)
            pairs += [(c, cell) for cell in cc]
    for conf, (persp, lvl, scen, rep) in pairs:
        sessions.append({"configuration": conf["id"], "perspective": persp, "level": lvl, "scenario": scen, "replicate": rep, "turns": int(cfg["turns"])})
    for e in cfg.get("extra_sessions", []):
        sessions.append({"configuration": e["configuration"], "perspective": e["perspective"], "level": int(e["level"]), "scenario": e["scenario"],
                         "replicate": int(e.get("replicate", 1)), "turns": int(e.get("turns", cfg["turns"])), "extra": True})
    out = []
    for n, s in enumerate(sessions, 1):
        s["index"] = n
        s["session_id"] = "s%03d_%s_L%d_%s_%s_r%d" % (n, s["perspective"], s["level"], s["scenario"], s["configuration"], s["replicate"])
        s["seed"] = seed_from(cfg["seed"], s["perspective"], s["level"], s["scenario"], s["replicate"], "x" if s.get("extra") else "")
        out.append(s)
    return out


def configuration(cfg, cid):
    return next(c for c in cfg["configurations"] if c["id"] == cid)
