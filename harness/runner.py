"""The benchmark runner: sessions, turns, scoring, incidents, proof folder, pause/resume.

Unit of work = a SESSION: fresh model-server start + fresh app profile + N turns. Everything is written to the run
folder as it happens (flushed per record), so a crash, a stop, or a bot/judge failure never loses recorded turns.

Resume: `python -m harness resume --run DIR` continues the same folder. Completed sessions are skipped; a session that was
cut off is re-run from its first turn as a new ATTEMPT on a fresh instance (a half-finished conversation cannot be resumed
honestly: the model instance and app profile are gone). Earlier attempts stay in the files and are excluded from aggregates.
"""
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback

from . import __version__, scoring
from .config import configuration, expand_sessions, load_companion, load_scenario, resolve
from .electron_driver import DriverError, ElectronSession
from .gemini_client import key_status
from .judges import make_judge, rubric_info
from .player_bots import BotError, ensure_pose_command, make_bot
from .server_manager import (ExternalServer, ExternalServerDown, ServerError, ServerManager, harness_descendants, harness_electron_processes,
                             make_stub_command, port_in_use, required_paths, stray_processes)
from .util import (FORBIDDEN_PORTS, HARNESS_ROOT, TUNNEL_PORT, JsonlAppender, dump_json, load_json, now_iso, read_jsonl, seed_from, sha256_file, slug, tree_hash)

SCORE_COLUMNS = [
    "run_id", "session_id", "attempt", "phase", "incident_id", "configuration", "guards_on", "perspective", "level", "scenario", "replicate", "turn_index", "turn_number", "turn_id", "bait", "bait_type", "model_name", "model_sent",
    "status", "pose_kind", "pose_chars", "reply_chars", "words", "paragraphs", "paragraphs_raw", "min_paragraphs", "target_paragraphs", "para_meets_level", "para_meets_target", "para_meets_level_raw",
    "head_hop", "head_hop_raw", "head_hop_sanitized", "head_hop_soft_final", "head_hop_reasons", "lane_high",
    "unmatched_quotes", "unmatched_quotes_raw", "quote_problems", "quote_misplaced", "quote_misplaced_raw", "quote_structure", "format_issues", "repeat_rate_per_1000", "stock_phrases", "stock_per_1000", "mattr", "mirror_rate",
    "person_ok", "tense_ok", "person_ok_raw", "tense_ok_raw", "person_tense_reasons", "hard_rule_violations", "hard_rule_violations_raw", "hard_rule_detail", "pref_violations", "pref_violations_raw", "pref_detail", "soften_hit", "adherence_ok", "adherence_rules_only_ok", "stasis_ok", "target_replica_mismatch",
    "guard_rewrote", "rewrite_count", "guard_would_fire", "ms_model", "instance_id", "instance_age_s",
    "judge_status", "judge_model", "judge_head_hop", "judge_head_hop_confidence", "judge_personality_lock", "judge_writing_quality", "judge_rules_adherence", "judge_rules_ok",
]
APPEND_ONLY_FILES = ["config.json", "transcripts.jsonl", "scores.csv", "judge_raw.jsonl", "incidents.jsonl", "sessions.jsonl", "resume_log.jsonl"]


class AbortRun(Exception):
    pass


def hash_path(path, mode):
    """File: size/mtime (+sha256 when mode == 'full'). Directory: every file listed (+ per-file sha256 when 'full')."""
    st = os.stat(path)
    if os.path.isfile(path):
        e = {"path": path, "size": st.st_size, "mtime": st.st_mtime}
        if mode == "full":
            e["sha256"] = sha256_file(path)
        return e
    files = {}
    for dp, dn, fn in os.walk(path):
        dn.sort()
        for n in sorted(fn):
            if n.startswith("._") or n == ".DS_Store":
                continue
            full = os.path.join(dp, n)
            fs = os.stat(full)
            ent = {"size": fs.st_size, "mtime": fs.st_mtime}
            if mode == "full":
                ent["sha256"] = sha256_file(full)
            files[os.path.relpath(full, path)] = ent
    return {"path": path, "type": "directory", "files": files, "n_files": len(files), "total_bytes": sum(f["size"] for f in files.values()),
            "listing_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), "hash_mode": mode}


class Run:
    def __init__(self, cfg, config_path, runs_root=None, name_suffix="", progress=print, resume_dir=None, max_minutes=None, nice=None, hard_extra_minutes=10):
        self.progress = progress
        self.t_start = time.time()
        self.max_minutes = max_minutes
        self.hard_extra_minutes = hard_extra_minutes
        self.resuming = resume_dir is not None
        self.config_path = config_path
        self.session_durations = []
        self.nice = nice
        if self.resuming:
            self.dir = os.path.abspath(resume_dir)
            if os.path.exists(os.path.join(self.dir, "MANIFEST.sha256")):
                raise AbortRun("this run is complete and sealed (MANIFEST.sha256 exists); runs are append-only and cannot be resumed")
            saved = load_json(os.path.join(self.dir, "config.json"))
            cfg = saved["config"]
            self.run_id = saved["run_id"]
            self.config_path = os.path.join(self.dir, "config.json")
        else:
            stamp = time.strftime("%Y-%m-%d_%H%M")
            base = "%s_%s" % (stamp, slug(cfg["name"] + name_suffix))
            self.runs_root = runs_root or os.path.join(HARNESS_ROOT, "runs")
            os.makedirs(self.runs_root, exist_ok=True)
            d, k = os.path.join(self.runs_root, base), 1
            while os.path.exists(d):  # runs are append-only: never reuse a folder
                k += 1
                d = os.path.join(self.runs_root, "%s_%d" % (base, k))
            self.dir = d
            self.run_id = os.path.basename(d)
        self.cfg = cfg
        self.scratch = os.path.join(self.dir, "_scratch")
        for sub in ("_scratch", "graphs", "logs"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        d = self.dir
        # existing state (resume)
        self.prev_sessions = read_jsonl(os.path.join(d, "sessions.jsonl")) if self.resuming else []
        self.done = {s["session_id"] for s in self.prev_sessions if s.get("event") == "session" and s.get("status") == "complete"}
        self.attempts = {}
        for r in self.prev_sessions:
            if r.get("event") == "session_start":
                self.attempts[r["session_id"]] = max(self.attempts.get(r["session_id"], 0), int(r["attempt"]))
        self.incidents = read_jsonl(os.path.join(d, "incidents.jsonl")) if self.resuming else []
        ids = [int(str(r["instance_id"])[1:]) for r in self.prev_sessions if r.get("instance_id")]
        ids += [int(str(i)[1:]) for r in self.prev_sessions for i in (r.get("instance_ids") or [])]
        self.instance_counter = max(ids) if ids else 0
        self.prior_elapsed = 0.0
        self.total_turns = 0
        if self.resuming:
            self.total_turns = sum(1 for r in read_jsonl(os.path.join(d, "transcripts.jsonl")) if not r.get("event"))
            for r in read_jsonl(os.path.join(d, "resume_log.jsonl")):
                self.prior_elapsed += float(r.get("elapsed_s", 0))
        self.transcripts = JsonlAppender(os.path.join(d, "transcripts.jsonl"))
        self.judge_raw = JsonlAppender(os.path.join(d, "judge_raw.jsonl"))
        self.incidents_f = JsonlAppender(os.path.join(d, "incidents.jsonl"))
        self.sessions_f = JsonlAppender(os.path.join(d, "sessions.jsonl"))
        self.resume_f = JsonlAppender(os.path.join(d, "resume_log.jsonl"))
        scores_path = os.path.join(d, "scores.csv")
        new_scores = not os.path.exists(scores_path) or os.path.getsize(scores_path) == 0
        self._scores_fh = open(scores_path, "a", newline="", encoding="utf-8")
        fieldnames = SCORE_COLUMNS
        if not new_scores:
            with open(scores_path, newline="", encoding="utf-8") as hf:
                fieldnames = next(csv.reader(hf))          # a resumed run keeps the columns it started with
        self._scores = csv.DictWriter(self._scores_fh, fieldnames=fieldnames, extrasaction="ignore")
        if new_scores:
            self._scores.writeheader()
            self._scores_fh.flush()
        self.bot = make_bot(cfg.get("player_bot"))
        self.judge = make_judge(cfg.get("judge"))
        self.sessions = expand_sessions(cfg)
        self.snapshot_dir = resolve(cfg.get("app_snapshot", "app_snapshot"))
        self.app_index = os.path.join(self.snapshot_dir, "app", "index.html")
        self.electron_bin = cfg["electron_bin"]
        self.holder = None            # (configuration_id, ServerManager, info)
        self.bot_failures = 0
        self.status = "running"
        self.status_pause = None
        self.electron_version = None
        self.isolation = {"probe_done": False}

    def log(self, msg):
        self.progress("[%s] %s" % (time.strftime("%H:%M:%S"), msg))

    # ------------------------------------------------------------------ config.json (written once, at the start)
    def snapshot_verify(self):
        sums = os.path.join(self.snapshot_dir, "SNAPSHOT.sha256")
        want = {}
        for line in open(sums, encoding="utf-8"):
            if line.strip():
                h, rel = line.rstrip("\n").split("  ", 1)
                want[rel] = h
        th, lines = tree_hash(self.snapshot_dir)
        got = {l.split("  ", 1)[1]: l.split("  ", 1)[0] for l in lines if not l.endswith("  SNAPSHOT.sha256")}
        bad = sorted([r for r in want if got.get(r) != want[r]] + [r for r in got if r not in want])
        if bad:
            raise AbortRun("app snapshot does not match SNAPSHOT.sha256 (changed, missing or extra): %s" % bad[:5])
        return {"index_html_sha256": got["app/index.html"], "tree_sha256": th, "files": len(got),
                "app_original_preload_sha256": got.get("preload.cjs"), "app_original_main_sha256": got.get("main.cjs"),
                "note": "the app's own preload.cjs is NOT used; driver/preload_stub.cjs replaces it so legacy shard folders are never read"}

    def write_config(self):
        cfg = self.cfg
        cfg_out = {
            "harness_version": __version__,
            "run_id": self.run_id,
            "created": now_iso(),
            "invocation": " ".join(sys.argv),
            "config_source": os.path.relpath(self.config_path, HARNESS_ROOT) if self.config_path.startswith(HARNESS_ROOT) else self.config_path,
            "config_file_sha256": sha256_file(self.config_path),
            "config": cfg,
            "app_snapshot": self.snapshot_verify(),
            "electron": {"binary": self.electron_bin, "binary_sha256": sha256_file(self.electron_bin) if os.path.isfile(self.electron_bin) else None},
            "python": platform.python_version(), "platform": platform.platform(),
            "seeds": {"run_seed": cfg["seed"], "per_session": {s["session_id"]: s["seed"] for s in self.sessions},
                      "turn_seed_rule": "seed_from(session_seed, 'turn', turn_index); injected into every chat request body as 'seed' when inject_seed is true. Session seeds exclude the configuration id, so every configuration sees the same poses and seeds."},
            "temperatures": {c["id"]: {"injected_by_harness": c.get("temperature"), "server_default": (c.get("server_defaults") or {}).get("temperature")} for c in cfg["configurations"]},
            "sampling_note": "when injected temperature is null the app sends none, so the server's launch-argument default applies (recorded under server_defaults and in server_args)",
            "player_bot": self.bot.describe(),
            "judge": self.judge.describe(),
            "rubric": rubric_info(),
            "scenarios": {s: {"sha256": sha256_file(os.path.join(HARNESS_ROOT, "scenarios", s + ".json"))} for s in sorted({x["scenario"] for x in self.sessions})},
            "companions": {c: {"sha256": sha256_file(os.path.join(HARNESS_ROOT, "companions", c + ".json"))} for c in sorted({load_scenario(x["scenario"])["companion"] for x in self.sessions})},
            "model_files": self.hash_model_files(),
            "model_names": {c["id"]: (c.get("model_name") or c.get("model", "default_model")) for c in cfg["configurations"]},
            "external_servers": self.describe_external_servers(),
            "gemini_key": {"status": key_status()[0], "note": "existence only; the value is never read into any output"},
            "matrix": {"perspectives": cfg["perspectives"], "levels": cfg["levels"], "scenarios": cfg["scenarios"], "turns": cfg["turns"], "replicates": cfg.get("replicates", 1),
                       "configurations": sorted({s["configuration"] for s in self.sessions}), "include_optional": bool(cfg.get("include_optional")), "order": cfg.get("order", "configuration_major"),
                       "sessions": len(self.sessions), "session_order": [s["session_id"] for s in self.sessions]},
            "isolation": {"forbidden_ports": list(FORBIDDEN_PORTS), "harness_ports": {"real": 1236, "stub": 1237, "external_tunnel": TUNNEL_PORT}, "user_data": "inside this run folder (_scratch, deleted after each session)",
                          "owner_profile_touched": False, "network_policy": "Electron main cancels every request except file/data/blob and http://127.0.0.1:<allowed port>/ (allowed port = the configuration's server port: 1236 local model, 1237 stub, 1238 tunnel to an external server)"},
        }
        dump_json(os.path.join(self.dir, "config.json"), cfg_out)

    def external_confs(self):
        return [c for c in self.cfg["configurations"] if c["server"]["kind"] == "external" and any(x["configuration"] == c["id"] for x in self.sessions)]

    def external_server_for(self, conf):
        """ExternalServer for a configuration; required models = every model_name that configuration needs."""
        srv = conf["server"]
        return ExternalServer(srv["port"], required_models=[conf.get("model_name")], health_timeout=srv.get("health_timeout_s", 30),
                              request_timeout=srv.get("request_timeout_s", 5))

    def describe_external_servers(self):
        """One read-only GET /v1/models per external port (no chat request); recorded in config.json."""
        out = {}
        for c in self.external_confs():
            port = c["server"]["port"]
            if str(port) not in out:
                pr = ExternalServer(port, request_timeout=c["server"].get("request_timeout_s", 5)).probe()
                out[str(port)] = {"base_url": pr["base_url"], "reachable_at_start": pr["reachable"], "served_models_at_start": pr["models"], "error": pr["error"], "polled_at": now_iso(),
                                  "configurations": [], "note": "external server reached through a local tunnel port; the harness never starts, stops or signals it and never touches the tunnel; only GET /v1/models is used for health"}
            out[str(port)]["configurations"].append({"id": c["id"], "model_name": c.get("model_name"), "served_at_start": c.get("model_name") in out[str(port)]["served_models_at_start"]})
        return out

    def hash_model_files(self):
        out = {}
        for c in self.cfg["configurations"]:
            files = {}
            for label, path in (c.get("model_files") or {}).items():
                if not path or str(path).startswith("REPLACE_ME") or not os.path.exists(path):
                    files[label] = {"path": path, "status": "missing_or_placeholder"}
                    continue
                files[label] = hash_path(path, (c.get("hash_modes") or {}).get(label, c.get("hash_model_files", "stat")))
            out[c["id"]] = files
        return out

    # ------------------------------------------------------------------ server instances
    def record_pid(self, kind, info):
        p = os.path.join(self.dir, "logs", "pids.json")
        cur = json.load(open(p)) if os.path.exists(p) else []
        cur.append({"kind": kind, "pid": info["pid"], "pgid": info["pgid"], "argv": info["argv"], "started": info["start_time"]})
        dump_json(p, cur)

    def acquire_server(self, conf, fresh=False):
        cid = conf["id"]
        policy = conf.get("instance_policy", "fresh_per_session")
        if self.holder and (fresh or self.holder[0] != cid or policy == "fresh_per_session"):
            self.release_server(force=True)
        if self.holder:
            return self.holder[1], self.holder[2]
        srv = conf["server"]
        self.instance_counter += 1
        iid = "i%03d" % self.instance_counter
        if srv["kind"] == "external":
            # Connect only: poll /v1/models until it answers and lists this configuration's model_name. Nothing is spawned.
            ext = self.external_server_for(conf)
            info = ext.start()               # ExternalServerDown if the tunnel / server is not answering or lacks the model
            info["instance_id"] = iid
            self.holder = (cid, ext, info)
            self.log("external server %s connected (%s, serves %s, ready after %ss); the harness will not start or stop it" % (iid, info["base_url"], info["served_models"], info.get("healthy_after_s")))
            return ext, info
        extra = [a.replace("{run_scratch}", self.scratch) for a in srv.get("extra_args", [])]
        if srv["kind"] == "stub":
            cmd = make_stub_command(mode=srv.get("mode", "mixed"), log=os.path.join(self.dir, "logs", "stub_requests_%s.jsonl" % iid), extra=extra)
            health = 30
        else:
            cmd = [t.replace("{run_scratch}", self.scratch) for t in srv["command"]] + extra
            health = srv.get("health_timeout_s", 600)
        mgr = ServerManager(cmd, srv["port"], kind=srv["kind"], env=srv.get("env"), log_path=os.path.join(self.dir, "logs", "server_%s.log" % iid), health_timeout=health)
        info = mgr.start()
        info["instance_id"] = iid
        self.record_pid("server", info)
        self.holder = (cid, mgr, info)
        self.log("server %s started (pid %s, port %s, healthy after %ss)" % (iid, info["pid"], info["port"], info.get("healthy_after_s")))
        return mgr, info

    def release_server(self, force=False):
        if not self.holder:
            return None
        cid, mgr, info = self.holder
        conf = configuration(self.cfg, cid)
        if not force and conf.get("instance_policy") == "long_running":
            return None
        v = mgr.stop()
        self.holder = None
        if info.get("kind") == "external":
            # nothing to stop or verify; the remote process and the tunnel are deliberately left alone
            self.sessions_f.write({"event": "external_server_released", "instance_id": info["instance_id"], "configuration": cid, "base_url": info["base_url"], "result": v})
            self.log("external server %s released (remote process and tunnel left untouched)" % info["instance_id"])
            return v
        self.sessions_f.write({"event": "server_stopped", "instance_id": info["instance_id"], "configuration": cid, "verification": v, "age_s_at_stop": round(time.monotonic() - info["start_monotonic"], 1)})
        self.log("server %s stopped; verified gone: %s" % (info["instance_id"], v["ok"]))
        if not v["ok"]:
            raise AbortRun("server %s was NOT cleanly shut down: %s" % (info["instance_id"], v))
        return v

    # ------------------------------------------------------------------ driver
    @staticmethod
    def model_for(conf):
        """The exact model name the app must send: the configuration's model_name (e.g. 'base' / 'lora'), else its 'model' (old behaviour)."""
        return conf.get("model_name") or conf.get("model", "default_model")

    def start_driver(self, s, conf, instance, suffix=""):
        companion, scenario = s["_companion"], s["_scenario"]
        sess = ElectronSession(self.electron_bin, self.app_index, self.scratch, s["session_id"] + suffix, conf["server"]["port"],
                               stderr_path=os.path.join(self.dir, "logs", "electron_%s_a%d%s.stderr.log" % (s["session_id"], s["attempt"], suffix)))
        ready = sess.start()
        self.electron_version = ready.get("versions", {}).get("electron")
        if not self.isolation["probe_done"]:
            # an unlisted port that is NOT the allowed one (1238 is the tunnel port in external mode, so use 1239 then)
            probe_port = 1239 if int(conf["server"]["port"]) == TUNNEL_PORT else 1238
            probe = sess.request("probe_blocked", {"port": probe_port}, timeout=30)
            self.isolation.update({"probe_done": True, "off_port_probe": probe})
            self.sessions_f.write({"event": "isolation_probe", "result": probe})
            if "reached" in str(probe.get("verdict")) or probe.get("newlyBlocked") != 1:
                raise AbortRun("isolation self-test failed: the network filter did not block an unlisted port: %s" % probe)
        scribe = dict(self.cfg.get("scribe", {}))
        setup = {
            "allowedOrigin": "http://127.0.0.1:%d" % conf["server"]["port"],
            "endpoint": "http://127.0.0.1:%d" % conf["server"]["port"],
            "model": self.model_for(conf),
            "temperature": conf.get("temperature"), "maxTokens": conf.get("max_tokens"),
            "inject": {"temperature": conf.get("temperature"), "top_p": conf.get("top_p"), "max_tokens": conf.get("max_tokens"),
                       "model": self.model_for(conf) if conf.get("force_model", True) else None},
            "guardsOn": conf.get("guards", True),
            "perspective": s["perspective"], "tense": self.cfg["tense"], "level": s["level"],
            "player": companion["player"], "companion": companion["fields"], "preferences": companion["preferences"],
            "scribe": {"enabled": bool(scribe.get("enabled", True)), "every": scribe.get("every", 3)},
            "startRoom": scenario.get("start_room", "R001"), "rooms": scenario.get("rooms", {}),
            "appTimeoutMs": int(self.cfg["app_timeout_s"]) * 1000 if self.cfg.get("app_timeout_s") else None,
        }
        snap = sess.request("setup", setup, timeout=60)
        return sess, snap

    # ------------------------------------------------------------------ one turn
    def judge_ctx(self, s, turn, pose, reply, history_replies, bait_type):
        comp = s["_companion"]
        f = comp["fields"]
        return {"companion": {"name": f["name"], "pronoun": comp.get("pronoun", "he")}, "player": comp["player"], "persona_summary": comp["checks"]["persona_summary"],
                "important_notes": f.get("importantNotes", ""), "hard_rules": f.get("hardRules", ""), "preferences": comp["preferences"],
                "perspective": s["perspective"], "tense": self.cfg["tense"], "level": s["level"], "min_paragraphs": scoring.required_paragraphs(s["level"]),
                "beat": turn["beat"], "bait_type": bait_type, "previous_replies": history_replies[-2:], "pose_text": pose["text"], "reply": reply,
                "companion_checks": comp["checks"]}

    def play_turn(self, s, conf, instance, driver, idx, phase, pose=None, incident_id=None, state=None):
        scenario, companion = s["_scenario"], s["_companion"]
        turn = scenario["turns"][idx]
        app_cmds = []
        for cmd in turn.get("before", []):
            app_cmds.append(driver.request("app_command", {"command": cmd}, timeout=60))
        if pose is None:
            ctx = {"scenario": scenario, "turn": turn, "turn_index": idx, "n_turns": s["turns"], "perspective": s["perspective"], "level": s["level"],
                   "tense": self.cfg["tense"], "player": companion["player"], "companion": {"name": companion["fields"]["name"], "pronoun": companion.get("pronoun")},
                   "history": state["history"], "seed": seed_from(s["seed"], "bot", idx)}
            pose = self.bot.next_pose(ctx)          # BotError propagates to run_session
        ensure_pose_command(pose["command"])
        scribe = self.cfg.get("scribe", {})
        turn_seed = seed_from(s["seed"], "turn", idx)
        tmo = int(self.cfg.get("timeouts", {}).get("turn_s", 600))
        rec = driver.request("turn", {"command": pose["command"], "guardsOn": conf.get("guards", True), "seed": turn_seed if conf.get("inject_seed", True) else None,
                                      "waitScribe": bool(scribe.get("enabled", True) and scribe.get("wait", True)), "timeoutMs": tmo * 1000}, timeout=tmo + 180)
        self.total_turns += 1
        final, raw, sanit = rec.get("final_text"), rec.get("raw_text"), rec.get("sanitized_text")
        status = "ok" if final else "no_reply"
        info = instance
        age = round(time.monotonic() - info["start_monotonic"], 1)
        first_req = next((c for c in rec["calls"] if c["kind"] == "reply"), None)
        want_model = self.model_for(conf)
        models_sent = sorted({str((c.get("request") or {}).get("model")) for c in rec["calls"] if c["kind"] in ("reply", "rewrite", "scribe") and c.get("request")})
        model_sent = str((first_req["request"] or {}).get("model")) if first_req and first_req.get("request") else None
        guard_would = sorted({g["name"] for g in rec.get("guard_events", []) if g["real"]})
        tr = {"run_id": self.run_id, "session_id": s["session_id"], "attempt": s["attempt"], "phase": phase, "incident_id": incident_id, "turn_index": idx, "turn_id": turn["id"],
              "configuration": conf["id"], "guards_on": rec["guards_on"], "perspective": s["perspective"], "level": s["level"], "scenario": s["scenario"], "replicate": s["replicate"],
              "model_name": want_model, "model_sent": model_sent, "models_sent_all_calls": models_sent, "app_model_setting": rec.get("model_setting"),
              "companion": companion["fields"]["name"], "companion_id": companion["id"], "bait": bool(turn.get("bait")), "bait_type": turn.get("bait_type", ""), "app_commands": app_cmds,
              "player_pose": {"command": pose["command"], "kind": pose.get("kind"), "text": pose["text"], "meta": pose.get("meta")},
              "briefing": rec.get("briefing"), "request_body": first_req["request"] if first_req else None, "injected": first_req["injected"] if first_req else None,
              "raw_reply": raw, "sanitized_reply": sanit, "final_reply": final, "guard_rewrote": bool(rec["rewrite_count"] > 0 or (final and sanit and final != sanit)),
              "rewrite_count": rec["rewrite_count"], "guard_events": rec["guard_events"], "calls": [{k: c.get(k) for k in ("n", "kind", "path", "status", "ms", "error", "injected")} for c in rec["calls"]],
              "rewrite_texts": [c["text"] for c in rec["calls"] if c["kind"] == "rewrite"], "scribe_calls": [c for c in rec["calls"] if c["kind"] == "scribe"],
              "scribe_log": rec["scribe_log"], "scene": rec["scene"], "error": rec.get("error"), "log_delta": rec.get("log_delta"),
              "timings_ms": {"model_total": rec["ms_model"], "scribe_wait": rec["scribe_waited_ms"], "per_call": [c["ms"] for c in rec["calls"]]},
              "instance": {"instance_id": info["instance_id"], "start_time": info["start_time"], "age_s": age, "pid": info["pid"], "argv": info["argv"],
                           "kind": info.get("kind"), "base_url": info.get("base_url"), "served_models": info.get("served_models")},
              "time": now_iso(), "status": status}
        self.transcripts.write(tr)
        if conf.get("model_name") and any(m != conf["model_name"] for m in models_sent):
            # The comparison between configurations is meaningless if the wrong model name went over the wire: stop, do not continue.
            raise AbortRun("configuration %s must send model %r but the app sent %s (session %s turn %d); the harness's model injection did not take effect" % (
                conf["id"], conf["model_name"], models_sent, s["session_id"], idx + 1))
        # Rest between turns so the GPU is not pinned for minutes at a time (keeps the owner's Mac responsive).
        try:
            ratio = float((self.cfg.get("throttle") or {}).get("rest_ratio", 0) or 0)
            ms = rec.get("ms_model") or 0
            if ratio > 0 and ms:
                time.sleep(min(60.0, ratio * ms / 1000.0))
        except Exception:
            pass
        row = {"run_id": self.run_id, "session_id": s["session_id"], "attempt": s["attempt"], "phase": phase, "incident_id": incident_id or "", "configuration": conf["id"], "guards_on": int(rec["guards_on"]),
               "perspective": s["perspective"], "level": s["level"], "scenario": s["scenario"], "replicate": s["replicate"], "turn_index": idx, "turn_number": idx + 1, "turn_id": turn["id"],
               "bait": int(bool(turn.get("bait"))), "bait_type": turn.get("bait_type", ""), "model_name": want_model, "model_sent": model_sent, "status": status, "pose_kind": pose.get("kind"), "pose_chars": len(pose["text"]),
               "guard_rewrote": int(tr["guard_rewrote"]), "rewrite_count": rec["rewrite_count"], "guard_would_fire": ";".join(guard_would), "ms_model": rec["ms_model"],
               "instance_id": info["instance_id"], "instance_age_s": age}
        if final:
            scribe_meaningful = bool(scribe.get("enabled", True) and conf["server"]["kind"] in ("real", "external"))
            scored = scoring.score_turn(pose_text=pose["text"], pose_command=pose["command"], reply_final=final, reply_raw=raw, reply_sanitized=sanit,
                                        level=s["level"], perspective=s["perspective"], tense=self.cfg["tense"], player=companion["player"],
                                        companion={"name": companion["fields"]["name"], "pronoun": companion.get("pronoun", "he")}, checks=companion["checks"],
                                        briefing_user=(rec.get("briefing") or {}).get("user", ""), history_final=state["hist_replies"], bait_type=turn.get("bait_type"),
                                        scene=rec["scene"], stasis_expect=scenario.get("stasis_expect"), turn_number=idx + 1, scribe_enabled=scribe_meaningful,
                                        rewrite_texts=[c["text"] for c in rec["calls"] if c["kind"] == "rewrite"])
            for k, v in scored.items():
                if not k.startswith("_"):
                    row[k] = v
            row["reply_chars"] = len(final)
            self.transcripts.write({"event": "score_detail", "run_id": self.run_id, "session_id": s["session_id"], "attempt": s["attempt"], "phase": phase, "turn_index": idx,
                                    "head_hop_hits": scored["_hh_final"]["hits"], "head_hop_soft_hits": scored["_hh_final"]["soft_hits"], "raw_head_hop_hits": scored["_hh_raw"]["hits"], "stasis": scored["_stasis"]})
        if final and phase == "main":
            jctx = self.judge_ctx(s, turn, pose, final, state["hist_replies"], turn.get("bait_type"))
            try:
                j = self.judge.judge(jctx)
            except Exception as e:                       # a judge failure must never lose recorded turns
                j = {"status": "error", "error": "%s: %s" % (type(e).__name__, str(e)[:200]), "parsed": None, "raw": "", "model": getattr(self.judge, "name", "judge"), "prompt_sha256": "", "attempts": 0}
            if j["status"] != "none":
                self.judge_raw.write({"run_id": self.run_id, "session_id": s["session_id"], "attempt": s["attempt"], "turn_index": idx, "turn_id": turn["id"], "status": j["status"], "model": j.get("model"),
                                      "prompt_sha256": j.get("prompt_sha256"), "attempts": j.get("attempts"), "error": j.get("error"), "raw": j.get("raw"), "parsed": j.get("parsed")})
            row["judge_status"], row["judge_model"] = j["status"], j.get("model")
            if j.get("parsed"):
                p = j["parsed"]
                row.update({"judge_head_hop": p["head_hop"], "judge_head_hop_confidence": p["head_hop_confidence"], "judge_personality_lock": p["personality_lock"],
                            "judge_writing_quality": p["writing_quality"], "judge_rules_adherence": p["rules_adherence"], "judge_rules_ok": p["rules_ok"]})
        elif phase == "replay":
            row["judge_status"] = "skipped_replay"
        self._scores.writerow(row)
        self._scores_fh.flush()
        if final:
            state["hist_replies"].append(final)
        return pose, row, tr

    # ------------------------------------------------------------------ pause logic
    def elapsed_min(self):
        return (time.time() - self.t_start) / 60.0

    def should_not_start_session(self):
        if not self.max_minutes:
            return False
        est = 0.8 * (sum(self.session_durations) / len(self.session_durations) / 60.0) if self.session_durations else 0.0
        return self.elapsed_min() + est >= self.max_minutes

    def hard_limit_hit(self):
        return bool(self.max_minutes) and self.elapsed_min() >= self.max_minutes + self.hard_extra_minutes

    # ------------------------------------------------------------------ one session
    def run_session(self, s):
        t_session = time.time()
        conf = configuration(self.cfg, s["configuration"])
        s["_scenario"] = load_scenario(s["scenario"])
        cid = self.cfg.get("companion_override") or s["_scenario"]["companion"]
        s["_companion"] = load_companion(cid)
        s["attempt"] = self.attempts.get(s["session_id"], 0) + 1
        self.attempts[s["session_id"]] = s["attempt"]
        if s["turns"] > len(s["_scenario"]["turns"]):
            raise AbortRun("scenario %s has only %d turns" % (s["scenario"], len(s["_scenario"]["turns"])))
        inc_cfg = self.cfg.get("incidents", {"streak": 3, "basis": "raw", "replay": True})
        thr, basis = int(inc_cfg.get("streak", 3)), inc_cfg.get("basis", "raw")
        self.sessions_f.write({"event": "session_start", "session_id": s["session_id"], "attempt": s["attempt"], "time": now_iso()})
        summary = {"event": "session", "run_id": self.run_id, "session_id": s["session_id"], "attempt": s["attempt"], "configuration": conf["id"], "perspective": s["perspective"], "level": s["level"],
                   "scenario": s["scenario"], "replicate": s["replicate"], "guards_on": conf.get("guards", True), "turns_planned": s["turns"], "turns_done": 0,
                   "status": "running", "reason": "", "started": now_iso(), "replayed": False, "bad_session": False, "instance_ids": []}
        mgr = driver = None
        state = {"history": [], "hist_replies": []}
        poses_used, hop_flags = [], []
        try:
            self.log("session %s attempt %d (%d/%d)" % (s["session_id"], s["attempt"], s["index"], len(self.sessions)))
            mgr, instance = self.acquire_server(conf)
            summary["instance_ids"].append(instance["instance_id"])
            summary["instance_start_time"] = instance["start_time"]
            driver, snap = self.start_driver(s, conf, instance)
            summary["app_state_after_setup"] = {k: snap[k] for k in ("endpoint", "model", "narration", "level", "requiredParagraphs", "personaInjection")}
            idx = 0
            replay_done = False
            while idx < s["turns"]:
                if self.hard_limit_hit():
                    summary["status"], summary["reason"] = "interrupted", "hard time limit (max-minutes + %d) reached mid-session; it will be re-run from its first turn on resume" % self.hard_extra_minutes
                    self.status_pause = "paused (hard time limit reached mid-session; that session will be re-run on resume)"
                    break
                try:
                    pose, row, tr = self.play_turn(s, conf, instance, driver, idx, "main", state=state)
                except BotError as e:
                    summary["status"], summary["reason"] = "incomplete", "player bot failed: %s" % str(e)[:200]
                    self.bot_failures += 1
                    self.log("player bot failed in %s: %s (session kept, %d turns recorded)" % (s["session_id"], str(e)[:120], idx))
                    break
                if conf["server"]["kind"] == "external":
                    self.external_health_after_turn(mgr, conf, s, idx, tr)     # raises ExternalServerDown -> the run pauses
                poses_used.append(pose)
                state["history"].append({"pose": pose["text"], "reply": tr["final_reply"]})
                summary["turns_done"] = idx + 1
                flag = int(row.get("head_hop_raw" if basis == "raw" else "head_hop", 0) or 0)
                hop_flags.append(flag)
                idx += 1
                run_len = 0
                for f in reversed(hop_flags):
                    if not f:
                        break
                    run_len += 1
                if run_len == thr:   # the streak has just reached the threshold
                    summary["bad_session"] = True
                    instance, driver = self.handle_incident(s, conf, instance, driver, idx - 1, run_len, poses_used, state, inc_cfg, replay_done, summary)
                    replay_done = replay_done or bool(inc_cfg.get("replay", True))
            if summary["status"] == "running":
                summary["status"] = "complete"
        except KeyboardInterrupt:
            summary["status"], summary["reason"] = "interrupted", "stopped by the operator; will be re-run from its first turn on resume"
            raise
        except ExternalServerDown as e:
            self.note_external_down(s, conf, summary, str(e), idx_hint=summary["turns_done"], logged=getattr(e, "logged", False))
        except (DriverError, ServerError) as e:
            down = self.external_down_after_exception(mgr, conf)
            if down:                                # a driver timeout/crash while the tunnel is down is an outage, not a harness crash
                self.note_external_down(s, conf, summary, "%s: %s; %s" % (type(e).__name__, str(e)[:200], down), idx_hint=summary["turns_done"])
            else:
                summary["status"], summary["reason"] = "crashed", "%s: %s" % (type(e).__name__, str(e)[:300])
                self.log("session %s crashed: %s" % (s["session_id"], summary["reason"]))
        except AbortRun:
            summary["status"] = "aborted"
            raise
        except Exception as e:
            summary["status"], summary["reason"] = "crashed", "%s: %s" % (type(e).__name__, str(e)[:300])
            summary["traceback"] = traceback.format_exc()[-1500:]
            self.log("session %s crashed: %s" % (s["session_id"], summary["reason"]))
        finally:
            # 1. the summary is written FIRST (a stop during cleanup must not lose the session's verdict)
            if summary["status"] == "running":
                summary["status"], summary["reason"] = "interrupted", summary["reason"] or "stopped before the session finished; it will be re-run from its first turn on resume"
            summary["ended"] = now_iso()
            summary["head_hop_flags_basis"] = basis
            summary["head_hop_flags"] = hop_flags
            summary["blocked_requests"] = None
            if driver:
                try:
                    summary["blocked_requests"] = [b["url"] for b in driver.request("blocked", timeout=5)["blocked"]]
                except BaseException:
                    pass
            self.sessions_f.write(summary)
            s["_status"] = summary["status"]
            # 2. cleanup, each step guaranteed to be attempted
            closed = {"event": "session_closed", "session_id": s["session_id"], "attempt": s["attempt"], "electron_shutdown_ok": None}
            try:
                if driver:
                    try:
                        closed["electron_shutdown_ok"] = bool(driver.close().get("ok"))
                    except Exception as e:
                        closed["electron_shutdown_ok"] = False
                        closed["electron_shutdown_error"] = str(e)[:200]
            finally:
                self.sessions_f.write(closed)
                shutil.rmtree(os.path.join(self.scratch, "session_" + s["session_id"]), ignore_errors=True)
                self.session_durations.append(time.time() - t_session)
                self.release_server()

    # ------------------------------------------------------------------ external server health
    @staticmethod
    def turn_looks_failed(tr):
        calls = [c for c in (tr.get("calls") or []) if c.get("kind") in ("reply", "rewrite", "scribe")]
        return (tr.get("status") != "ok" or bool(tr.get("error")) or
                any(c.get("error") or c.get("status") is None or int(c.get("status") or 0) >= 500 for c in calls))

    def external_health_after_turn(self, mgr, conf, s, idx, tr):
        """After every turn on an external server: one /v1/models probe (no chat). A turn that looked failed gets a grace period
        of retries first (a brief tunnel hiccup is not an outage). If the server is still not answering, raise ExternalServerDown:
        the session is recorded as interrupted and the run pauses, resumable."""
        grace = float(conf["server"].get("health_recheck_s", 30)) if self.turn_looks_failed(tr) else 0.0
        h = mgr.check(grace_s=grace)
        if h["ok"]:
            return h
        self.sessions_f.write({"event": "external_server_failure", "time": now_iso(), "session_id": s["session_id"], "attempt": s["attempt"], "configuration": conf["id"],
                               "turn_index": idx, "turn_id": tr["turn_id"], "turn_status": tr["status"], "turn_error": tr.get("error"), "base_url": mgr.base_url,
                               "health_error": h["error"], "served_models_now": h["models"], "grace_s": grace,
                               "call_errors": [{k: c.get(k) for k in ("kind", "status", "error", "ms")} for c in (tr.get("calls") or []) if c.get("error") or not c.get("status") or int(c.get("status") or 0) >= 400]})
        err = ExternalServerDown("external server %s stopped answering after turn %d of %s: %s" % (mgr.base_url, idx + 1, s["session_id"], h["error"]))
        err.logged = True
        raise err

    def external_down_after_exception(self, mgr, conf):
        """For a DriverError/ServerError inside an external session: None if the server is fine (a real harness problem), else a reason."""
        if conf["server"]["kind"] != "external":
            return None
        try:
            h = (mgr if isinstance(mgr, ExternalServer) else self.external_server_for(conf)).check(grace_s=min(15.0, float(conf["server"].get("health_recheck_s", 30))))
        except Exception as e:
            return "health probe failed: %s" % e
        return None if h["ok"] else "external server not answering: %s" % h["error"]

    def note_external_down(self, s, conf, summary, reason, idx_hint=0, logged=False):
        summary["status"] = "interrupted"
        summary["reason"] = "external server problem: %s; the session will be re-run from its first turn (new attempt) on resume" % reason[:300]
        summary["external_failure"] = True
        if not logged:
            self.sessions_f.write({"event": "external_server_failure", "time": now_iso(), "session_id": s["session_id"], "attempt": s["attempt"], "configuration": conf["id"],
                                   "turn_index": idx_hint, "health_error": reason[:300], "note": "surfaced at session start or outside a turn"})
        self.status_pause = "paused (external server problem: %s; re-open the tunnel / check the server, then: python -m harness resume --run %s --i-have-permission)" % (reason[:160], self.dir)
        self.log("EXTERNAL SERVER PROBLEM in %s: %s -- pausing the run (resumable)" % (s["session_id"], reason[:200]))

    # ------------------------------------------------------------------ incidents
    def handle_incident(self, s, conf, instance, driver, end_idx, run_len, poses_used, state, inc_cfg, already_replayed, summary):
        thr = int(inc_cfg.get("streak", 3))
        start_idx = end_idx - run_len + 1
        iid = "inc%03d" % (len(self.incidents) + 1)
        recent = [r for r in read_jsonl(os.path.join(self.dir, "transcripts.jsonl"))
                  if r.get("session_id") == s["session_id"] and r.get("attempt") == s["attempt"] and r.get("phase") == "main" and not r.get("event")]
        streak_turns = [r for r in recent if start_idx <= r["turn_index"] <= end_idx]
        first = streak_turns[0]
        details = [{"turn_index": r["turn_index"], "turn_id": r["turn_id"], "player_pose": r["player_pose"]["command"][:600], "raw_reply": r["raw_reply"], "final_reply": r["final_reply"]} for r in streak_turns]
        age_at_start = first["instance"]["age_s"]
        inc = {"incident_id": iid, "run_id": self.run_id, "session_id": s["session_id"], "attempt": s["attempt"], "detected_at": now_iso(), "streak_threshold": thr, "streak_basis": inc_cfg.get("basis", "raw"),
               "streak_start_turn_index": start_idx, "streak_len_at_detection": run_len, "streak_turns": details,
               "instance": {"instance_id": instance["instance_id"], "start_time": instance["start_time"], "age_s_at_streak_start": age_at_start, "pid": instance["pid"], "server_args": instance["argv"], "port": instance["port"], "kind": instance["kind"]},
               "briefing_sent_first_streak_turn": first["briefing"], "request_body_first_streak_turn": first["request_body"],
               "config": {"configuration": conf["id"], "guards_on": conf.get("guards", True), "perspective": s["perspective"], "level": s["level"], "scenario": s["scenario"], "model": conf.get("model")}}
        shared = []
        for prev in self.incidents:
            same = [k for k in ("configuration", "perspective", "level", "scenario", "guards_on") if prev["config"].get(k) == inc["config"].get(k)]
            if prev["instance"].get("instance_id") != instance["instance_id"] and same:
                shared.append({"with": prev["incident_id"], "same": same, "streak_start_turn_index": [prev["streak_start_turn_index"], start_idx],
                               "instance_age_s_at_streak_start": [prev["instance"]["age_s_at_streak_start"], age_at_start]})
        inc["in_common_with_earlier"] = shared
        replay_res = {"attempted": False}
        new_instance, new_driver = instance, driver
        if conf["server"]["kind"] == "external":
            # A fresh restart means a new server process; the harness cannot (and must not) restart a remote server.
            inc["cleared_by_fresh_restart"] = None
            replay_res["reason"] = "external server: the harness never restarts it, so no fresh-restart replay was done (the incident is recorded only)"
        elif inc_cfg.get("replay", True) and not already_replayed:
            replay_res = {"attempted": True, "restart_note": "server and app profile restarted fresh; poses 0..%d replayed once" % end_idx}
            self.log("INCIDENT %s in %s: %d consecutive head-hops from turn %d; restarting instance and replaying" % (iid, s["session_id"], run_len, start_idx + 1))
            driver.close()
            _, new_instance = self.acquire_server(conf, fresh=True)
            summary["instance_ids"].append(new_instance["instance_id"])
            summary["replayed"] = True
            new_driver, _ = self.start_driver(s, conf, new_instance, suffix="_replay")
            rstate = {"history": [], "hist_replies": []}
            flags = []
            for k in range(end_idx + 1):
                _, row, tr = self.play_turn(s, conf, new_instance, new_driver, k, "replay", pose=poses_used[k], incident_id=iid, state=rstate)
                flags.append(int(row.get("head_hop_raw" if inc_cfg.get("basis", "raw") == "raw" else "head_hop", 0) or 0))
            longest = cur = 0
            for f in flags[start_idx:]:
                cur = cur + 1 if f else 0
                longest = max(longest, cur)
            replay_res.update({"new_instance": {"instance_id": new_instance["instance_id"], "start_time": new_instance["start_time"], "pid": new_instance["pid"], "server_args": new_instance["argv"]},
                               "replayed_turn_indexes": list(range(end_idx + 1)), "replay_hop_flags": flags, "replay_streak_window_flags": flags[start_idx:end_idx + 1],
                               "longest_hop_run_in_replay_window": longest, "cleared": longest < thr})
            inc["cleared_by_fresh_restart"] = replay_res["cleared"]
        else:
            inc["cleared_by_fresh_restart"] = None
            replay_res["reason"] = "replay disabled or already used once in this session"
        inc["replay"] = replay_res
        self.incidents.append(inc)
        self.incidents_f.write(inc)
        return new_instance, new_driver

    # ------------------------------------------------------------------ checkpoint (paused runs) and seal (finished runs)
    def write_checkpoint(self):
        files = {}
        for rel in APPEND_ONLY_FILES:
            p = os.path.join(self.dir, rel)
            if os.path.exists(p):
                files[rel] = {"size": os.path.getsize(p), "sha256": sha256_file(p)}
        path = os.path.join(self.dir, "CHECKPOINT_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
        dump_json(path, {"written": now_iso(), "status": self.status, "note": "append-only files at pause time; a resume verifies these byte prefixes are unchanged", "files": files})
        return path

    @staticmethod
    def verify_checkpoints(run_dir):
        """Each CHECKPOINT_*.json: the first `size` bytes of every append-only file must still hash to the recorded value."""
        import glob
        problems = []
        for cp in sorted(glob.glob(os.path.join(run_dir, "CHECKPOINT_*.json"))):
            for rel, meta in load_json(cp)["files"].items():
                p = os.path.join(run_dir, rel)
                if not os.path.exists(p) or os.path.getsize(p) < meta["size"]:
                    problems.append("%s: %s missing or shorter than at checkpoint" % (os.path.basename(cp), rel))
                    continue
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    left = meta["size"]
                    while left:
                        b = f.read(min(1 << 20, left))
                        if not b:
                            break
                        h.update(b)
                        left -= len(b)
                if h.hexdigest() != meta["sha256"]:
                    problems.append("%s: %s was modified since the checkpoint" % (os.path.basename(cp), rel))
        return problems

    # ------------------------------------------------------------------ run
    def preflight(self):
        ext_ports = {int(c["server"]["port"]) for c in self.external_confs()}
        for p in (1236, 1237):
            if p in ext_ports:
                continue                  # an external server is EXPECTED to answer on its port (it is not ours; reachability is checked per session)
            if port_in_use(p):
                raise AbortRun("port %d is already in use before the run started; refusing to continue (stale server from an earlier crash? see: python -m harness cleanup --run <folder>)" % p)
        for conf in self.cfg["configurations"]:
            if conf["server"]["kind"] == "real" and any(s["configuration"] == conf["id"] for s in self.sessions):
                missing = [p for p in required_paths([t.replace("{run_scratch}", self.scratch) for t in conf["server"]["command"]]) if not os.path.exists(p)]
                if missing:
                    raise AbortRun("configuration %s: required path(s) missing (drive not mounted?): %s" % (conf["id"], missing))
        if stray_processes(1236):
            raise AbortRun("a process that looks like a model server on port 1236 is already running: %s" % stray_processes(1236))

    def execute(self):
        from . import report
        t0 = time.time()
        if self.nice:
            try:
                os.nice(int(self.nice))
                self.log("running at nice +%s (child processes inherit it)" % self.nice)
            except OSError as e:
                self.log("could not lower priority: %s" % e)
        try:
            if self.resuming:
                problems = self.verify_checkpoints(self.dir)
                if problems:
                    raise AbortRun("append-only check failed; refusing to resume: %s" % problems)
                self.resume_f.write({"event": "resume", "time": now_iso(), "completed_sessions": len(self.done), "planned": len(self.sessions), "max_minutes": self.max_minutes, "nice": self.nice})
            else:
                self.write_config()
            self.preflight()
            todo = [s for s in self.sessions if s["session_id"] not in self.done]
            self.log("run %s: %d sessions planned, %d already complete, %d to do" % (self.run_id, len(self.sessions), len(self.done), len(todo)))
            consecutive = 0
            stopped_early = False
            for s in todo:
                if self.should_not_start_session():
                    left = len([x for x in todo if x["session_id"] not in self.done])
                    self.status = "paused (max-minutes %s reached; %d sessions left; resume to continue)" % (self.max_minutes, left)
                    self.log(self.status)
                    stopped_early = True
                    break
                before = self.bot_failures
                self.run_session(s)
                if s.get("_status") == "complete":
                    self.done.add(s["session_id"])
                if self.status_pause:
                    self.status = self.status_pause
                    stopped_early = True
                    break
                consecutive = consecutive + 1 if self.bot_failures > before else 0
                if consecutive >= 3:
                    raise AbortRun("player bot failed in 3 sessions in a row; stopping the run (all recorded turns are kept)")
            if not stopped_early:
                self.status = "complete" if all(s["session_id"] in self.done for s in self.sessions) else "finished with incomplete sessions (resume to retry them)"
        except KeyboardInterrupt:
            self.status = "interrupted (resume with: python -m harness resume --run %s)" % self.dir
        except AbortRun as e:
            self.status = "aborted: %s" % e
            self.log("RUN ABORTED: %s" % e)
        finally:
            try:
                self.release_server(force=True)
            except Exception as e:
                self.log("server shutdown problem: %s" % e)
                self.status += " (server shutdown problem: %s)" % e
            for f in (self.transcripts, self.judge_raw, self.incidents_f, self.sessions_f):
                f.close()
            self._scores_fh.close()
            shutil.rmtree(self.scratch, ignore_errors=True)
            elapsed = round(time.time() - t0, 1)
            self.resume_f.write({"event": "invocation_end", "time": now_iso(), "elapsed_s": elapsed, "status": self.status})
            self.resume_f.close()
            sealed = self.status == "complete"
            post = {"port_1236_free": not port_in_use(1236), "port_1237_free": not port_in_use(1237), "stray_model_processes": stray_processes(1236)}
            ext = self.external_confs()
            if ext:
                # Prove the harness itself left nothing running, and that it did not close the tunnel: the external port is never
                # touched by us, so whatever answered before still answers (or not) independently of this run.
                post["harness_descendant_processes"] = harness_descendants()
                post["harness_electron_processes"] = harness_electron_processes()
                post["no_stray_local_processes"] = not post["harness_descendant_processes"] and not post["harness_electron_processes"]
                post["external_ports"] = {}
                for port in sorted({int(c["server"]["port"]) for c in ext}):
                    pr = ExternalServer(port).probe()
                    post["external_ports"][str(port)] = {"still_answering": pr["reachable"], "served_models": pr["models"], "harness_closed_it": False,
                                                         "note": "the harness only ever sent GET /v1/models (and the app's chat requests); it never started, stopped or signalled the server or the tunnel"}
            dump_json(os.path.join(self.dir, "run_end.json" if sealed else "run_status.json"),
                      {"run_id": self.run_id, "status": self.status, "ended": now_iso(), "elapsed_s_this_invocation": elapsed, "elapsed_s_total": round(self.prior_elapsed + elapsed, 1),
                       "turns_recorded": self.total_turns, "incidents": len(self.incidents), "sessions_planned": len(self.sessions), "sessions_complete": len(self.done),
                       "electron_version": self.electron_version, "isolation": self.isolation, "post_run_checks": post})
            try:
                report.build_report(self.dir)
                from . import labels
                labels.write_label_sheet(self.dir, self.cfg.get("labels", {}))
            except Exception:
                self.log("report/labels generation failed:\n" + traceback.format_exc())
            if sealed:
                from .manifest import write_manifest
                write_manifest(self.dir)
            else:
                cp = self.write_checkpoint()
                self.log("checkpoint written: %s (run NOT sealed; continue with: python -m harness resume --run %s)" % (os.path.basename(cp), self.dir))
        return self.dir
