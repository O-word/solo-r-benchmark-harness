"""Command line: python -m harness <command>

  dryrun                       whole pipeline against the stub server / stub bots; verifies the result; exit 0 = pass
  run --config FILE [--max-minutes N] [--nice] [--with-guards-off] [--i-have-permission]
                               a configured run (a REAL run needs --i-have-permission)
  resume --run DIR [--max-minutes N] [--nice]   continue an interrupted/paused run in the same folder
  status --run DIR             how far a run got; finalize --run DIR seals a partial run; cleanup --run DIR stops recorded servers after a hard crash
  check --config FILE          pre-flight only: files, snapshot hash, ports, Gemini key presence (never its value); for an external-server
                               config: does the tunnel port answer, which model names are served (GET /v1/models only, no chat request)
  verify --run DIR             recompute MANIFEST.sha256 and compare
  agreement --run DIR --labels FILE   judge/detector agreement with the hand labels (written as an addendum)
  export-incidents --run DIR   markdown rows for the project's BAD_INSTANCE_LOG.md (not written automatically)
"""
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from . import __version__
from .config import ConfigError, load_config
from .gemini_client import key_status
from .manifest import verify_manifest
from .server_manager import ExternalServer, port_in_use
from .util import HARNESS_ROOT, STUB_PORT, read_jsonl


def _config_path(p):
    return p if os.path.isabs(p) else os.path.join(HARNESS_ROOT, p)


def _install_signal_handlers():
    import signal

    def _stop(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _stop)


def needs_permission(cfg):
    """A REAL run (local model) or an EXTERNAL run through the tunnel needs --i-have-permission. An external config that only
    points at the stub port (tests, dry run) does not."""
    if cfg["mode"] == "real":
        return True
    if cfg["mode"] == "external":
        return any(int(c["server"]["port"]) != STUB_PORT for c in cfg["configurations"])
    return False


def check_external(cfg, sessions):
    """Report on every external server a config would use: does the port answer, which model names are served, are the
    configurations' model names among them. GET /v1/models only; no chat request is ever sent. Returns True if all is well."""
    ok = True
    by_port = {}
    for c in cfg["configurations"]:
        if c["server"]["kind"] == "external" and any(s["configuration"] == c["id"] for s in sessions):
            by_port.setdefault(int(c["server"]["port"]), []).append(c)
    for port, confs in sorted(by_port.items()):
        pr = ExternalServer(port).probe()
        print("external server http://127.0.0.1:%d (tunnel port): %s" % (port, "ANSWERS" if pr["reachable"] else "NOT ANSWERING (%s)" % pr["error"]))
        if pr["reachable"]:
            print("  served model names:", ", ".join(pr["models"]) or "(none listed)")
        else:
            print("  start the SSH tunnel to the remote server (local end 127.0.0.1:%d) and run check again" % port)
        ok &= pr["reachable"]
        for c in confs:
            want = c.get("model_name")
            here = pr["reachable"] and want in pr["models"]
            print("  [%s] configuration %s sends model_name %r" % ("ok" if here else "MISSING" if pr["reachable"] else "unknown", c["id"], want))
            ok &= bool(here)
    print("no chat request was sent: this check only read /v1/models")
    return ok


def cmd_check(args):
    from .config import expand_sessions
    from .server_manager import required_paths, stray_processes
    cfg = load_config(_config_path(args.config))
    if getattr(args, "with_guards_off", False):
        cfg["include_optional"] = True
    ok = True
    sessions = expand_sessions(cfg)
    print("config ok:", cfg["name"], "| mode", cfg["mode"], "| sessions", len(sessions), "| turns", sum(s["turns"] for s in sessions))
    print("electron binary exists:", os.path.isfile(cfg["electron_bin"]))
    ok &= os.path.isfile(cfg["electron_bin"])
    has_local = any(c["server"]["kind"] != "external" and any(s["configuration"] == c["id"] for s in sessions) for c in cfg["configurations"])
    if has_local:
        print("ports 1236/1237 free:", not port_in_use(1236), not port_in_use(1237))
        print("stray model processes on 1236:", stray_processes(1236) or "none")
    ks = key_status()
    print("gemini key:", ks[0], "-", ks[1])
    for c in cfg["configurations"]:
        if not any(s["configuration"] == c["id"] for s in sessions):
            continue
        toks = c["server"].get("command", [])
        ph = [t for t in toks if str(t).startswith("REPLACE_ME")]
        if ph:
            print("configuration %s still has %d placeholder(s) in its server command" % (c["id"], len(ph)))
            ok &= cfg["mode"] == "stub"
        if c["server"]["kind"] == "real":
            for p in required_paths(toks):
                ex = os.path.exists(p)
                print("  [%s] %s" % ("ok" if ex else "MISSING", p))
                ok &= ex
    if any(c["server"]["kind"] == "external" for c in cfg["configurations"]):
        ok &= check_external(cfg, sessions)
    needs_key = cfg.get("player_bot", {}).get("kind") == "gemini" or cfg.get("judge", {}).get("kind") == "gemini"
    print("player bot: %s | judge: %s" % (cfg.get("player_bot", {}).get("kind"), cfg.get("judge", {}).get("kind")))
    if needs_key and ks[0] == "missing":
        print("this config needs a Gemini key and none is available")
        ok = False
    return 0 if ok else 1


def cmd_run(args):
    from .runner import Run
    path = _config_path(args.config)
    cfg = load_config(path)
    if args.with_guards_off:
        cfg["include_optional"] = True
    for c in cfg["configurations"]:
        if any(str(t).startswith("REPLACE_ME") for t in c["server"].get("command", [])):
            print("refusing: configuration %s still contains REPLACE_ME placeholders in its server command" % c["id"])
            return 2
    if needs_permission(cfg) and not args.i_have_permission:
        print("refusing a REAL/EXTERNAL run without --i-have-permission (a real run starts a real model server, an external run sends chat requests through the tunnel to a remote server; only after the owner says go)")
        return 2
    if json.dumps(cfg).count("REPLACE_ME"):
        print("refusing: config still contains REPLACE_ME placeholders")
        return 2
    _install_signal_handlers()
    nice = args.nice if args.nice is not None else cfg.get("nice")
    run = Run(cfg, path, name_suffix=args.suffix or "", max_minutes=args.max_minutes, nice=nice, hard_extra_minutes=args.hard_extra_minutes)
    d = run.execute()
    print("run folder:", d, "\nstatus:", run.status)
    return 0


def cmd_resume(args):
    from .runner import AbortRun, Run
    _install_signal_handlers()
    try:
        run = Run(None, None, resume_dir=args.run, max_minutes=args.max_minutes, nice=args.nice, hard_extra_minutes=args.hard_extra_minutes)
    except AbortRun as e:
        print("cannot resume:", e)
        return 2
    cfg = run.cfg
    if getattr(args, "rest_ratio", None) is not None:
        cfg["throttle"] = {"rest_ratio": args.rest_ratio}
        run.resume_f.write({"event": "rest_ratio_override", "rest_ratio": args.rest_ratio})
    if needs_permission(cfg) and not args.i_have_permission:
        print("refusing to resume a REAL/EXTERNAL run without --i-have-permission")
        return 2
    d = run.execute()
    print("run folder:", d, "\nstatus:", run.status)
    return 0


def cmd_status(args):
    from . import runview
    d = os.path.abspath(args.run)
    ses = runview.load_sessions(d)
    cfgj = json.load(open(os.path.join(d, "config.json")))
    planned = cfgj["matrix"]["session_order"]
    v = runview.valid_attempts(ses)
    complete = sorted({s["session_id"] for s in ses if s.get("status") == "complete"})
    print("run:", cfgj["run_id"], "| sealed:", os.path.exists(os.path.join(d, "MANIFEST.sha256")))
    print("sessions planned %d | complete %d | counted %d | attempts recorded %d" % (len(planned), len(complete), len(v), len(ses)))
    for name in ("run_end.json", "run_status.json"):
        if os.path.exists(os.path.join(d, name)):
            e = json.load(open(os.path.join(d, name)))
            print(name, "->", e.get("status"), "| turns recorded", e.get("turns_recorded"), "| elapsed_s_total", e.get("elapsed_s_total"))
    todo = [x for x in planned if x not in complete]
    print("still to do:", len(todo), todo[:4], "..." if len(todo) > 4 else "")
    return 0


def cmd_finalize(args):
    """Seal a paused/aborted run as it stands (partial results). Writes the final MANIFEST.sha256; the run can no longer be resumed."""
    from . import labels, report
    from .manifest import write_manifest
    d = os.path.abspath(args.run)
    if os.path.exists(os.path.join(d, "MANIFEST.sha256")):
        print("already sealed")
        return 1
    report.build_report(d)
    labels.write_label_sheet(d, json.load(open(os.path.join(d, "config.json")))["config"].get("labels", {}))
    n = write_manifest(d)
    print("sealed %s with %d files in MANIFEST.sha256 (partial run: unfinished sessions are not counted)" % (d, n))
    return 0


def cmd_cleanup(args):
    """After a hard crash: stop harness-started servers recorded in logs/pids.json IF the process still matches what we started."""
    import signal
    d = os.path.abspath(args.run)
    p = os.path.join(d, "logs", "pids.json")
    if not os.path.exists(p):
        print("no pids.json")
        return 0
    killed = 0
    for rec in json.load(open(p)):
        pid, pgid = rec["pid"], rec["pgid"]
        try:
            cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            cmd = ""
        if cmd and rec["argv"][0] in cmd and ("--port" in " ".join(rec["argv"])):
            print("stopping pid %d (process group %d): %s" % (pid, pgid, cmd[:100]))
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                time.sleep(2)
            killed += 1
    import time as _t
    print("stopped %d recorded server(s); ports 1236/1237 free: %s %s" % (killed, not port_in_use(1236), not port_in_use(1237)))
    return 0


def _procs_left():
    """Any process still running from this harness (electron driver or stub server)?"""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return []
    keys = [os.path.join(HARNESS_ROOT, "driver", "electron_main.cjs"), os.path.join(HARNESS_ROOT, "harness", "stub_server.py")]
    return [l.strip()[:160] for l in out.splitlines() if any(k in l for k in keys) and "ps -axo" not in l]


def _spawn_external_stub(log=None, die_after=None, port=STUB_PORT):
    """An already-running 'external server' for the dry run / tests: the stub on port 1237 serving two model names, started by the
    dry-run command itself (never by a Run). Returns the Popen."""
    from .server_manager import make_stub_command
    extra = ["--model-id", "base,lora", "--strict-models"] + (["--die-after", str(die_after)] if die_after is not None else [])
    cmd = [str(c).replace("{port}", str(port)) for c in make_stub_command(mode="mixed", extra=extra, log=log)]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 20:
        if ExternalServer(port).probe()["reachable"]:
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    _kill_stub(proc)
    raise RuntimeError("external stub did not start")


def _kill_stub(proc):
    import signal
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _electron_left():
    from .server_manager import harness_electron_processes
    return harness_electron_processes()


def dryrun_external_phase(check, py, env, runs_dir):
    """Phase D: the EXTERNAL-server path against an already-running stub on 1237 (two model names). The run never starts or stops it."""
    from . import runview
    from .config import expand_sessions, load_config as _lc
    ext_cfg = os.path.join("configs", "dryrun_external.json")
    cfg = _lc(_config_path(ext_cfg))
    planned = expand_sessions(cfg)
    tmp = tempfile.mkdtemp(prefix="hf_ext_stub_")
    stublog = os.path.join(tmp, "stub.jsonl")

    def newest(suffix):
        c = sorted(glob.glob(os.path.join(runs_dir, "*_dryrun_external" + suffix)))
        return c[-1] if c else None

    def chat_lines():
        return len(read_jsonl(stublog)) if os.path.exists(stublog) else 0
    stub = None
    try:
        # -- D1: healthy external server, full run
        stub = _spawn_external_stub(log=stublog)
        before = chat_lines()
        rc = subprocess.run(py + ["check", "--config", ext_cfg], cwd=HARNESS_ROOT, capture_output=True, text=True, timeout=120, env=env)
        check("external check: tunnel port answers, served names listed, no chat request sent", rc.returncode == 0 and "ANSWERS" in rc.stdout and "base, lora" in rc.stdout and chat_lines() == before, rc.stdout[-300:])
        r1 = subprocess.run(py + ["run", "--config", ext_cfg, "--suffix", "_ext"], cwd=HARNESS_ROOT, capture_output=True, text=True, timeout=900, env=env)
        d1 = newest("_ext")
        check("external run completed and sealed (stub started by the dry run, not by the run)", d1 and os.path.exists(os.path.join(d1, "MANIFEST.sha256")) and verify_manifest(d1)["ok"] and "status: complete" in r1.stdout, r1.stdout[-200:])
        if d1:
            tr = [r for r in read_jsonl(os.path.join(d1, "transcripts.jsonl")) if not r.get("event") and r["phase"] == "main"]
            check("every request carried the configuration's exact model name (recorded per transcript row)", tr and all(t["model_sent"] == t["model_name"] and set(t["models_sent_all_calls"]) == {t["model_name"]} for t in tr) and {t["model_name"] for t in tr} == {"base", "lora"}, "%d turns" % len(tr))
            logged = [r.get("model") for r in read_jsonl(stublog) if r.get("path") == "/v1/chat/completions"]
            check("the stub server itself saw both 'base' and 'lora' requests", {"base", "lora"} <= set(logged), str({m: logged.count(m) for m in set(logged)}))
            seq = [x["configuration"] for x in planned]
            pairs = [(planned[i], planned[i + 1]) for i in range(0, len(planned), 2)]
            firsts = [a["configuration"] for a, b in pairs]
            check("sessions alternate BASE/LORA by cell (paired, first position rotates)", all((a["perspective"], a["level"], a["scenario"]) == (b["perspective"], b["level"], b["scenario"]) and a["configuration"] != b["configuration"] for a, b in pairs)
                  and firsts.count("ext_base") == firsts.count("ext_lora"), str([c[4:] for c in seq]))
            cj = json.load(open(os.path.join(d1, "config.json")))
            check("config.json records the external base URL and the served model names", cj["external_servers"].get("1237", {}).get("served_models_at_start") == ["base", "lora"] and cj["model_names"] == {"ext_base": "base", "ext_lora": "lora"}, str(cj["external_servers"])[:160])
            ev = read_jsonl(os.path.join(d1, "sessions.jsonl"))
            check("external server was never stopped or verified-gone (no server_stopped, no pids recorded), only released", not [e for e in ev if e.get("event") == "server_stopped"] and not os.path.exists(os.path.join(d1, "logs", "pids.json")) and len([e for e in ev if e.get("event") == "external_server_released"]) == len(planned))
            pc = json.load(open(os.path.join(d1, "run_end.json")))["post_run_checks"]
            check("harness left no stray local processes and did not close the external port", pc.get("no_stray_local_processes") and pc["external_ports"]["1237"]["still_answering"] and stub.poll() is None and ExternalServer(STUB_PORT).probe()["reachable"], str(pc.get("external_ports"))[:160])
            check("no Electron driver left running after the external run", not _electron_left())
        stub_alive_after = stub.poll() is None
        check("the external stub was still running after the run (the run does not stop it)", stub_alive_after)
        _kill_stub(stub)
        stub = None
        rc = subprocess.run(py + ["check", "--config", ext_cfg], cwd=HARNESS_ROOT, capture_output=True, text=True, timeout=120, env=env)
        check("external check with the server down: reports NOT ANSWERING and fails", rc.returncode != 0 and "NOT ANSWERING" in rc.stdout, rc.stdout[-200:])

        # -- D2: the external server dies mid-run -> the run pauses (no crash), the failure is recorded, then it resumes
        stub = _spawn_external_stub(die_after=9)
        r2 = subprocess.run(py + ["run", "--config", ext_cfg, "--suffix", "_outage"], cwd=HARNESS_ROOT, capture_output=True, text=True, timeout=900, env=env)
        d2 = newest("_outage")
        st = json.load(open(os.path.join(d2, "run_status.json"))) if d2 and os.path.exists(os.path.join(d2, "run_status.json")) else {}
        check("outage: server died mid-run, run paused (not crashed, not sealed), checkpoint written", "paused (external server" in st.get("status", "") and not os.path.exists(os.path.join(d2, "MANIFEST.sha256")) and glob.glob(os.path.join(d2, "CHECKPOINT_*.json")) and r2.returncode == 0, st.get("status", r2.stdout[-200:])[:120])
        ev2 = read_jsonl(os.path.join(d2, "sessions.jsonl")) if d2 else []
        fail = [e for e in ev2 if e.get("event") == "external_server_failure"]
        check("outage: the failure is recorded (event with session, turn and health error)", len(fail) == 1 and fail[0].get("health_error") and fail[0].get("session_id"), str(fail)[:140])
        check("outage: harness left no stray processes", st.get("post_run_checks", {}).get("no_stray_local_processes") and not _electron_left())
        _kill_stub(stub)
        stub = None
        r3 = subprocess.run(py + ["resume", "--run", d2], cwd=HARNESS_ROOT, capture_output=True, text=True, timeout=300, env=env)
        st3 = json.load(open(os.path.join(d2, "run_status.json")))
        check("outage: resume while the server is still down pauses again (no crash)", "paused (external server" in st3["status"] and not os.path.exists(os.path.join(d2, "MANIFEST.sha256")) and r3.returncode == 0, st3["status"][:100])
        stub = _spawn_external_stub()
        r4 = subprocess.run(py + ["resume", "--run", d2], cwd=HARNESS_ROOT, capture_output=True, text=True, timeout=900, env=env)
        ses = runview.load_sessions(d2)
        cut = [x for x in ses if x["status"] == "interrupted" and x.get("external_failure")]
        check("outage: after the server is back, resume finishes every session and seals the run", os.path.exists(os.path.join(d2, "MANIFEST.sha256")) and verify_manifest(d2)["ok"] and len({x["session_id"] for x in ses if x["status"] == "complete"}) == len(planned), r4.stdout[-200:])
        check("outage: the cut-off session was re-run as a new attempt and the failed attempt is excluded from the counts",
              cut and any(x["attempt"] >= 2 and x["status"] == "complete" for x in ses if x["session_id"] == cut[0]["session_id"]) and len(runview.load_rows(d2, only_valid=True)) == sum(s["turns"] for s in planned),
              "%d counted rows" % len(runview.load_rows(d2, only_valid=True)))
    finally:
        _kill_stub(stub)
        shutil.rmtree(tmp, ignore_errors=True)


def cmd_dryrun(args):
    from .runner import Run
    from . import labels
    path = _config_path(args.config or "configs/dryrun.json")
    cfg = load_config(path)
    assert cfg["mode"] == "stub", "dry run must use stub mode"
    run = Run(cfg, path, name_suffix="")
    d = run.execute()
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    check("run completed", run.status == "complete", run.status)
    ver = verify_manifest(d)
    check("MANIFEST.sha256 verifies", ver["ok"], "checked %d files; %s" % (ver["checked"], {k: v for k, v in ver.items() if k in ("missing", "changed", "unlisted") and v}))
    # tamper evidence: flip one byte in a copy and expect verification to fail
    tmp = tempfile.mkdtemp(prefix="hf_tamper_")
    try:
        cp = os.path.join(tmp, "copy")
        shutil.copytree(d, cp, ignore=shutil.ignore_patterns("._*"))
        with open(os.path.join(cp, "scores.csv"), "ab") as f:
            f.write(b"x")
        check("manifest detects tampering", not verify_manifest(cp)["ok"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    pngs = sorted(glob.glob(os.path.join(d, "graphs", "*.png")))
    check("graphs written (8 PNGs)", len(pngs) >= 8, "%d: %s" % (len(pngs), [os.path.basename(p) for p in pngs]))
    check("REPORT.md written", os.path.getsize(os.path.join(d, "REPORT.md")) > 3000)
    tr = [r for r in read_jsonl(os.path.join(d, "transcripts.jsonl")) if not r.get("event")]
    with open(os.path.join(d, "scores.csv"), newline="", encoding="utf-8") as f:
        sc = list(csv.DictReader(f))
    check("transcripts and scores agree (one row per turn)", len(tr) == len(sc) and len(tr) > 0, "%d transcript turns, %d score rows" % (len(tr), len(sc)))
    main_ok = [r for r in sc if r["phase"] == "main" and r["status"] == "ok"]
    planned = sum(s["turns"] for s in run.sessions)
    check("every planned main turn recorded", len(main_ok) == planned, "%d of %d" % (len(main_ok), planned))
    sess = [r for r in read_jsonl(os.path.join(d, "sessions.jsonl")) if r.get("event") == "session"]
    check("all sessions complete", all(s["status"] == "complete" for s in sess), str([s["session_id"] for s in sess if s["status"] != "complete"]))
    check("every briefing recorded verbatim", all(t.get("briefing") and t["briefing"].get("system") and t["briefing"].get("user") for t in tr if t["phase"] == "main"))
    incs = read_jsonl(os.path.join(d, "incidents.jsonl"))
    check("incident detector fired in the bad-instance demo", len(incs) >= 1, "%d incident(s)" % len(incs))
    check("fresh restart + replay cleared the demo incident", any(i.get("cleared_by_fresh_restart") is True for i in incs))
    check("incident records instance start time and server args", all(i["instance"].get("start_time") and i["instance"].get("server_args") for i in incs))
    off = [t for t in tr if t["phase"] == "main" and t["configuration"] == "stub_guards_off" and t.get("final_reply") and t.get("sanitized_reply")]
    same = sum(1 for t in off if t["final_reply"] == t["sanitized_reply"])
    check("guards OFF switches only the guards (the app's sanitizer, incl. its paragraph re-splitting, still runs)", off and same / len(off) >= 0.95, "%d of %d guards-off replies equal the sanitizer output" % (same, len(off)))
    check("guards OFF: no rewrite call was ever made", all(t["rewrite_count"] == 0 for t in off))
    check("human_labels.csv written", os.path.exists(os.path.join(d, "human_labels.csv")))
    sv = [r for r in read_jsonl(os.path.join(d, "sessions.jsonl")) if r.get("event") == "server_stopped"]
    check("every server stop verified (process, children and port gone)", sv and all(r["verification"]["ok"] for r in sv), "%d stops" % len(sv))
    closed = [r for r in read_jsonl(os.path.join(d, "sessions.jsonl")) if r.get("event") == "session_closed"]
    check("every Electron shutdown verified", closed and all(c.get("electron_shutdown_ok") for c in closed), "%d sessions closed" % len(closed))
    blocked = sorted({u for s in sess for u in (s.get("blocked_requests") or [])})
    check("app's boot-time request to the forbidden default port was blocked before the network", any(":1234/" in u for u in blocked) and all(":1234/" in u or ":1238/" in u for u in blocked), str(blocked))
    end = json.load(open(os.path.join(d, "run_end.json")))
    check("ports 1236 and 1237 are free after the run", end["post_run_checks"]["port_1236_free"] and end["post_run_checks"]["port_1237_free"])
    left = _procs_left()
    check("no stray harness processes", not left, str(left))
    check("no scratch profile left behind", not os.path.exists(os.path.join(d, "_scratch")))
    # agreement code path on SIMULATED labels (copy of the sheet; never written into the run folder)
    sheet = os.path.join(d, "human_labels.csv")
    if os.path.exists(sheet):
        import random
        tmp = tempfile.mkdtemp(prefix="hf_sim_labels_")
        try:
            rows = [l for l in open(sheet, encoding="utf-8") if not l.startswith("#")]
            rdr = list(csv.DictReader(rows))
            rng = random.Random(1)
            for r in rdr:
                r["human_head_hop"] = rng.choice([0, 0, 0, 1]); r["human_personality_lock"] = rng.randint(2, 5)
                r["human_writing_quality"] = rng.randint(2, 5); r["human_rules_ok"] = rng.choice([1, 1, 1, 0])
            sim = os.path.join(tmp, "simulated_labels.csv")
            with open(sim, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=rdr[0].keys()); w.writeheader(); w.writerows(rdr)
            ag = labels.compute_agreement(d, sim, write=False)
            check("agreement calculation runs on simulated labels (not saved)", ag["rows_labeled"] == len(rdr) and ag["human_vs_llm_judge"]["head_hop"]["n"] > 0, "n=%s" % ag["human_vs_llm_judge"]["head_hop"].get("n"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    # ---- phase B: pause with --max-minutes, then resume to completion
    import signal as _sig
    from . import runview
    py = [sys.executable, "-m", "harness"]
    env = dict(os.environ)
    resume_cfg = os.path.join("configs", "dryrun_resume.json")
    runs_dir = os.path.join(HARNESS_ROOT, "runs")

    def newest(suffix):
        c = sorted(glob.glob(os.path.join(runs_dir, "*_dryrun_resume" + suffix)))
        return c[-1] if c else None
    r1 = subprocess.run(py + ["run", "--config", resume_cfg, "--suffix", "_pause", "--max-minutes", "0.05"], cwd=HARNESS_ROOT, capture_output=True, text=True, timeout=600, env=env)
    dp = newest("_pause")
    check("pause: run stopped at --max-minutes without sealing", dp and os.path.exists(os.path.join(dp, "run_status.json")) and not os.path.exists(os.path.join(dp, "MANIFEST.sha256")) and "paused" in json.load(open(os.path.join(dp, "run_status.json")))["status"], (r1.stdout[-200:] if r1 else ""))
    check("pause: checkpoint written", dp and glob.glob(os.path.join(dp, "CHECKPOINT_*.json")))
    done_before = len([x for x in runview.load_sessions(dp) if x["status"] == "complete"]) if dp else -1
    r2 = subprocess.run(py + ["resume", "--run", dp], cwd=HARNESS_ROOT, capture_output=True, text=True, timeout=600, env=env)
    ses = runview.load_sessions(dp)
    check("resume: finished the remaining sessions and sealed the run", os.path.exists(os.path.join(dp, "MANIFEST.sha256")) and verify_manifest(dp)["ok"] and len({x["session_id"] for x in ses if x["status"] == "complete"}) == 4,
          "complete before resume: %s; %s" % (done_before, r2.stdout[-300:]))
    check("resume: sessions finished before the pause were not re-run", all(x["attempt"] == 1 for x in ses if x["status"] == "complete"))
    check("local scripted player + no judge path produced a report", os.path.exists(os.path.join(dp, "REPORT.md")) and "no judge" in open(os.path.join(dp, "REPORT.md")).read().lower())

    # ---- phase C: stop the harness hard mid-session (SIGTERM, like the owner stopping it), then resume
    proc = subprocess.Popen(py + ["run", "--config", resume_cfg, "--suffix", "_kill", "--nice"], cwd=HARNESS_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    t0 = time.time()
    dk = None
    while time.time() - t0 < 180:
        dk = newest("_kill")
        sp = os.path.join(dk, "scores.csv") if dk else None
        if sp and os.path.exists(sp) and sum(1 for _ in open(sp)) - 1 >= 4:      # session 1 finished and session 2 is under way
            break
        time.sleep(0.2)
    proc.send_signal(_sig.SIGTERM)
    try:
        proc.wait(timeout=90)
    except subprocess.TimeoutExpired:
        proc.kill()
    st = json.load(open(os.path.join(dk, "run_status.json")))
    check("stop: SIGTERM mid-session leaves a clean, resumable folder", "interrupted" in st["status"] and glob.glob(os.path.join(dk, "CHECKPOINT_*.json")) and st["post_run_checks"]["port_1237_free"] and not _procs_left(), st["status"][:80])
    n_rows_before = sum(1 for _ in open(os.path.join(dk, "scores.csv"))) - 1
    r3 = subprocess.run(py + ["resume", "--run", dk], cwd=HARNESS_ROOT, capture_output=True, text=True, timeout=600, env=env)
    ses = runview.load_sessions(dk)
    rows_all = runview.load_rows(dk, only_valid=False)
    rows_valid = runview.load_rows(dk, only_valid=True)
    interrupted = [x for x in ses if x["status"] == "interrupted"]
    check("stop: the cut-off session was recorded as interrupted", len(interrupted) == 1, str([(x["session_id"], x["status"]) for x in ses]))
    check("resume after stop: every session complete, folder sealed and verified", os.path.exists(os.path.join(dk, "MANIFEST.sha256")) and verify_manifest(dk)["ok"] and len({x["session_id"] for x in ses if x["status"] == "complete"}) == 4, r3.stdout[-300:])
    check("resume after stop: nothing recorded before the stop was lost, cut-off session re-run as attempt 2 and excluded from aggregates",
          len(rows_all) >= n_rows_before and interrupted and any(x["attempt"] == 2 and x["status"] == "complete" for x in ses if x["session_id"] == interrupted[0]["session_id"]) and len(rows_valid) < len(rows_all) and len(rows_valid) == 12,
          "rows before stop %d, all rows %d, counted rows %d" % (n_rows_before, len(rows_all), len(rows_valid)))
    check("no stray harness processes after resume tests", not _procs_left())

    # ---- phase D: external-server mode (already-running stub on 1237 serving two model names, no spawn/stop by the run)
    dryrun_external_phase(check, py, env, runs_dir)
    check("no stray harness processes after the external-server tests", not _procs_left())

    print("\nDRY RUN RESULTS  (run folder: %s)" % d)
    for name, ok, detail in results:
        print("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  -- " + detail) if detail and (not ok or len(detail) < 120) else ""))
    allok = all(ok for _, ok, _ in results)
    print("\nOVERALL:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


def cmd_verify(args):
    r = verify_manifest(os.path.abspath(args.run))
    print(json.dumps(r, indent=2))
    return 0 if r["ok"] else 1


def cmd_agreement(args):
    from . import labels
    r = labels.compute_agreement(os.path.abspath(args.run), os.path.abspath(args.labels))
    print(json.dumps(r, indent=2))
    return 0


def cmd_export_incidents(args):
    for i in read_jsonl(os.path.join(os.path.abspath(args.run), "incidents.jsonl")):
        c = i["config"]
        print("| %s | benchmark run %s, session %s | %s | %ss at streak start | %d consecutive head-hops from turn %d (%s person, L%s, %s) | %s | instance %s started %s; server args: `%s` |" % (
            i["detected_at"][:10], i["run_id"], i["session_id"], c.get("model"), i["instance"]["age_s_at_streak_start"], i["streak_len_at_detection"], i["streak_start_turn_index"] + 1,
            c["perspective"], c["level"], c["scenario"], "fresh restart cleared it" if i.get("cleared_by_fresh_restart") else ("fresh restart did NOT clear it" if i.get("cleared_by_fresh_restart") is False else "not tested"),
            i["instance"]["instance_id"], i["instance"]["start_time"], " ".join(i["instance"]["server_args"])[:140]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m harness", description="Solo R Benchmark Harness v%s" % __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dryrun"); p.add_argument("--config"); p.set_defaults(fn=cmd_dryrun)
    p = sub.add_parser("run"); p.add_argument("--config", required=True); p.add_argument("--suffix"); p.add_argument("--i-have-permission", action="store_true")
    p.add_argument("--with-guards-off", action="store_true", help="also run the optional guards-off configurations"); p.add_argument("--max-minutes", type=float, help="stop starting new sessions near this limit (pausable; resume later)")
    p.add_argument("--hard-extra-minutes", type=float, default=10, help="mid-session hard stop at max-minutes + this (default 10)")
    p.add_argument("--nice", type=int, nargs="?", const=10, help="lower scheduling priority (default +10 when given without a value)"); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("resume"); p.add_argument("--run", required=True); p.add_argument("--i-have-permission", action="store_true"); p.add_argument("--max-minutes", type=float)
    p.add_argument("--hard-extra-minutes", type=float, default=10); p.add_argument("--nice", type=int, nargs="?", const=10); p.add_argument("--rest-ratio", type=float, default=None, help="override the between-turn rest for this invocation (0 = full speed, for unattended overnight runs); recorded in resume_log.jsonl"); p.set_defaults(fn=cmd_resume)
    p = sub.add_parser("status"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("finalize"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_finalize)
    p = sub.add_parser("cleanup"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_cleanup)
    p = sub.add_parser("check"); p.add_argument("--config", required=True); p.add_argument("--with-guards-off", action="store_true"); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("verify"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("agreement"); p.add_argument("--run", required=True); p.add_argument("--labels", required=True); p.set_defaults(fn=cmd_agreement)
    p = sub.add_parser("export-incidents"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_export_incidents)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except ConfigError as e:
        print("config error:", e)
        return 2
