"""Tests for the EXTERNAL-server mode (an already-running OpenAI-compatible server reached through a local tunnel port).

Everything runs against the existing stub on port 1237, started here as an 'external' server (two model names: base, lora).
Nothing real is contacted: no model, no network beyond 127.0.0.1:1237, never ports 1234/1235/1238 (the tunnel port is only
ever *validated*, never connected to).
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import cli, config as C  # noqa: E402
from harness.electron_driver import DriverError, ElectronSession  # noqa: E402
from harness.runner import Run  # noqa: E402
from harness.server_manager import (ExternalServer, ExternalServerDown, ServerError, ServerManager, check_port_allowed, port_in_use)  # noqa: E402
from harness.util import EXTERNAL_PORTS, HARNESS_PORTS, HARNESS_ROOT, TUNNEL_PORT, read_jsonl  # noqa: E402


def ext_cfg(**kw):
    conf = lambda cid, name: {"id": cid, "model_name": name, "model": name, "server": {"kind": "external", "port": 1237}}  # noqa: E731
    base = {"name": "t", "mode": "external", "seed": 5, "tense": "past", "turns": 3, "perspectives": ["first", "second", "third"], "levels": [1, 2, 3],
            "scenarios": ["a", "b"], "replicates": 1, "configurations": [conf("base", "base"), conf("lora", "lora")]}
    base.update(kw)
    return base


class StubMixin:
    """Start/stop the stub as an external server on 1237 (two model names)."""

    def start_stub(self, **kw):
        self.assertFalse(port_in_use(1237), "port 1237 must be free for these tests")
        self.stublog = os.path.join(tempfile.mkdtemp(prefix="hf_ext_test_"), "stub.jsonl")
        self.addCleanup(shutil.rmtree, os.path.dirname(self.stublog), True)
        self.stub = cli._spawn_external_stub(log=self.stublog, **kw)
        self.addCleanup(cli._kill_stub, self.stub)
        return self.stub

    def stop_stub(self):
        cli._kill_stub(self.stub)
        for _ in range(50):
            if not port_in_use(1237):
                break
            time.sleep(0.1)


class StubTwoModels(StubMixin, unittest.TestCase):
    def post(self, model):
        body = json.dumps({"model": model, "messages": [{"role": "system", "content": "You are Brick in Holo."}, {"role": "user", "content": "Player: Mara\nPlayer message:\nsay hi"}]}).encode()
        req = urllib.request.Request("http://127.0.0.1:1237/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(req, timeout=10)

    def test_serves_two_names_and_strict_models(self):
        self.start_stub()
        self.assertEqual(ExternalServer(1237).models(), ["base", "lora"])
        for name in ("base", "lora"):
            r = self.post(name)
            self.assertEqual(r.status, 200)
            self.assertEqual(json.loads(r.read())["model"], name)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("default_model")
        self.assertEqual(cm.exception.code, 404)
        self.assertEqual([r["model"] for r in read_jsonl(self.stublog)], ["base", "lora"])


class ExternalServerTests(StubMixin, unittest.TestCase):
    def test_ports(self):
        self.assertEqual(ExternalServer(1237).port, 1237)
        self.assertEqual(ExternalServer(TUNNEL_PORT).base_url, "http://127.0.0.1:1238")     # constructing never connects
        for p in (1234, 1235, 1236, 8080, 80):
            with self.assertRaises(ServerError):
                ExternalServer(p)
        # the tunnel port is valid for external servers only, never for a server the harness spawns
        self.assertEqual(check_port_allowed(1238, EXTERNAL_PORTS), 1238)
        with self.assertRaises(ServerError):
            check_port_allowed(1238)
        with self.assertRaises(ServerError):
            ServerManager(["x", "--port", "{port}"], 1238, kind="real")
        self.assertEqual(HARNESS_PORTS, (1236, 1237, 1238))

    def test_start_records_url_and_models_and_stop_does_not_touch_the_server(self):
        stub = self.start_stub()
        ext = ExternalServer(1237, required_models=["base"], health_timeout=5)
        info = ext.start()
        self.assertEqual(info["base_url"], "http://127.0.0.1:1237")
        self.assertEqual(info["served_models"], ["base", "lora"])
        self.assertIsNone(info["pid"])
        self.assertEqual(info["kind"], "external")
        v = ext.stop()
        self.assertTrue(v["ok"] and v["external"] and not v["stopped"] and v["remote_process_untouched"] and v["tunnel_untouched"])
        self.assertIsNone(stub.poll())                       # still running
        self.assertEqual(ExternalServer(1237).models(), ["base", "lora"])
        self.assertEqual(read_jsonl(self.stublog), [])        # readiness used GET /v1/models only: no chat request

    def test_unreachable_raises_externalserverdown_quickly(self):
        self.assertFalse(port_in_use(1237))
        t0 = time.time()
        with self.assertRaises(ExternalServerDown):
            ExternalServer(1237, health_timeout=1.0, poll_interval=0.2).start()
        self.assertLess(time.time() - t0, 6)
        pr = ExternalServer(1237).probe()
        self.assertFalse(pr["reachable"])
        self.assertTrue(pr["error"])

    def test_missing_model_name_is_a_failure(self):
        self.start_stub()
        with self.assertRaises(ExternalServerDown) as cm:
            ExternalServer(1237, required_models=["base", "nope"], health_timeout=1.0, poll_interval=0.2).start()
        self.assertIn("nope", str(cm.exception))

    def test_mid_run_check_detects_a_dropped_server_without_killing_anything(self):
        self.start_stub()
        ext = ExternalServer(1237, required_models=["lora"])
        ext.start()
        self.assertTrue(ext.check()["ok"])
        self.stop_stub()                                      # the 'tunnel' drops
        t0 = time.time()
        h = ext.check(grace_s=1.0)
        self.assertFalse(h["ok"])
        self.assertTrue(h["error"])
        self.assertGreaterEqual(time.time() - t0, 0.9)        # it retried for the grace period before giving up

    def test_die_after_simulates_a_dropped_tunnel(self):
        stub = self.start_stub(die_after=1)
        StubTwoModels.post(self, "base")                     # 1st chat request answered
        with self.assertRaises(Exception):
            StubTwoModels.post(self, "base")                 # 2nd: the process exits without answering
        for _ in range(50):
            if stub.poll() is not None:
                break
            time.sleep(0.1)
        self.assertIsNotNone(stub.poll())
        self.assertFalse(ExternalServer(1237).probe()["reachable"])


class ConfigTests(unittest.TestCase):
    def test_external_config_validates(self):
        C.validate(ext_cfg())

    def test_external_rules(self):
        def bad(mut):
            cfg = ext_cfg()
            mut(cfg)
            with self.assertRaises(C.ConfigError):
                C.validate(cfg)
        bad(lambda c: c["configurations"][0].pop("model_name"))
        bad(lambda c: c["configurations"][0].update(model_name="two words"))
        for port in (1234, 1235, 1236, 8080):
            bad(lambda c, port=port: c["configurations"][0]["server"].update(port=port))
        bad(lambda c: c["configurations"][0]["server"].update(command=["x"]))
        bad(lambda c: c["configurations"][0]["server"].update(host="10.0.0.5"))
        bad(lambda c: c["configurations"][0]["server"].update(kind="real"))
        bad(lambda c: c["configurations"][0].update(force_model=False))
        cfg = ext_cfg()
        cfg["configurations"][0]["server"]["port"] = TUNNEL_PORT        # the tunnel port is allowed
        C.validate(cfg)

    def test_external_kind_only_in_external_mode(self):
        cfg = ext_cfg(mode="real")
        with self.assertRaises(C.ConfigError):
            C.validate(cfg)

    def test_alternating_order_default_for_external(self):
        s = C.expand_sessions(ext_cfg())
        self.assertEqual(len(s), 36)
        firsts = []
        for i in range(0, 36, 2):
            a, b = s[i], s[i + 1]
            self.assertEqual((a["perspective"], a["level"], a["scenario"], a["seed"]), (b["perspective"], b["level"], b["scenario"], b["seed"]))   # same cell, same input
            self.assertNotEqual(a["configuration"], b["configuration"])
            firsts.append(a["configuration"])
        self.assertEqual(firsts.count("base"), 9)
        self.assertEqual(firsts.count("lora"), 9)
        self.assertEqual(firsts[:4], ["base", "lora", "base", "lora"])
        # no long runs of one configuration (fair across the time of day)
        seq = [x["configuration"] for x in s]
        self.assertLessEqual(max(len(list(g)) for _, g in __import__("itertools").groupby(seq)), 2)
        self.assertEqual([x["session_id"] for x in C.expand_sessions(ext_cfg())], [x["session_id"] for x in s])

    def test_alternating_with_three_configurations_rotates(self):
        cfg = ext_cfg()
        cfg["configurations"].append({"id": "third", "model_name": "third", "model": "third", "server": {"kind": "external", "port": 1237}})
        s = C.expand_sessions(cfg)
        self.assertEqual(len(s), 54)
        firsts = [s[i]["configuration"] for i in range(0, 54, 3)]
        self.assertEqual({firsts.count(x) for x in ("base", "lora", "third")}, {6})
        for i in range(0, 54, 3):
            self.assertEqual(len({s[i + k]["configuration"] for k in range(3)}), 3)

    def test_explicit_order_is_respected(self):
        s = C.expand_sessions(ext_cfg(order="configuration_major"))
        self.assertEqual([x["configuration"] for x in s], ["base"] * 18 + ["lora"] * 18)

    def test_local_order_unchanged(self):
        cfg = {"name": "t", "mode": "real", "seed": 1, "tense": "past", "turns": 3, "perspectives": ["first", "second", "third"], "levels": [1, 2, 3], "scenarios": ["a", "b"],
               "configurations": [{"id": "c1", "server": {"kind": "stub", "port": 1237}}, {"id": "c2", "server": {"kind": "stub", "port": 1237}}]}
        self.assertEqual([x["configuration"] for x in C.expand_sessions(cfg)], ["c1"] * 18 + ["c2"] * 18)

    def test_model_name_per_configuration(self):
        base = {"model": "default_model"}
        self.assertEqual(Run.model_for(dict(base)), "default_model")
        self.assertEqual(Run.model_for(dict(base, model_name="lora")), "lora")
        self.assertEqual(Run.model_for({}), "default_model")

    def test_shipped_runpod_config(self):
        cfg = C.load_config(os.path.join(HARNESS_ROOT, "configs", "pilot_runpod.json"))
        self.assertEqual(cfg["mode"], "external")
        self.assertEqual([c["id"] for c in cfg["configurations"]], ["base", "lora"])
        self.assertEqual([c["model_name"] for c in cfg["configurations"]], ["base", "lora"])
        self.assertTrue(all(c["server"] == {"kind": "external", "host": "127.0.0.1", "port": 1238, "health_timeout_s": 30, "health_recheck_s": 30, "request_timeout_s": 5} for c in cfg["configurations"]))
        self.assertEqual(cfg["perspectives"], ["first", "second", "third"])
        self.assertEqual(cfg["levels"], [1, 2, 3])
        self.assertEqual(len(cfg["scenarios"]), 2)
        self.assertEqual(cfg["turns"], 6)
        self.assertEqual(cfg["throttle"]["rest_ratio"], 0)
        self.assertEqual(cfg["player_bot"]["kind"], "local_scripted")
        self.assertEqual(cfg["judge"]["kind"], "none")
        self.assertEqual(len(C.expand_sessions(cfg)), 36)
        self.assertEqual(sum(s["turns"] for s in C.expand_sessions(cfg)), 216)
        self.assertTrue(cli.needs_permission(cfg))

    def test_gemini_template_is_a_template(self):
        path = os.path.join(HARNESS_ROOT, "configs", "pilot_runpod_gemini_TEMPLATE.json")
        cfg = C.load_config(path)
        self.assertEqual((cfg["player_bot"]["kind"], cfg["judge"]["kind"]), ("gemini", "gemini"))
        self.assertIn("REPLACE_ME", json.dumps(cfg))
        self.assertEqual([c["model_name"] for c in cfg["configurations"]], ["base", "lora"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["run", "--config", path, "--i-have-permission"]), 2)      # refuses before anything starts
        self.assertIn("REPLACE_ME", out.getvalue())

    def test_pilot_runpod_run_needs_permission(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["run", "--config", os.path.join(HARNESS_ROOT, "configs", "pilot_runpod.json")]), 2)               # no --i-have-permission: refuses, starts nothing
        self.assertIn("--i-have-permission", out.getvalue())
        self.assertFalse(cli.needs_permission(C.load_config(os.path.join(HARNESS_ROOT, "configs", "dryrun_external.json"))))             # stub-only external config: no permission needed


class CheckCommand(StubMixin, unittest.TestCase):
    CFG = os.path.join(HARNESS_ROOT, "configs", "dryrun_external.json")

    def run_check(self, cfg_path=None):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["check", "--config", cfg_path or self.CFG])
        return rc, buf.getvalue()

    def test_reports_answering_and_models_without_any_chat_request(self):
        if not os.path.exists(json.load(open(os.path.join(HARNESS_ROOT, "configs", "dryrun_external.json")))["electron_bin"]):
            self.skipTest("set electron_bin in the configs to your Electron binary to run this check")
        self.start_stub()
        rc, out = self.run_check()
        self.assertEqual(rc, 0, out)
        self.assertIn("ANSWERS", out)
        self.assertIn("served model names: base, lora", out)
        self.assertIn("[ok] configuration ext_base sends model_name 'base'", out)
        self.assertIn("[ok] configuration ext_lora sends model_name 'lora'", out)
        self.assertIn("no chat request was sent", out)
        self.assertEqual(read_jsonl(self.stublog), [])

    def test_reports_not_answering(self):
        self.assertFalse(port_in_use(1237))
        rc, out = self.run_check()
        self.assertNotEqual(rc, 0)
        self.assertIn("NOT ANSWERING", out)

    def test_reports_missing_model_name(self):
        self.start_stub()
        cfg = json.load(open(self.CFG))
        cfg["configurations"][1]["model_name"] = "adapter_v2"
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        p = os.path.join(tmp, "c.json")
        json.dump(cfg, open(p, "w"))
        rc, out = self.run_check(p)
        self.assertNotEqual(rc, 0)
        self.assertIn("[MISSING] configuration ext_lora", out)
        self.assertEqual(read_jsonl(self.stublog), [])

    def test_pilot_runpod_check_does_not_hang_or_crash(self):
        """Never connects to anything but 127.0.0.1:1238 /v1/models (nothing listens there in the tests; a live tunnel would just be listed)."""
        if port_in_use(TUNNEL_PORT):
            self.skipTest("something is listening on 1238 (a real tunnel?); not touching it")
        rc, out = self.run_check(os.path.join(HARNESS_ROOT, "configs", "pilot_runpod.json"))
        self.assertNotEqual(rc, 0)
        self.assertIn("127.0.0.1:1238", out)
        self.assertIn("NOT ANSWERING", out)


class AllowlistTests(unittest.TestCase):
    def test_python_side_ports(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        scratch = os.path.join(tmp, "_scratch")
        args = ("/bin/echo", "/tmp/x.html", scratch, "t")
        for p in (1234, 1235, 8080, 1239):
            with self.assertRaises(DriverError):
                ElectronSession(*args, p)
        s = ElectronSession(*args, TUNNEL_PORT)           # builds the session config only; starts nothing
        cfg = json.load(open(s.cfg_path))
        self.assertEqual((cfg["allowedHost"], cfg["allowedPort"], cfg["forbiddenPorts"]), ("127.0.0.1", 1238, [1234, 1235]))
        self.assertEqual(cfg["harnessPorts"], [1236, 1237, 1238])

    def node_filter(self, port, urls):
        node = shutil.which("node")
        if not node:
            self.skipTest("node not available")
        js = ("const {makeFilter}=require(%s);let f;try{f=makeFilter({allowedPort:%d,forbiddenPorts:[1234,1235]})}catch(e){console.log(JSON.stringify({error:e.message}));process.exit(0)}"
              "console.log(JSON.stringify(%s.map(u=>f.allowed(u))))") % (json.dumps(os.path.join(HARNESS_ROOT, "driver", "netfilter.cjs")), port, json.dumps(urls))
        out = subprocess.run([node, "-e", js], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_electron_filter_allows_1238_and_refuses_1234_1235(self):
        urls = ["http://127.0.0.1:1238/v1/chat/completions", "http://127.0.0.1:1238/v1/models", "http://127.0.0.1:1238",
                "http://127.0.0.1:1234/v1/models", "http://127.0.0.1:1235/v1/chat/completions", "http://127.0.0.1:1237/v1/models", "http://127.0.0.1:12380/v1/models",
                "http://localhost:1238/v1/models", "http://127.0.0.1:1238@evil.example/", "https://example.com/", "ws://127.0.0.1:1234/", "file:///tmp/index.html", "data:text/plain,hi"]
        got = self.node_filter(1238, urls)
        self.assertEqual(got, [True, True, True, False, False, False, False, False, False, False, False, True, True])

    def test_electron_filter_refuses_forbidden_or_unknown_allowed_ports(self):
        for port in (1234, 1235, 8080, 1239):
            self.assertIn("error", self.node_filter(port, []))
        self.assertEqual(self.node_filter(1237, ["http://127.0.0.1:1237/x", "http://127.0.0.1:1238/x"]), [True, False])


if __name__ == "__main__":
    unittest.main()
