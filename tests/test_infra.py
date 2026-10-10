import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import config as C, gemini_client as G, judges as J, labels as L, manifest as M, player_bots as B, poses as P, runview, stats as ST  # noqa: E402
from harness.server_manager import ServerError, ServerManager, check_port_allowed, make_stub_command, port_in_use, required_paths  # noqa: E402


class Stats(unittest.TestCase):
    def test_wilson(self):
        p, lo, hi = ST.wilson(3, 40)
        self.assertAlmostEqual(p, 0.075)
        self.assertAlmostEqual(lo, 0.0258, places=3)
        self.assertAlmostEqual(hi, 0.1987, places=3)
        self.assertEqual(ST.wilson(0, 10)[1], 0.0)
        self.assertTrue(ST.wilson(0, 0)[0] != ST.wilson(0, 0)[0])  # nan

    def test_kappa(self):
        self.assertEqual(ST.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0])[0], 1.0)
        k, po, n = ST.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertAlmostEqual(k, 0.0)
        self.assertEqual(n, 4)

    def test_weighted_kappa(self):
        self.assertAlmostEqual(ST.weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0)
        self.assertLess(ST.weighted_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]), 0)

    def test_pearson(self):
        self.assertAlmostEqual(ST.pearson([1, 2, 3], [2, 4, 6]), 1.0)


class Poses(unittest.TestCase):
    def test_conj(self):
        self.assertEqual(P.conj("step", "third", "past"), "stepped")
        self.assertEqual(P.conj("grip", "first", "past"), "gripped")
        self.assertEqual(P.conj("carry", "third", "present"), "carries")
        self.assertEqual(P.conj("be", "second", "past"), "were")
        self.assertEqual(P.conj("watch", "third", "present"), "watches")

    def test_every_scenario_pose_is_legal_and_sized(self):
        for sid in ("train_robbery", "dragon_fight", "quiet_conversation", "room_change", "clothing_change"):
            sc = C.load_scenario(sid)
            self.assertEqual(len(sc["turns"]), 8)
            for tense in ("past", "present"):
                for persp in ("first", "second", "third"):
                    for lvl in (1, 2, 3):
                        for t in sc["turns"]:
                            r = P.compose_pose(t, sc, persp, lvl, tense, "Brick", "she", seed=3, bait=t.get("bait"))
                            B.ensure_pose_command(r["command"])
                            self.assertNotIn("\n", r["command"])
                            n = r["chars"]
                            if lvl == 1:
                                self.assertLessEqual(n, 345, (sid, t["id"]))
                            elif lvl == 2:
                                self.assertTrue(900 < n <= 1250, (sid, t["id"], n))
                            else:
                                self.assertGreater(n, 1800, (sid, t["id"]))
                                self.assertGreaterEqual(r["paragraphs"], 3)

    def test_bait_turns_have_gaps_and_persona_baits_exist(self):
        for sid in ("train_robbery", "dragon_fight", "quiet_conversation", "room_change", "clothing_change"):
            sc = C.load_scenario(sid)
            self.assertTrue(any(t.get("bait_type") == "persona" for t in sc["turns"]), sid)
            for t in sc["turns"]:
                if t.get("bait_type") == "headhop":
                    self.assertTrue(t.get("gap"), (sid, t["id"]))

    def test_third_person_colon_pose_has_no_subject(self):
        sc = C.load_scenario("train_robbery")
        r = P.compose_pose(sc["turns"][0], sc, "third", 1, "present", "Brick", "she", seed=1)
        self.assertTrue(r["command"].startswith(":"))
        self.assertFalse(r["text"].startswith("She "))

    def test_same_poses_across_calls(self):
        sc = C.load_scenario("dragon_fight")
        a = P.compose_pose(sc["turns"][2], sc, "first", 3, "past", "Brick", "she", seed=9)
        b = P.compose_pose(sc["turns"][2], sc, "first", 3, "past", "Brick", "she", seed=9)
        self.assertEqual(a["command"], b["command"])


class Bots(unittest.TestCase):
    def test_ensure_pose_command(self):
        for ok in ("say hi", "s hi", ":waves", "@emit text", "@e text", "SAY hi"):
            B.ensure_pose_command(ok)
        for bad in ("hello there", "I walk in.", "/go east", "say ", "@emit", "say hi\nthere"):
            with self.assertRaises(ValueError):
                B.ensure_pose_command(bad)

    def test_normalize_pose(self):
        self.assertEqual(B.normalize_pose("emit", "One.\n\nTwo.")[0], "@emit One.%r%rTwo.")
        self.assertEqual(B.normalize_pose("say", '"Hi there"')[0], "say Hi there")
        self.assertEqual(B.normalize_pose("colon", "Mara steps in.", "Mara")[0], ":steps in.")

    def test_local_scripted_and_stub_describe(self):
        self.assertIn("scripted", B.make_bot({"kind": "local_scripted"}).describe()["label"])
        self.assertEqual(B.make_bot({"kind": "stub"}).describe()["kind"], "stub")

    def _ctx(self, level=1):
        sc = C.load_scenario("train_robbery")
        return {"scenario": sc, "turn": sc["turns"][3], "turn_index": 3, "n_turns": 8, "perspective": "third", "level": level, "tense": "past",
                "player": {"name": "Mara", "pronoun": "she"}, "companion": {"name": "Brick"}, "history": [], "seed": 5}

    def test_scripted_bot_is_identical_for_identical_ctx(self):
        b = B.make_bot({"kind": "local_scripted"})
        self.assertEqual(b.next_pose(self._ctx())["command"], b.next_pose(self._ctx())["command"])

    def test_gemini_bot_with_fake_client(self):
        class Fake:
            def __init__(self, text):
                self.text = text
                self.calls = 0

            def generate(self, *a, **k):
                self.calls += 1
                return {"text": self.text, "attempts": 1}
        bot = B.GeminiPlayerBot(client=Fake(json.dumps({"kind": "colon", "text": "Mara steps back, hand on the rail."})))
        out = bot.next_pose(self._ctx())
        self.assertTrue(out["command"].startswith(":steps back"))
        bot = B.GeminiPlayerBot(client=Fake("no json here"))
        with self.assertRaises(B.BotError):
            bot.next_pose(self._ctx())

    def test_gemini_bot_error_becomes_boterror(self):
        class Boom:
            def generate(self, *a, **k):
                raise G.GeminiError("HTTP 429: slow down", status=429, retryable=True)
        with self.assertRaises(B.BotError):
            B.GeminiPlayerBot(client=Boom()).next_pose(self._ctx())


class GeminiClient(unittest.TestCase):
    def _client(self, responses, key="TESTKEY-not-real-123"):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        kf = os.path.join(tmp, "k.env")
        with open(kf, "w") as f:
            f.write("OTHER=1\nGEMINI_API_KEY=%s\n" % key)
        seen = []
        it = iter(responses)

        def transport(url, headers, body, timeout):
            seen.append((url, dict(headers)))
            return next(it)
        env = os.environ.pop("GEMINI_API_KEY", None)
        self.addCleanup(lambda: os.environ.__setitem__("GEMINI_API_KEY", env) if env else None)
        sleeps = []
        return G.GeminiClient("m", key_file=kf, transport=transport, sleep=sleeps.append), seen, sleeps, key

    def test_success_key_in_header_not_url(self):
        ok = (200, {}, json.dumps({"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}]}).encode())
        c, seen, _, key = self._client([ok])
        self.assertEqual(c.generate("s", "u")["text"], "hi")
        self.assertNotIn(key, seen[0][0])
        self.assertEqual(seen[0][1]["x-goog-api-key"], key)

    def test_rate_limit_retries_with_retry_after(self):
        ok = (200, {}, json.dumps({"candidates": [{"content": {"parts": [{"text": "yes"}]}}]}).encode())
        c, seen, sleeps, _ = self._client([(429, {"Retry-After": "7"}, b"slow"), (503, {}, b"down"), ok])
        self.assertEqual(c.generate("s", "u")["text"], "yes")
        self.assertEqual(len(seen), 3)
        self.assertGreaterEqual(sleeps[0], 7)
        self.assertEqual(c.stats["rate_limited"], 1)

    def test_fatal_error_scrubs_key(self):
        c, _, _, key = self._client([(400, {}, ("bad request for key %s" % "TESTKEY-not-real-123").encode())])
        with self.assertRaises(G.GeminiError) as cm:
            c.generate("s", "u")
        self.assertNotIn(key, str(cm.exception))
        self.assertIn("REDACTED", str(cm.exception))

    def test_retries_exhausted(self):
        c, _, _, _ = self._client([(500, {}, b"x")] * 10)
        c.max_retries = 2
        with self.assertRaises(G.GeminiError):
            c.generate("s", "u")

    def test_missing_key(self):
        env = os.environ.pop("GEMINI_API_KEY", None)
        try:
            c = G.GeminiClient("m", key_file="/nonexistent/key.env", transport=lambda *a: (200, {}, b""))
            self.assertFalse(c.available())
            with self.assertRaises(G.GeminiError):
                c.generate("s", "u")
            self.assertEqual(G.key_status("/nonexistent/key.env")[0], "missing")
        finally:
            if env:
                os.environ["GEMINI_API_KEY"] = env


class Judges(unittest.TestCase):
    GOOD = '{"head_hop":0,"head_hop_confidence":"high","head_hop_evidence":"","personality_lock":4,"writing_quality":3,"rules_adherence":5,"rules_ok":1,"reasoning":"fine"}'

    def test_parse(self):
        self.assertEqual(J.parse_judge_json("noise " + self.GOOD + " trailing")["personality_lock"], 4)
        for bad in ('{"head_hop":2}', "no json", self.GOOD.replace('"personality_lock":4', '"personality_lock":9')):
            with self.assertRaises(Exception):
                J.parse_judge_json(bad)

    def _ctx(self):
        return {"companion": {"name": "Brick", "pronoun": "he"}, "player": {"name": "Mara", "pronoun": "she"}, "persona_summary": "gruff", "important_notes": "n", "hard_rules": "r",
                "preferences": ["p"], "perspective": "third", "tense": "past", "level": 1, "min_paragraphs": 1, "beat": "b", "bait_type": "persona", "previous_replies": [],
                "pose_text": "steps in.", "reply": 'Brick snorts. "Fine. Friends, then."', "companion_checks": {"hard_rule_patterns": [], "preference_patterns": [], "soften_patterns": ["friends"]}}

    def test_stub_judge_shape(self):
        out = J.StubJudge().judge(self._ctx())
        self.assertEqual(out["status"], "ok")
        self.assertTrue(1 <= out["parsed"]["personality_lock"] <= 5)

    def test_gemini_judge_never_raises(self):
        class Boom:
            def generate(self, *a, **k):
                raise G.GeminiError("HTTP 500", status=500)
        j = J.GeminiJudge(client=Boom())
        out = j.judge(self._ctx())
        self.assertEqual(out["status"], "error")
        class Junk:
            def generate(self, *a, **k):
                return {"text": "I refuse to output JSON", "attempts": 1}
        self.assertEqual(J.GeminiJudge(client=Junk()).judge(self._ctx())["status"], "unparseable")
        class Good:
            def generate(self, *a, **k):
                return {"text": JudgesGood, "attempts": 2}
        JudgesGood = self.GOOD
        r = J.GeminiJudge(client=Good()).judge(self._ctx())
        self.assertEqual((r["status"], r["parsed"]["rules_ok"]), ("ok", 1))
        self.assertEqual(len(r["prompt_sha256"]), 64)

    def test_null_judge(self):
        self.assertEqual(J.make_judge({"kind": "none"}).judge({})["status"], "none")

    def test_rubric_versioned(self):
        self.assertEqual(J.rubric_info()["version"], "1.0")


class Config(unittest.TestCase):
    def _cfg(self, **kw):
        base = {"name": "t", "mode": "stub", "seed": 1, "tense": "past", "turns": 3, "perspectives": ["first", "second", "third"], "levels": [1, 2, 3], "scenarios": ["a", "b"], "replicates": 1,
                "configurations": [{"id": "c1", "server": {"kind": "stub", "port": 1237}}, {"id": "c2", "server": {"kind": "stub", "port": 1237}},
                                   {"id": "opt", "optional": True, "server": {"kind": "stub", "port": 1237}}]}
        base.update(kw)
        return base

    def test_pilot_size_and_order(self):
        s = C.expand_sessions(self._cfg())
        self.assertEqual(len(s), 36)
        confs = [x["configuration"] for x in s]
        self.assertEqual(confs, ["c1"] * 18 + ["c2"] * 18)     # BASE completes before LORA starts

    def test_optional_included_on_request(self):
        self.assertEqual(len(C.expand_sessions(self._cfg(include_optional=True))), 54)

    def test_seeds_shared_across_configurations(self):
        s = C.expand_sessions(self._cfg())
        by = {}
        for x in s:
            by.setdefault((x["perspective"], x["level"], x["scenario"]), set()).add(x["seed"])
        self.assertTrue(all(len(v) == 1 for v in by.values()))

    def test_deterministic(self):
        self.assertEqual([x["session_id"] for x in C.expand_sessions(self._cfg())], [x["session_id"] for x in C.expand_sessions(self._cfg())])

    def test_forbidden_port_rejected(self):
        for port in (1234, 1235):
            cfg = self._cfg()
            cfg["configurations"][0]["server"]["port"] = port
            with self.assertRaises(C.ConfigError):
                C.validate(cfg)

    def test_real_mode_needs_1236_and_localhost(self):
        cmd = ["x", "--host", "127.0.0.1", "--port", "{port}"]
        cfg = self._cfg(mode="real", configurations=[{"id": "r", "server": {"kind": "real", "port": 1236, "command": cmd}}])
        C.validate(cfg)
        cfg["configurations"][0]["server"]["command"] = ["x", "--host", "0.0.0.0", "--port", "{port}"]
        with self.assertRaises(C.ConfigError):
            C.validate(cfg)
        cfg["configurations"][0]["server"]["command"] = ["x", "--host", "127.0.0.1", "--port", "1234"]
        with self.assertRaises(C.ConfigError):
            C.validate(cfg)

    def test_shipped_configs_validate(self):
        for f in sorted(os.listdir(os.path.join(C.HARNESS_ROOT, "configs"))):
            if f.startswith("._") or not f.endswith(".json"):
                continue
            C.validate(json.load(open(os.path.join(C.HARNESS_ROOT, "configs", f))))

    @unittest.skipUnless(os.path.exists(os.path.join(C.HARNESS_ROOT, "configs", "pilot_real.json")), "local-server config not shipped")
    def test_pilot_real_commands(self):
        cfg = json.load(open(os.path.join(C.HARNESS_ROOT, "configs", "pilot_real.json")))
        base = next(c for c in cfg["configurations"] if c["id"] == "base_nemo_guards_on")["server"]["command"]
        lora = next(c for c in cfg["configurations"] if c["id"] == "lora_nemo_guards_on")["server"]["command"]
        self.assertEqual(lora[:len(base)], base)
        self.assertEqual(lora[len(base):][0], "--adapter-path")
        self.assertNotIn("--adapter-path", base)
        self.assertIn("--use-default-chat-template", base)
        self.assertEqual(base[base.index("--chat-template-args") + 1], '{"enable_thinking":false}')
        self.assertEqual(base[base.index("--temp") + 1:base.index("--temp") + 2], ["0.6"])
        self.assertTrue(all("1234" not in t and "1235" not in t for t in base + lora))
        self.assertEqual(cfg["judge"]["kind"], "none")
        self.assertEqual(cfg["player_bot"]["kind"], "local_scripted")
        self.assertTrue(all(c.get("optional") for c in cfg["configurations"] if "guards_off" in c["id"]))


class ServerMgr(unittest.TestCase):
    def test_port_rules(self):
        for p in (1234, 1235, 8080):
            with self.assertRaises(ServerError):
                check_port_allowed(p)
        self.assertEqual(check_port_allowed(1236), 1236)
        self.assertEqual(check_port_allowed(1237), 1237)

    def test_placeholder_refused(self):
        with self.assertRaises(ServerError):
            ServerManager(["REPLACE_ME_cmd", "--port", "{port}"], 1237, kind="stub").preflight()

    def test_forbidden_port_in_command_refused(self):
        with self.assertRaises(ServerError):
            ServerManager(["python3", "--port", "1235"], 1237, kind="stub").preflight()
        with self.assertRaises(ServerError):
            ServerManager(["python3", "--host", "127.0.0.1:1234"], 1237, kind="stub").preflight()

    def test_mismatched_port_arg_refused(self):
        with self.assertRaises(ServerError):
            ServerManager(["python3", "--port", "1236"], 1237, kind="stub").preflight()

    def test_missing_path_refused_for_real(self):
        with self.assertRaises(ServerError):
            ServerManager(["/nonexistent/mlx_lm.server", "--model", "/nonexistent/model", "--port", "{port}"], 1236, kind="real").preflight()

    def test_required_paths(self):
        self.assertEqual(required_paths(["exe", "--model", "m", "--adapter-path", "a", "--port", "1"]), ["exe", "m", "a"])

    def test_start_health_stop_verify(self):
        self.assertFalse(port_in_use(1237))
        m = ServerManager(make_stub_command(mode="clean"), 1237, kind="stub", health_timeout=30)
        info = m.start()
        try:
            self.assertTrue(port_in_use(1237))
            self.assertEqual(info["port"], 1237)
        finally:
            v = m.stop()
        self.assertTrue(v["ok"], v)
        self.assertFalse(port_in_use(1237))

    def test_refuses_when_port_busy(self):
        a = ServerManager(make_stub_command(mode="clean"), 1237, kind="stub")
        a.start()
        try:
            with self.assertRaises(ServerError):
                ServerManager(make_stub_command(mode="clean"), 1237, kind="stub").preflight()
        finally:
            self.assertTrue(a.stop()["ok"])

    def test_stub_server_refuses_forbidden_port(self):
        import subprocess
        r = subprocess.run([sys.executable, os.path.join(C.HARNESS_ROOT, "harness", "stub_server.py"), "--port", "1234"], capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 2)


class Manifest(unittest.TestCase):
    def test_write_verify_tamper_addendum(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "graphs"))
        for rel, txt in (("a.txt", "A"), ("graphs/g.png", "PNG")):
            with open(os.path.join(d, rel), "w") as fh:
                fh.write(txt)
        with open(os.path.join(d, "._a.txt"), "w") as fh:
            fh.write("appledouble noise")
        n = M.write_manifest(d)
        self.assertEqual(n, 2)
        self.assertTrue(M.verify_manifest(d)["ok"])
        with open(os.path.join(d, "late.txt"), "w") as fh:
            fh.write("added later")
        self.assertFalse(M.verify_manifest(d)["ok"])       # unlisted file
        M.write_addendum(d, ["late.txt"])
        self.assertTrue(M.verify_manifest(d)["ok"])
        with open(os.path.join(d, "a.txt"), "a") as fh:
            fh.write("x")
        r = M.verify_manifest(d)
        self.assertFalse(r["ok"])
        self.assertEqual(r["changed"], ["a.txt"])
        os.remove(os.path.join(d, "graphs/g.png"))
        self.assertEqual(M.verify_manifest(d)["missing"], ["graphs/g.png"])


class RunView(unittest.TestCase):
    def test_valid_attempts(self):
        s = [{"event": "session", "session_id": "a", "attempt": 1, "status": "interrupted"},
             {"event": "session", "session_id": "a", "attempt": 2, "status": "complete"},
             {"event": "session", "session_id": "b", "attempt": 1, "status": "complete"},
             {"event": "session", "session_id": "b", "attempt": 2, "status": "crashed"},
             {"event": "session", "session_id": "c", "attempt": 1, "status": "crashed"}]
        self.assertEqual(runview.valid_attempts(s), {"a": 2, "b": 1})


class Labels(unittest.TestCase):
    def _fake_run(self, n_per_cell=10):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        rows, trs, sess = [], [], []
        k = 0
        for persp in ("first", "second", "third"):
            for lvl in (1, 2, 3):
                sid = "s_%s_%s" % (persp, lvl)
                sess.append({"event": "session", "session_id": sid, "attempt": 1, "status": "complete"})
                for t in range(n_per_cell):
                    k += 1
                    rows.append({"session_id": sid, "attempt": "1", "phase": "main", "status": "ok", "turn_index": str(t), "turn_id": "t%02d" % t, "perspective": persp, "level": str(lvl), "scenario": "train_robbery",
                                 "head_hop": str(k % 2), "judge_head_hop": str(k % 2 if k % 5 else 1 - k % 2), "adherence_rules_only_ok": "1", "judge_rules_ok": "1",
                                 "judge_personality_lock": str(1 + k % 5), "judge_writing_quality": str(1 + (k + 1) % 5)})
                    trs.append({"session_id": sid, "attempt": 1, "phase": "main", "turn_index": t, "companion": "Brick", "companion_id": "brick", "player_pose": {"command": "@emit x"}, "final_reply": "reply %d" % k})
        with open(os.path.join(d, "scores.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(d, "transcripts.jsonl"), "w") as f:
            for r in trs:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(d, "sessions.jsonl"), "w") as f:
            for r in sess:
                f.write(json.dumps(r) + "\n")
        json.dump({"run_id": "x"}, open(os.path.join(d, "config.json"), "w"))
        return d

    def test_sheet_is_blind_stratified_and_capped(self):
        d = self._fake_run()
        p = L.write_label_sheet(d, {"sample": 50, "seed": 7})
        lines = [l for l in open(p) if not l.startswith("#")]
        rows = list(csv.DictReader(lines))
        self.assertEqual(len(rows), 50)
        self.assertTrue({"human_head_hop", "human_personality_lock", "human_writing_quality", "human_rules_ok"} <= set(rows[0].keys()))
        for banned in ("configuration", "guards_on", "judge_head_hop", "head_hop", "judge_personality_lock"):
            self.assertNotIn(banned, rows[0].keys())
        cells = {(r["perspective"], r["level"]) for r in rows}
        self.assertEqual(len(cells), 9)                       # every perspective x level represented
        # deterministic
        p2 = L.write_label_sheet(d, {"sample": 50, "seed": 7})
        self.assertEqual(open(p).read(), open(p2).read())

    def test_agreement(self):
        d = self._fake_run()
        p = L.write_label_sheet(d, {"sample": 20, "seed": 3})
        scores = {(r["session_id"], int(r["turn_index"])): r for r in csv.DictReader(open(os.path.join(d, "scores.csv")))}
        rows = list(csv.DictReader([l for l in open(p) if not l.startswith("#")]))
        for r in rows:
            s = scores[(r["session_id"], int(r["turn_index"]))]
            r["human_head_hop"] = s["judge_head_hop"]            # human agrees with judge on every row
            r["human_personality_lock"] = s["judge_personality_lock"]
            r["human_writing_quality"] = s["judge_writing_quality"]
            r["human_rules_ok"] = "1"
        filled = os.path.join(d, "filled.csv")
        with open(filled, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        out = L.compute_agreement(d, filled, write=False)
        hj = out["human_vs_llm_judge"]
        self.assertEqual(hj["head_hop"]["n"], 20)
        self.assertAlmostEqual(hj["head_hop"]["exact_agreement"], 1.0)
        self.assertAlmostEqual(hj["personality_lock"]["weighted_kappa_quadratic"], 1.0)
        self.assertEqual(hj["personality_lock"]["within_1"], 1.0)
        self.assertLess(out["human_vs_deterministic_scorer"]["head_hop"]["exact_agreement"], 1.0)
        w = L.compute_agreement(d, filled, write=True)         # written as addenda, manifest untouched
        self.assertEqual(len(w["written"]), 3)
        self.assertTrue(all(os.path.exists(os.path.join(d, x)) for x in w["written"]))


if __name__ == "__main__":
    unittest.main()
