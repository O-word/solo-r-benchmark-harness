"""Integration test: the Python replicas of the app's rules agree with the real app code (runs Electron headless, no network)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import scoring as S  # noqa: E402
from harness.util import HARNESS_ROOT  # noqa: E402

ELECTRON = "/path/to/your/Electron"
SAMPLES = ["x" * 10, "x" * 349, "x" * 351, "x" * 900, "x" * 901, "x" * 1800, "x" * 1801, "a\nb\nc", "a\nb\nc\nd\ne\nf", "\n".join("l" for _ in range(10)), "a\n\n b \n\n\n c", ""]
QUOTES = ['He said "go" and left.', 'He said "go and left.', "He left (quietly.", 'She nodded.\n"Fine," he said.', 'A "b" c "d" e "f']


@unittest.skipUnless(os.path.exists(ELECTRON), "Electron not installed")
class AppParity(unittest.TestCase):
    def test_replicas_match_app(self):
        tmp = tempfile.mkdtemp(prefix="hf_parity_")
        self.addCleanup(shutil.rmtree, tmp, True)
        js = os.path.join(tmp, "t.js")
        with open(js, "w") as f:
            f.write("""(() => {
              const out = {len: [], par: [], unpaired: []};
              for (const lv of ["1","2","3"]) { state.playerProfile.rpStyleLevel = lv; for (const t of %s) out.len.push([lv, t, describeReplyLengthTarget(t)]); }
              for (const t of %s) out.par.push([t, countParagraphs(t)]);
              for (const t of %s) out.unpaired.push([t, hasUnpairedPunctuationDrift(t)]);
              return out; })()""" % (json.dumps(SAMPLES), json.dumps(SAMPLES), json.dumps(QUOTES)))
        r = subprocess.run([ELECTRON, os.path.join(HARNESS_ROOT, "tests", "app_eval_main.cjs"), os.path.join(HARNESS_ROOT, "app_snapshot", "app", "index.html"),
                            os.path.join(HARNESS_ROOT, "driver", "preload_stub.cjs"), js, os.path.join(tmp, "ud")], capture_output=True, text=True, timeout=120)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
        self.assertTrue(line, r.stdout + r.stderr[-500:])
        out = json.loads(line[0][7:])
        for lv, text, app_text in out["len"]:
            n = S.describe_reply_length_target(text, lv)
            self.assertIn("at least %d paragraph" % n, app_text, (lv, text[:20], app_text))
        for text, n in out["par"]:
            self.assertEqual(S.count_paragraphs(text), n, text)
        for text, flagged in out["unpaired"]:
            if flagged:
                self.assertTrue(S.quote_problems(text), "app flags %r but harness does not" % text)


if __name__ == "__main__":
    unittest.main()
