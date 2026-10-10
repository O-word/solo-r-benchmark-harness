#!/usr/bin/env python3
"""Re-score head-hop (final) for finished runs with the current scoring.py, without touching the run folders.
Used after the 2026-10-07 scorer fix (artifact-sentence dropping is skipped when the quotation marks are intact).
Usage: python rescore_headhop.py RUN_DIR [RUN_DIR ...]   -> prints recorded vs rescored per configuration."""
import json, os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import scoring
import aggregate_runs as a
comp = {}
for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "companions", "*.json")):
    c = json.load(open(p)); comp[os.path.basename(p)[:-5]] = c
def pose_of(t):
    pp = t.get("player_pose")
    pose = pp.get("command", "") if isinstance(pp, dict) else str(pp or "")
    for pre in ("@emit ", "@pose ", "say ", "s "):
        if pose.startswith(pre): return pose[len(pre):]
    return pose
tot = {}
for run in sys.argv[1:]:
    ok = {(x["session_id"], str(x["attempt"]), str(x["turn_index"])): x for x in a.load(run)}
    for l in open(os.path.join(run, "transcripts.jsonl")):
        t = json.loads(l)
        if "guard_events" not in t: continue
        k = (t["session_id"], str(t["attempt"]), str(t["turn_index"]))
        if k not in ok: continue
        cid = t["companion_id"]; cp = comp.get(cid, {}).get("pronoun") or "he"
        qs = [t.get("raw_reply") or ""] + list(t.get("rewrite_texts") or [])
        hh = scoring.head_hop(t.get("final_reply") or "", pose_of(t), "Mara", "she", t["companion"], cp, t["perspective"], quote_sources=[q for q in qs if q])
        d = tot.setdefault((os.path.basename(run), t["configuration"]), [0, 0, 0])
        d[0] += 1; d[1] += int(float(ok[k].get("head_hop") or 0) >= 1); d[2] += int(hh["flag"])
for (run, cfg), (n, rec, new) in sorted(tot.items()):
    print(f"{run:40} {cfg:5} turns {n}  recorded final head-hop {rec} ({100*rec/n:.0f}%)  rescored {new} ({100*new/n:.0f}%)")
