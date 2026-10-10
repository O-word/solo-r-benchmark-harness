#!/usr/bin/env python3
"""Compare benchmark builds side by side (the "growth" chart for the model card and the repo).
Usage: python3 compare_versions.py --out aggregates/growth_2026-10-08 \
   "build 1: original app (4-run mean)=RUN1,RUN2,RUN3,RUN4" "build 2: + cleanup and rewrite guard fixes=RUN5" "build 3: + freshness guard=RUN6"
Reads only scores.csv and transcripts.jsonl. Per-run rates are averaged; run-to-run range is shown. Never edits run folders."""
import argparse, csv, json, os, re, statistics as st, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("builds", nargs="+"); a = ap.parse_args()
def toks(t): return re.findall(r"[a-z0-9']+", str(t).lower().replace("’", "'"))
def run_metrics(path):
    rows = [r for r in csv.DictReader(open(os.path.join(path, "scores.csv"))) if r["status"] == "ok"]
    tr = [json.loads(l) for l in open(os.path.join(path, "transcripts.jsonl"))]
    tr = [t for t in tr if "turn_id" in t and t.get("status") == "ok"]
    by = collections.defaultdict(list)
    for t in tr: by[t["session_id"]].append(t)
    cfg_of = {}; opener = {"base": [0, 0], "lora": [0, 0]}
    for sid, ts in by.items():
        ts.sort(key=lambda x: x["turn_index"]); cfg = ts[0]["configuration"]; cfg_of[sid] = cfg
        for i in range(1, len(ts)):
            hist = [" ".join(toks(x["final_reply"])[:3]) for x in ts[max(0, i - 5):i]]
            o = " ".join(toks(ts[i]["final_reply"])[:3]); opener[cfg][1] += 1; opener[cfg][0] += 1 if o in hist else 0
    out = {}
    for cfg in ("base", "lora"):
        rs = [r for r in rows if r["configuration"] == cfg]; n = len(rs)
        f = lambda k: sum(1 for r in rs if str(r.get(k, "")).strip() in ("1", "True", "true")) / n if n else 0.0
        out[cfg] = {"turns": n, "head_hop_final": f("head_hop"), "head_hop_raw": f("head_hop_raw"), "tense_ok": f("tense_ok_raw"),
                    "repeat_per_1000": st.mean(float(r["repeat_rate_per_1000"] or 0) for r in rs) if n else 0.0,
                    "words": st.mean(float(r["words"] or 0) for r in rs) if n else 0.0,
                    "opener_reuse": opener[cfg][0] / opener[cfg][1] if opener[cfg][1] else 0.0,
                    "stock_per_1000": st.mean(float(r["stock_per_1000"] or 0) for r in rs) if n else 0.0}
    return out
builds = []
for spec in a.builds:
    label, runs = spec.split("=", 1); runs = [r for r in runs.split(",") if r]
    per = [run_metrics(r) for r in runs]
    agg = {}
    for cfg in ("base", "lora"):
        agg[cfg] = {}
        for k in per[0][cfg]:
            vals = [p[cfg][k] for p in per]
            agg[cfg][k] = {"mean": st.mean(vals), "min": min(vals), "max": max(vals)}
    builds.append({"label": label, "runs": runs, "agg": agg})
os.makedirs(os.path.join(a.out, "graphs"), exist_ok=True)
json.dump(builds, open(os.path.join(a.out, "comparison.json"), "w"), indent=1)
METRICS = [("head_hop_final", "Head-hop in the final reply (lower is better)", True, True), ("repeat_per_1000", "Repeated 4-grams per 1,000 words (lower is better)", False, False),
           ("opener_reuse", "Reply opens like one of the last 5 replies (lower is better)", True, False), ("tense_ok", "Tense consistent in the model's own text (higher is better)", True, True),
           ("words", "Average words per reply", False, False)]
PAL = ["#8a8f98", "#e8731a", "#6a4cf5", "#1f9d62"]
for cfg, title in (("lora", "With the Anti-H LoRA"), ("base", "Base model")):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.6))
    for ax, (k, lab, pct, _) in zip(axes, METRICS):
        for i, b in enumerate(builds):
            m = b["agg"][cfg][k]; sc = 100 if pct else 1
            ax.bar(i, m["mean"] * sc, color=PAL[i % len(PAL)]); ax.errorbar(i, m["mean"] * sc, yerr=[[m["mean"] * sc - m["min"] * sc], [m["max"] * sc - m["mean"] * sc]], color="#14233f", capsize=4)
            ax.text(i, m["mean"] * sc, (f"{m['mean']*sc:.0f}%" if pct else f"{m['mean']*sc:.0f}"), ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(builds))); ax.set_xticklabels([f"build {i+1}" for i in range(len(builds))]); ax.set_title(lab, fontsize=9, wrap=True)
    fig.suptitle(f"{title}: three builds of the app, same scripted turns", fontsize=12); fig.tight_layout()
    fig.savefig(os.path.join(a.out, "graphs", f"growth_{cfg}.png"), dpi=140); plt.close(fig)
lines = ["# Build-to-build comparison", "", "Builds (same benchmark matrix; builds 2 and 3 use the same run seed as run 1, so the poses are identical):"]
for i, b in enumerate(builds): lines.append(f"- build {i+1}: {b['label']} (runs: {', '.join(os.path.basename(r) for r in b['runs'])})")
for cfg in ("lora", "base"):
    lines += ["", f"## {cfg}", "", "| metric | " + " | ".join(f"build {i+1}" for i in range(len(builds))) + " |", "|---|" + "---|" * len(builds)]
    for k, lab, pct, _ in METRICS:
        cells = []
        for b in builds:
            m = b["agg"][cfg][k]; sc = 100 if pct else 1
            cells.append(f"{m['mean']*sc:.1f}{'%' if pct else ''}" + (f" ({m['min']*sc:.0f} to {m['max']*sc:.0f})" if len(b["runs"]) > 1 else ""))
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
open(os.path.join(a.out, "COMPARISON.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
