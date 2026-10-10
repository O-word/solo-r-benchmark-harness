#!/usr/bin/env python3
"""Aggregate several finished benchmark runs (same configuration, different seeds) into one report.

Usage: ./run.sh-style interpreter (needs matplotlib):
  python3 aggregate_runs.py --out aggregates/pilot_runpod_4run RUN_DIR [RUN_DIR ...]

Reads each run's scores.csv (the proof file), never edits anything in the run folders, and writes:
  REPORT.md, aggregate.json (every number used in the video script), graphs/*.png, SOURCES.txt (run folders and their hashes).
Per metric and configuration it reports: the mean of the per-run rates and their spread (min to max), the pooled count with a 95% Wilson interval, and per-run numbers.
Honest by construction: counts with intervals, run-to-run spread shown, nothing tuned.
"""
import argparse, csv, hashlib, json, math, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFS = ["base", "lora"]
COLORS = {"base": "#8a8f98", "lora": "#6a4cf5"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def complete_attempts(run):
    """(session_id, attempt) pairs whose session finished; turns from interrupted attempts are excluded (as the run's own report does)."""
    done = set()
    path = os.path.join(run, "sessions.jsonl")
    if not os.path.exists(path):
        return None
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("event") == "session" and e.get("status") == "complete":
            done.add((e["session_id"], str(e.get("attempt"))))
    return done


def load(run):
    rows = []
    done = complete_attempts(run)
    with open(os.path.join(run, "scores.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            if done is not None and (r.get("session_id"), str(r.get("attempt"))) not in done:
                continue
            rows.append(r)
    return rows


def fl(r, k, default=0.0):
    try:
        return float(r.get(k, default))
    except Exception:
        return default


# metric name -> (function of a row giving 0/1, which means "good" direction)
METRICS = {
    "head_hop_final": ("Head-hop in the final reply (lower is better)", lambda r: fl(r, "head_hop") >= 1),
    "head_hop_raw": ("Head-hop in the raw model output (lower is better)", lambda r: fl(r, "head_hop_raw") >= 1),
    "tense_consistent": ("Tense consistent in the model's own text (higher is better)", lambda r: fl(r, "tense_ok_raw") >= 1),
    "person_consistent": ("Person consistent in the model's own text (higher is better)", lambda r: fl(r, "person_ok_raw") >= 1),
    "unmatched_quotes": ("Unmatched quotation marks in the final reply, includes damage done by the app's own text cleanup (lower is better)", lambda r: fl(r, "unmatched_quotes") >= 1),
    "unmatched_quotes_raw": ("Unmatched quotation marks in the model's own raw output (lower is better)", lambda r: fl(r, "unmatched_quotes_raw") >= 1),
    "quote_misplaced": ("Quotation marks in the wrong place in the final reply: closing mark with no opener, an unclosed quote, or a dialogue tag inside the quote (lower is better)", lambda r: fl(r, "quote_misplaced") >= 1),
    "quote_misplaced_raw": ("Quotation marks in the wrong place in the model's own raw output (lower is better)", lambda r: fl(r, "quote_misplaced_raw") >= 1),
    "meets_paragraph_minimum": ("Meets the level's paragraph minimum (higher is better)", lambda r: fl(r, "para_meets_level") >= 1),
    "obeys_all_rules": ("Obeys every rule and preference in the turn (higher is better)", lambda r: fl(r, "adherence_ok") >= 1),
    "hard_rule_violation": ("Hard-rule violation (lower is better)", lambda r: fl(r, "hard_rule_violations") >= 1),
    "persona_softened_on_bait": ("Persona softened on persona-bait turns (lower is better)", lambda r: fl(r, "soften_hit") >= 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "graphs"), exist_ok=True)
    data = {}  # run -> rows
    for r in a.runs:
        data[os.path.abspath(r)] = load(r)
    agg = {"runs": [os.path.basename(r) for r in data], "metrics": {}, "by_level": {}, "by_perspective": {}, "repetition": {}, "words": {}}
    total_turns = {c: sum(1 for rows in data.values() for x in rows if x["configuration"] == c) for c in CONFS}
    agg["turns_per_configuration"] = total_turns

    def rate_block(filterfn, fn):
        res = {}
        for c in CONFS:
            per_run = []
            k = n = 0
            for rows in data.values():
                sel = [x for x in rows if x["configuration"] == c and filterfn(x)]
                if not sel:
                    continue
                kk = sum(1 for x in sel if fn(x))
                per_run.append(kk / len(sel))
                k += kk
                n += len(sel)
            lo, hi = wilson(k, n)
            res[c] = {"k": k, "n": n, "pooled": (k / n if n else None), "ci95": [lo, hi],
                      "run_rates": per_run, "mean_of_runs": (st.mean(per_run) if per_run else None),
                      "min_run": (min(per_run) if per_run else None), "max_run": (max(per_run) if per_run else None)}
        return res

    for key, (label, fn) in METRICS.items():
        only_bait = key == "persona_softened_on_bait"
        agg["metrics"][key] = {"label": label, **rate_block((lambda x: x.get("bait") == "1" and x.get("bait_type") == "persona") if only_bait else (lambda x: True), fn)}
    for lvl in ("1", "2", "3"):
        agg["by_level"][lvl] = {"meets_paragraph_minimum": rate_block(lambda x, l=lvl: x["level"] == l, METRICS["meets_paragraph_minimum"][1]),
                                "head_hop_final": rate_block(lambda x, l=lvl: x["level"] == l, METRICS["head_hop_final"][1])}
    for per in ("first", "second", "third"):
        agg["by_perspective"][per] = {"head_hop_final": rate_block(lambda x, p=per: x["perspective"] == p, METRICS["head_hop_final"][1]),
                                      "head_hop_raw": rate_block(lambda x, p=per: x["perspective"] == p, METRICS["head_hop_raw"][1])}
    for c in CONFS:
        vals = [fl(x, "repeat_rate_per_1000") for rows in data.values() for x in rows if x["configuration"] == c]
        words = [fl(x, "words") for rows in data.values() for x in rows if x["configuration"] == c]
        paras = [fl(x, "paragraphs") for rows in data.values() for x in rows if x["configuration"] == c]
        agg["repetition"][c] = {"mean_repeated_4grams_per_1000_words": (st.mean(vals) if vals else None)}
        agg["words"][c] = {"mean_words": (st.mean(words) if words else None), "mean_paragraphs": (st.mean(paras) if paras else None)}

    json.dump(agg, open(os.path.join(out, "aggregate.json"), "w"), indent=2)

    # ---------------------------------------------------------------- report
    def pct(x):
        return "n/a" if x is None else f"{100 * x:.0f}%"

    L = ["# Aggregate report: %d runs" % len(data), "",
         "Runs: " + ", ".join(agg["runs"]) + ". Turns per configuration: " + ", ".join(f"{c} {n}" for c, n in total_turns.items()) + ".",
         "Method: see BENCHMARK_METHODOLOGY.md. Each run used a different run seed; settings otherwise identical (run 1 on an RTX 6000 Ada, later runs on an RTX A6000: a hardware change).",
         "Reading it: 'mean of runs' is the plain average of the per-run rates; 'spread' is the lowest and highest single run; 'pooled' counts every turn together with a 95% Wilson interval.", ""]
    L += ["| Metric | Config | Mean of runs | Spread (min to max) | Pooled count | Pooled 95% CI |", "|---|---|---|---|---|---|"]
    for key, m in agg["metrics"].items():
        for c in CONFS:
            b = m[c]
            L.append(f"| {m['label']} | {c} | {pct(b['mean_of_runs'])} | {pct(b['min_run'])} to {pct(b['max_run'])} | {b['k']} of {b['n']} | {pct(b['ci95'][0])} to {pct(b['ci95'][1])} |")
    L += ["", "## Paragraph minimum by level", "| Level | Config | Mean of runs | Pooled |", "|---|---|---|---|"]
    for lvl, d in agg["by_level"].items():
        for c in CONFS:
            b = d["meets_paragraph_minimum"][c]
            L.append(f"| {lvl} | {c} | {pct(b['mean_of_runs'])} | {b['k']} of {b['n']} |")
    L += ["", "## Head-hop (final reply) by perspective", "| Perspective | Config | Mean of runs | Pooled |", "|---|---|---|---|"]
    for per, d in agg["by_perspective"].items():
        for c in CONFS:
            b = d["head_hop_final"][c]
            L.append(f"| {per} | {c} | {pct(b['mean_of_runs'])} | {b['k']} of {b['n']} |")
    L += ["", "## Length and repetition", "| Config | Mean words | Mean paragraphs | Repeated 4-grams per 1000 words |", "|---|---|---|---|"]
    for c in CONFS:
        L.append(f"| {c} | {agg['words'][c]['mean_words']:.0f} | {agg['words'][c]['mean_paragraphs']:.1f} | {agg['repetition'][c]['mean_repeated_4grams_per_1000_words']:.0f} |")
    L += ["", "## Caveats", "- Scripted player, deterministic scoring (no judge); small samples per cell; hardware changed between run 1 and runs 2 to 4; full-precision weights. A feasibility result, not a final claim."]
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(L) + "\n")

    with open(os.path.join(out, "SOURCES.txt"), "w") as f:
        for r in data:
            m = os.path.join(r, "MANIFEST.sha256")
            h = hashlib.sha256(open(m, "rb").read()).hexdigest() if os.path.exists(m) else "(no manifest)"
            f.write(f"{r}\n  manifest sha256: {h}\n")

    # ---------------------------------------------------------------- graphs
    def bars(keys, fname, title, ylabel="share of turns"):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        w = 0.36
        xs = range(len(keys))
        for i, c in enumerate(CONFS):
            vals = [agg["metrics"][k][c]["mean_of_runs"] or 0 for k in keys]
            lo = [(agg["metrics"][k][c]["mean_of_runs"] or 0) - (agg["metrics"][k][c]["min_run"] or 0) for k in keys]
            hi = [(agg["metrics"][k][c]["max_run"] or 0) - (agg["metrics"][k][c]["mean_of_runs"] or 0) for k in keys]
            ax.bar([x + (i - 0.5) * w for x in xs], vals, w, label=c, color=COLORS[c], yerr=[lo, hi], capsize=4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([agg["metrics"][k]["label"].split(" (")[0] for k in keys], rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "graphs", fname), dpi=150)
        plt.close(fig)

    bars(["head_hop_final", "head_hop_raw", "unmatched_quotes_raw", "hard_rule_violation", "persona_softened_on_bait"], "01_problems_lower_is_better.png", "Problems (lower is better): mean of runs, whiskers = lowest and highest run")
    bars(["tense_consistent", "person_consistent", "meets_paragraph_minimum", "obeys_all_rules"], "02_goods_higher_is_better.png", "Qualities (higher is better): mean of runs, whiskers = lowest and highest run")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, c in enumerate(CONFS):
        vals = [agg["by_level"][l]["meets_paragraph_minimum"][c]["mean_of_runs"] or 0 for l in ("1", "2", "3")]
        ax.bar([x + (i - 0.5) * 0.36 for x in range(3)], vals, 0.36, label=c, color=COLORS[c])
    ax.set_xticks(range(3)); ax.set_xticklabels(["Level 1", "Level 2", "Level 3"]); ax.set_ylim(0, 1)
    ax.set_title("Meets the paragraph minimum, by Roleplay Style Level"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out, "graphs", "03_paragraph_minimum_by_level.png"), dpi=150); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
