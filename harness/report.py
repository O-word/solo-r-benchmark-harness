"""REPORT.md and graphs from a finished run folder (reads scores.csv, sessions.jsonl, incidents.jsonl, config.json).

Honesty rules (GOAL_AND_EVALUATION.md): counts with intervals, never a bare percentage; failures shown, not only
averages; player bot and judge labelled model-generated; judge calibrated against hand labels.
"""
import csv
import json
import math
import os

from .labels import judge_vs_deterministic
from . import runview
from .stats import mean, wilson
from .util import read_jsonl

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7A5CFA", "#555555"]
PERSPECTIVES = ["first", "second", "third"]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(run_dir):
    sessions_all = runview.load_sessions(run_dir)
    valid = runview.valid_attempts(sessions_all)
    all_rows = runview.load_rows(run_dir, only_valid=False)
    rows = [r for r in all_rows if valid.get(r["session_id"]) == int(r.get("attempt") or 1)]
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    sessions = runview.valid_session_summaries(sessions_all)
    incidents = read_jsonl(os.path.join(run_dir, "incidents.jsonl"))
    end = {}
    for name in ("run_end.json", "run_status.json"):
        if os.path.exists(os.path.join(run_dir, name)):
            end = json.load(open(os.path.join(run_dir, name)))
            break
    return rows, cfg, sessions, incidents, end, all_rows, sessions_all


def rate_str(k, n):
    if n == 0:
        return "n/a (0 turns)"
    p, lo, hi = wilson(k, n)
    return "%d of %d (%.0f%%, 95%% CI %.0f-%.0f%%)" % (k, n, 100 * p, 100 * lo, 100 * hi)


def count(rows, col, val="1"):
    sel = [r for r in rows if r.get(col) not in ("", None)]
    return sum(1 for r in sel if r[col] == val), len(sel)


def md_table(header, body):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in body:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def build_report(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cfg, sessions, incidents, end, all_rows, sessions_all = load(run_dir)
    has_judge = any(r.get("judge_status") == "ok" for r in rows)
    main = [r for r in rows if r["phase"] == "main" and r["status"] == "ok"]
    confs = [c["id"] for c in cfg["config"]["configurations"] if any(r["configuration"] == c["id"] for r in main)]
    color = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(confs)}
    levels = sorted({int(r["level"]) for r in main})
    gdir = os.path.join(run_dir, "graphs")
    is_stub = cfg["config"]["mode"] == "stub" or (cfg["config"]["mode"] == "external" and all(int(c["server"]["port"]) == 1237 for c in cfg["config"]["configurations"]))
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
    foot = "%s | n = scored main-phase turns | player bot: %s | judge: %s%s" % (
        cfg["run_id"], cfg["player_bot"].get("model"), cfg["judge"].get("model") if has_judge or cfg["judge"].get("kind") != "none" else "none (deterministic scoring + hand labels)", "  | STUB DRY RUN: numbers test the pipeline, not a model" if is_stub else "")

    def finish(fig, name, title):
        fig.suptitle(title, fontsize=11, x=0.01, ha="left")
        fig.text(0.01, 0.005, foot, fontsize=6.5, color="#555555", ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(os.path.join(gdir, name))
        plt.close(fig)

    def rate_bars(ax, groups, getter, ylabel, ylim=(0, 1.15)):
        """groups: list of x labels; getter(conf, xlabel) -> (k, n)."""
        w = 0.8 / max(1, len(confs))
        for i, c in enumerate(confs):
            xs, ys, lo, hi, labels = [], [], [], [], []
            for j, g in enumerate(groups):
                k, n = getter(c, g)
                if not n:
                    continue                      # nothing was run for this configuration in this cell: draw nothing
                p, l, h = wilson(k, n)
                xs.append(j + (i - (len(confs) - 1) / 2) * w)
                ys.append(p); lo.append(max(0.0, p - l)); hi.append(max(0.0, h - p))
                labels.append("%d/%d" % (k, n))
            ax.bar(xs, ys, w * 0.92, color=color[c], label=c, yerr=[lo, hi] if xs else None, capsize=2, error_kw={"lw": 0.8})
            for x, y, h, t in zip(xs, ys, hi, labels):
                ax.text(x, min(y + h + 0.02, 1.08), t, ha="center", va="bottom", fontsize=6, rotation=90)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)

    # ---- 1. personality lock by turn + mean by configuration
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    turns = sorted({int(r["turn_number"]) for r in main})
    if not has_judge:
        # no judge configured: deterministic proxy = share of turns with no persona-softening / hard-rule pattern hit
        for c in confs:
            xs, ys, lo, hi = [], [], [], []
            for t in turns:
                sub = [r for r in main if r["configuration"] == c and int(r["turn_number"]) == t]
                n = len(sub)
                k = sum(1 for r in sub if r["soften_hit"] == "0" and r["hard_rule_violations"] == "0")
                if n:
                    p_, l_, h_ = wilson(k, n)
                    xs.append(t); ys.append(p_); lo.append(l_); hi.append(h_)
            a1.plot(xs, ys, marker="o", ms=4, color=color[c], label=c)
            a1.fill_between(xs, lo, hi, color=color[c], alpha=0.12)
        a1.set_ylim(0, 1.05); a1.set_xticks(turns); a1.set_xlabel("turn number in session")
        a1.set_ylabel("share of turns with no softening / hard-rule pattern"); a1.legend(fontsize=7, frameon=False)
        a1.set_title("DETERMINISTIC PROXY (no judge configured): persona held, by turn (band: Wilson 95% CI)", fontsize=8)
        vals = []
        for c in confs:
            sub = [r for r in main if r["configuration"] == c]
            vals.append(sum(1 for r in sub if r["soften_hit"] == "0" and r["hard_rule_violations"] == "0") / len(sub) if sub else 0)
        a2.bar(range(len(confs)), vals, color=[color[c] for c in confs]); a2.set_xticks(range(len(confs))); a2.set_xticklabels(confs, rotation=20, ha="right", fontsize=7)
        a2.set_ylim(0, 1.1); a2.set_title("Mean by configuration (proxy)", fontsize=9)
        for i, v in enumerate(vals):
            a2.text(i, v + 0.02, "%.2f" % v, ha="center", fontsize=7)
        finish(fig, "01_personality_lock_by_turn.png", "Target 1 - Personality lock (deterministic proxy; judge score needs an API key)")
        fig = None
    for c in (confs if has_judge else []):
        ys, es, ns = [], [], []
        for t in turns:
            v = [_f(r["judge_personality_lock"]) for r in main if r["configuration"] == c and int(r["turn_number"]) == t and _f(r["judge_personality_lock"]) is not None]
            ys.append(mean(v) if v else float("nan"))
            es.append((math.sqrt(sum((x - mean(v)) ** 2 for x in v) / max(1, len(v) - 1)) / math.sqrt(len(v))) if len(v) > 1 else 0)
            ns.append(len(v))
        a1.errorbar(turns, ys, yerr=es, marker="o", ms=4, lw=1.4, capsize=2, color=color[c], label="%s (n=%d)" % (c, sum(ns)))
    if has_judge:
        a1.set_xlabel("turn number in session"); a1.set_ylabel("mean judge personality lock (1-5)"); a1.set_ylim(0.8, 5.2); a1.set_xticks(turns); a1.legend(fontsize=7, frameon=False)
        a1.set_title("Drift over a session (mean +/- SE)", fontsize=9)
        vals = []
        for c in confs:
            v = [_f(r["judge_personality_lock"]) for r in main if r["configuration"] == c and _f(r["judge_personality_lock"]) is not None]
            vals.append(mean(v) if v else 0)
        a2.bar(range(len(confs)), vals, color=[color[c] for c in confs]); a2.set_xticks(range(len(confs))); a2.set_xticklabels(confs, rotation=20, ha="right", fontsize=7)
        a2.set_ylim(0, 5.2); a2.set_title("Mean by configuration", fontsize=9)
        for i, v in enumerate(vals):
            a2.text(i, v + 0.05, "%.2f" % v, ha="center", fontsize=7)
        finish(fig, "01_personality_lock_by_turn.png", "Target 1 - Personality lock")

    # ---- 2. head-hop by perspective x level (final, then raw)
    for col, fname, label in (("head_hop", "02_head_hop_by_perspective_level.png", "final reply (what the player sees)"), ("head_hop_raw", "02b_head_hop_by_perspective_level_raw_model_output.png", "raw model output (before the app's guards)")):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
        for ax, p in zip(axes, PERSPECTIVES):
            rate_bars(ax, ["L%d" % l for l in levels], lambda c, g, p=p, col=col: count([r for r in main if r["configuration"] == c and r["perspective"] == p and "L" + r["level"] == g], col), "head-hop rate" if p == "first" else "")
            ax.set_title("%s person" % p, fontsize=9)
        axes[0].legend(fontsize=7, frameon=False, loc="upper left")
        finish(fig, fname, "Target 2 - Head lock: head-hop rate, %s (bars: Wilson 95%% CI; k/n over each bar)" % label)

    # ---- 3. writing metrics
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7))
    lv = ["L%d" % l for l in levels]
    sel = lambda c, g: [r for r in main if r["configuration"] == c and "L" + r["level"] == g]
    rate_bars(axes[0][0], lv, lambda c, g: count(sel(c, g), "para_meets_level"), "share of replies"); axes[0][0].set_title("meets the level's paragraph minimum", fontsize=9)
    rate_bars(axes[0][1], lv, lambda c, g: count(sel(c, g), "unmatched_quotes"), "share of replies"); axes[0][1].set_title("unmatched quotes / brackets", fontsize=9)
    rate_bars(axes[0][2], lv, lambda c, g: (sum(1 for r in sel(c, g) if (r["stock_phrases"] or "0") != "0"), len(sel(c, g))), "share of replies"); axes[0][2].set_title("contains a stock phrase", fontsize=9)
    for ax, col, ttl, ylim in ((axes[1][0], "repeat_rate_per_1000", "repeated 4-grams per 1,000 words (mean)", None), (axes[1][1], "mattr", "vocabulary variety, MATTR-50 (mean)", (0, 1.05)), (axes[1][2], "judge_writing_quality", "judge writing quality 1-5 (mean)", (0, 5.3))):
        w = 0.8 / max(1, len(confs))
        for i, c in enumerate(confs):
            ys = []
            for g in lv:
                v = [_f(r[col]) for r in sel(c, g) if _f(r[col]) is not None]
                ys.append(mean(v) if v else 0)
            ax.bar([j + (i - (len(confs) - 1) / 2) * w for j in range(len(lv))], ys, w * 0.92, color=color[c], label=c)
        ax.set_xticks(range(len(lv))); ax.set_xticklabels(lv); ax.set_title(ttl, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    finish(fig, "03_writing_metrics.png", "Target 3 - Writing ability (sub-metrics by Roleplay Style Level and configuration)")

    # ---- 4. adherence by turn
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for c in confs:
        xs, ys, lo, hi = [], [], [], []
        for t in turns:
            sub = [r for r in main if r["configuration"] == c and int(r["turn_number"]) == t]
            k, n = count(sub, "adherence_ok")
            if n:
                p, l, h = wilson(k, n)
                xs.append(t); ys.append(p); lo.append(l); hi.append(h)
        ax.plot(xs, ys, marker="o", ms=4, color=color[c], label=c)
        ax.fill_between(xs, lo, hi, color=color[c], alpha=0.12)
    ax.set_ylim(0, 1.05); ax.set_xticks(turns); ax.set_xlabel("turn number in session"); ax.set_ylabel("share of turns obeying all rules and preferences"); ax.legend(fontsize=7, frameon=False)
    finish(fig, "04_rules_adherence_by_turn.png", "Target 4 - Adherence to Hard Rules, Important Notes, preferences, perspective and tense (band: Wilson 95% CI)")

    # ---- 5. four-target summary
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    targets = ["Personality lock\n(judge 1-5 / 5)" if has_judge else "Personality lock\n(no-softening share;\nproxy, no judge)", "Head lock\n(1 - head-hop rate)", "Writing ability\n(index)", "Rule adherence\n(share of turns)"]
    w = 0.8 / max(1, len(confs))
    for i, c in enumerate(confs):
        sub = [r for r in main if r["configuration"] == c]
        pl = [_f(r["judge_personality_lock"]) for r in sub if _f(r["judge_personality_lock"]) is not None]
        k_h, n_h = count(sub, "head_hop")
        k_a, n_a = count(sub, "adherence_ok")
        k_p, n_p = count(sub, "para_meets_level"); k_q, n_q = count(sub, "unmatched_quotes")
        jw = [_f(r["judge_writing_quality"]) for r in sub if _f(r["judge_writing_quality"]) is not None]
        parts = [k_p / n_p if n_p else 0, 1 - (k_q / n_q if n_q else 0)] + ([mean(jw) / 5] if jw else [])
        pers = (mean(pl) / 5 if pl else 0) if has_judge else (sum(1 for r in sub if r["soften_hit"] == "0" and r["hard_rule_violations"] == "0") / len(sub) if sub else 0)
        vals = [pers, 1 - (k_h / n_h if n_h else 0), mean(parts), k_a / n_a if n_a else 0]
        xs = [j + (i - (len(confs) - 1) / 2) * w for j in range(4)]
        ax.bar(xs, vals, w * 0.92, color=color[c], label="%s (n=%d)" % (c, len(sub)))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, "%.2f" % v, ha="center", fontsize=7)
        if n_h:
            p, l, h = wilson(n_h - k_h, n_h)
            ax.errorbar([xs[1]], [p], yerr=[[p - l], [h - p]], color="black", lw=0.8, capsize=2)
    ax.set_xticks(range(4)); ax.set_xticklabels(targets); ax.set_ylim(0, 1.12); ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_ylabel("higher is better (all four scaled 0-1)")
    finish(fig, "05_four_target_summary.png", "The four targets by configuration (writing index = mean of paragraph pass rate, clean-quote rate, judge writing/5)")

    # ---- 6. session level
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, c in enumerate(confs):
        rates = []
        for s in sessions:
            if s["configuration"] == c and s.get("head_hop_flags"):
                rates.append(sum(s["head_hop_flags"]) / len(s["head_hop_flags"]))
        a1.scatter([i + (j % 7 - 3) * 0.04 for j in range(len(rates))], rates, color=color[c], s=18, alpha=0.7)
    a1.set_xticks(range(len(confs))); a1.set_xticklabels(confs, rotation=20, ha="right", fontsize=7); a1.set_ylim(-0.05, 1.05)
    a1.set_ylabel("head-hop rate within the session (raw model output)"); a1.set_title("One dot per session (session is the unit)", fontsize=9)
    for i, c in enumerate(confs):
        n_s = sum(1 for s_ in sessions if s_["configuration"] == c)
        k_s = sum(1 for s_ in sessions if s_["configuration"] == c and s_.get("bad_session"))
        p_, l_, h_ = wilson(k_s, n_s) if n_s else (0, 0, 0)
        a2.bar([i], [p_], 0.6, color=color[c], yerr=[[max(0, p_ - l_)], [max(0, h_ - p_)]], capsize=3, error_kw={"lw": 0.8})
        a2.text(i, min(h_ + 0.03, 1.08), "%d/%d" % (k_s, n_s), ha="center", fontsize=7)
    a2.set_xticks(range(len(confs))); a2.set_xticklabels(confs, rotation=20, ha="right", fontsize=7); a2.set_ylim(0, 1.15)
    a2.set_ylabel("share of sessions that went bad (Wilson 95% CI)")
    a2.set_title("Sessions with a streak of 3+ consecutive head-hops", fontsize=9)
    finish(fig, "06_session_level_head_hop.png", "Bad-instance view: session-level head-hop and bad-session counts")

    # ---- 7. guard effect
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    rate_bars(a1, ["raw model output", "final reply"], lambda c, g: count([r for r in main if r["configuration"] == c], "head_hop_raw" if g.startswith("raw") else "head_hop"), "head-hop rate")
    a1.set_title("Head-hop: before vs after the app's guards", fontsize=9); a1.legend(fontsize=7, frameon=False)
    rate_bars(a2, ["raw model output", "final reply"], lambda c, g: (sum(1 for r in main if r["configuration"] == c and r["para_meets_level" + ("_raw" if g.startswith("raw") else "")] == "0"), sum(1 for r in main if r["configuration"] == c)), "share below the paragraph minimum")
    a2.set_title("Too few paragraphs: before vs after the guards", fontsize=9); a2.get_legend() and a2.get_legend().remove()
    finish(fig, "07_guard_effect.png", "What the reply guards change (guards-off configurations show the same bars on both sides)")

    # ================================================================ REPORT.md
    L = []
    L.append("# Benchmark report: %s" % cfg["run_id"])
    L.append("")
    if is_stub:
        L.append("> **DRY RUN against a STUB server and a STUB player bot.** No language model, no Gemini call. These numbers exercise the pipeline; they say nothing about any model.")
        L.append("")
    L.append("Status: **%s**. Created %s. Harness %s. Elapsed %ss (all invocations)." % (end.get("status", "?"), cfg["created"], cfg["harness_version"], end.get("elapsed_s_total")))
    L.append("")
    L.append("## 1. What was run")
    app = cfg["app_snapshot"]
    L.append("- App code: pinned snapshot, `index.html` SHA-256 `%s` (tree `%s`, %d files). The app's own `preload.cjs` is not used (see isolation)." % (app["index_html_sha256"], app["tree_sha256"][:16] + "...", app["files"]))
    L.append("- Electron %s (`%s`)." % (end.get("electron_version"), cfg["electron"]["binary"]))
    L.append("- Matrix: perspectives %s x Roleplay Style Levels %s x scenarios %s x %d turns x configurations %s; replicates %s; narration tense: %s. Sessions planned %s, turns recorded (all attempts) %s." % (
        cfg["matrix"]["perspectives"], cfg["matrix"]["levels"], cfg["matrix"]["scenarios"], cfg["matrix"]["turns"], cfg["matrix"]["configurations"], cfg["matrix"]["replicates"], cfg["config"]["tense"], end.get("sessions_planned"), end.get("turns_recorded")))
    L.append("- Seeds: run seed %s; per-session and per-turn seeds are in `config.json`. Temperatures: %s." % (cfg["seeds"]["run_seed"], cfg["temperatures"]))
    L.append("- Player bot: **%s** (model-generated: %s). Judge: **%s** (rubric v%s, SHA-256 `%s...`)." % (cfg["player_bot"].get("kind"), cfg["player_bot"].get("model"), cfg["judge"].get("kind"), cfg["rubric"]["version"], cfg["rubric"]["sha256"][:12]))
    planned_ids = cfg["matrix"]["session_order"]
    counted = {x["session_id"] for x in sessions}
    unc = [i for i in planned_ids if i not in counted]
    extra_attempts = sum(1 for x in sessions_all if x.get("status") not in ("complete", "incomplete"))
    L.append("- Sessions counted in this report: %d of %d planned%s. Attempts that were interrupted/crashed or superseded by a re-run stay in the files but are excluded: %d such attempt record(s)." % (
        len(counted), len(planned_ids), (" (not yet counted: %s)" % ", ".join(unc[:6]) + (" ..." if len(unc) > 6 else "")) if unc else "", extra_attempts))
    L.append("- Configurations:")
    for c in cfg["config"]["configurations"]:
        L.append("  - `%s`: %s; guards %s; model name sent `%s`; temperature %s; %s." % (c["id"], c.get("label", ""), "ON" if c.get("guards", True) else "OFF", c.get("model_name") or c.get("model"), c.get("temperature"),
                                                                                                                                           "external server on tunnel port %s (not started or stopped by the harness)" % c["server"]["port"] if c["server"]["kind"] == "external" else "instance policy `%s`" % c.get("instance_policy", "fresh_per_session")))
    L.append("")
    L.append("## 2. Isolation and integrity")
    blocked = []
    for s in sessions:
        blocked += s.get("blocked_requests", [])
    L.append("- Own throwaway Electron profile per session, inside this run folder (deleted after each session). The owner's profile, saves, shards, chat logs and Library were never opened.")
    if cfg["config"]["mode"] == "external":
        ext_ports = sorted({str(c["server"]["port"]) for c in cfg["config"]["configurations"]})
        L.append("- EXTERNAL model server reached through local port(s) %s (an SSH tunnel to a server the harness does not control): it was never started, stopped or signalled by the harness, and the tunnel was never opened or closed by it. Health = GET /v1/models only. Ports 1234/1235 are refused in code and were never contacted." % ", ".join(ext_ports))
        for pt, info in (cfg.get("external_servers") or {}).items():
            L.append("  - port %s: reachable at start = %s; served models at start: %s; configurations: %s." % (pt, info.get("reachable_at_start"), ", ".join(info.get("served_models_at_start") or []) or "none", ", ".join("%s -> `%s`" % (x["id"], x["model_name"]) for x in info.get("configurations", []))))
        pc = end.get("post_run_checks", {})
        L.append("  - After the run: no stray local processes = %s; external port(s) still answering = %s." % (pc.get("no_stray_local_processes"), {k: v.get("still_answering") for k, v in (pc.get("external_ports") or {}).items()}))
    else:
        L.append("- Own model server on port %s only; ports 1234/1235 are refused in code and were never contacted." % ("1237 (stub)" if is_stub else "1236"))
    L.append("- Off-port network self-test: %s." % (end.get("isolation", {}).get("off_port_probe") or "not run"))
    L.append("- Requests the app attempted that the harness cancelled before any network use: %d (the app's built-in default endpoint `http://127.0.0.1:1234/...` is tried once at boot; the filter drops it before the network is touched, so the owner's server never sees it) %s" % (len(blocked), "- " + ", ".join(sorted(set(blocked)))[:200] if blocked else ""))
    L.append("- After the run: port 1236 free = %s, port 1237 free = %s." % (end.get("post_run_checks", {}).get("port_1236_free"), end.get("post_run_checks", {}).get("port_1237_free")))
    L.append("- `MANIFEST.sha256` lists a SHA-256 for every file; this folder is append-only.")
    L.append("")

    L.append("## 3. Target 2 - Head lock (head-hop)")
    L.append("Final reply (what the player sees). Counts with 95% Wilson intervals; never a bare percentage.")
    body = []
    for c in confs:
        for p in PERSPECTIVES:
            for l in levels:
                sub = [r for r in main if r["configuration"] == c and r["perspective"] == p and int(r["level"]) == l]
                if sub:
                    kf, nf = count(sub, "head_hop"); kr, nr = count(sub, "head_hop_raw")
                    body.append([c, p, "L%d" % l, rate_str(kf, nf), rate_str(kr, nr)])
    L.append(md_table(["configuration", "perspective", "level", "final reply", "raw model output"], body))
    L.append("")
    body = []
    for c in confs:
        sub = [r for r in main if r["configuration"] == c]
        kf, nf = count(sub, "head_hop"); kr, nr = count(sub, "head_hop_raw")
        ks, ns = count(sub, "head_hop_soft_final")
        body.append([c, rate_str(kf, nf), rate_str(kr, nr), "%d of %d" % (ks, ns)])
    L.append("All turns per configuration:")
    L.append(md_table(["configuration", "final", "raw", "soft flags (pronoun-only reads; not counted)"], body))
    L.append("")
    L.append("### Sessions (the unit for bad instances)")
    body = []
    for c in confs:
        ss = [s for s in sessions if s["configuration"] == c]
        bad = sum(1 for s in ss if s.get("bad_session"))
        anyhop = sum(1 for s in ss if s.get("head_hop_flags") and sum(s["head_hop_flags"]) > 0)
        body.append([c, len(ss), rate_str(bad, len(ss)), rate_str(anyhop, len(ss)), sum(1 for s in ss if s["status"] != "complete")])
    L.append(md_table(["configuration", "sessions", "went bad (3+ consecutive head-hops)", "had at least one head-hop", "incomplete/crashed"], body))
    L.append("")
    L.append("Per-turn percentages overstate how independent turns are: violations inside a bad instance are correlated. A real bad-instance *rate* needs hundreds of sessions; this run can only show the detector and the restart-and-replay check work.")
    L.append("")
    L.append("### Incidents")
    if not incidents:
        L.append("No incidents recorded.")
    for inc in incidents:
        rp = inc["replay"]
        L.append("- **%s** in `%s` (config `%s`, %s person, L%s, scenario `%s`): %d consecutive head-hops starting at turn %d; instance `%s` started %s, age %ss at streak start; server args `%s`. Fresh restart + replay: %s." % (
            inc["incident_id"], inc["session_id"], inc["config"]["configuration"], inc["config"]["perspective"], inc["config"]["level"], inc["config"]["scenario"], inc["streak_len_at_detection"],
            inc["streak_start_turn_index"] + 1, inc["instance"]["instance_id"], inc["instance"]["start_time"], inc["instance"]["age_s_at_streak_start"], " ".join(inc["instance"]["server_args"])[:160],
            ("cleared = %s (longest run in replay window %s, replay flags %s)" % (rp["cleared"], rp["longest_hop_run_in_replay_window"], rp["replay_hop_flags"])) if rp.get("attempted") else "not attempted (%s)" % rp.get("reason")))
        if inc["in_common_with_earlier"]:
            L.append("  - In common with earlier incidents: %s" % inc["in_common_with_earlier"])
    L.append("")
    L.append("### Failures (not only averages): flagged head-hop replies")
    shown = 0
    tr_idx = {}
    for r in read_jsonl(os.path.join(run_dir, "transcripts.jsonl")):
        if r.get("event") == "score_detail" and r.get("phase") == "main" and r.get("head_hop_hits"):
            tr_idx[(r["session_id"], r["turn_index"])] = r["head_hop_hits"]
    for (sid, ti), hits in list(tr_idx.items())[:8]:
        L.append("- `%s` turn %d: rule `%s`, matched \"%s\"" % (sid, ti + 1, hits[0]["rule"], hits[0]["match"][:110].replace("\n", " ")))
        shown += 1
    if not shown:
        L.append("None flagged in the final replies.")
    L.append("")

    L.append("## 4. Target 1 - Personality lock%s" % (" (judge, 1-5)" if has_judge else " (no judge configured: deterministic proxy only)"))
    body = []
    for c in confs:
        sub = [r for r in main if r["configuration"] == c]
        v = [_f(r["judge_personality_lock"]) for r in sub if _f(r["judge_personality_lock"]) is not None]
        bait = [_f(r["judge_personality_lock"]) for r in sub if r["bait_type"] == "persona" and _f(r["judge_personality_lock"]) is not None]
        ks = [r for r in sub if r["bait_type"] == "persona"]
        kk, nn = count(ks, "soften_hit")
        body.append([c, "%.2f (n=%d)" % (mean(v), len(v)) if v else "n/a", "%.2f (n=%d)" % (mean(bait), len(bait)) if bait else "n/a", rate_str(kk, nn)])
    if not has_judge:
        for r_ in body:
            r_[1] = r_[2] = "n/a (no judge)"
        L.append("No judge was configured, so there is no 1-5 personality score. The proxy below is the pattern detector: how often the reply softens the persona on the persona-bait turns (the bully test). Hand labels (`human_labels.csv`) carry the personality-lock judgement.")
    L.append(md_table(["configuration", "mean personality lock", "mean on persona-bait turns", "softened on persona-bait turns (detector)"], body))
    L.append("")
    L.append("## 5. Target 3 - Writing ability")
    body = []
    for c in confs:
        for l in levels:
            sub = [r for r in main if r["configuration"] == c and int(r["level"]) == l]
            if not sub:
                continue
            pm = count(sub, "para_meets_level"); pt = count(sub, "para_meets_target"); uq = count(sub, "unmatched_quotes")
            rr = [_f(r["repeat_rate_per_1000"]) for r in sub if _f(r["repeat_rate_per_1000"]) is not None]
            mt = [_f(r["mattr"]) for r in sub if _f(r["mattr"]) is not None]
            jw = [_f(r["judge_writing_quality"]) for r in sub if _f(r["judge_writing_quality"]) is not None]
            body.append([c, "L%d" % l, rate_str(*pm), rate_str(*pt), rate_str(*uq), "%.1f" % mean(rr) if rr else "n/a", "%.3f" % mean(mt) if mt else "n/a", "%.2f" % mean(jw) if jw else "n/a"])
    L.append(md_table(["configuration", "level", "meets level minimum (paragraphs)", "meets pose-length target", "unmatched quotes/brackets", "repeated 4-grams /1000w", "MATTR", "judge writing"], body))
    L.append("")
    L.append("## 6. Target 4 - Rules and preferences adherence")
    body = []
    for c in confs:
        sub = [r for r in main if r["configuration"] == c]
        a = count(sub, "adherence_ok"); h = sum(int(r["hard_rule_violations"] or 0) for r in sub); p = sum(int(r["pref_violations"] or 0) for r in sub)
        po = count(sub, "person_ok_raw"); to = count(sub, "tense_ok_raw")
        body.append([c, rate_str(*a), h, p, rate_str(*po), rate_str(*to)])
    L.append(md_table(["configuration", "turns obeying everything", "hard-rule hits", "preference hits", "person consistent (model's own text)", "tense consistent (model's own text)"], body))
    L.append("")
    L.append("Person and tense are judged on the model's own text; the app's sanitizer can strip quotation marks from the final reply and turn dialogue into prose. Hard rules, preferences and head-hop are judged on what the player sees (quote-scoped rules read quotes from the model's own text).")
    L.append("")
    L.append("## 7. Guards")
    body = []
    for c in confs:
        sub = [r for r in main if r["configuration"] == c]
        body.append([c, "ON" if sub and sub[0]["guards_on"] == "1" else "OFF", sum(1 for r in sub if r["guard_rewrote"] == "1"), sum(int(r["rewrite_count"] or 0) for r in sub), sum(1 for r in sub if r["guard_would_fire"]), len(sub)])
    L.append(md_table(["configuration", "guards", "turns rewritten", "rewrite calls", "turns where a guard predicate would fire", "turns"], body))
    L.append("")
    L.append("## 8. STASIS expectations (scene memory)")
    body = []
    for sc in sorted({r["scenario"] for r in main}):
        sub = [r for r in main if r["scenario"] == sc and r["stasis_ok"] != ""]
        if sub:
            body.append([sc, rate_str(*count(sub, "stasis_ok"))])
    L.append(md_table(["scenario", "expectation checks passed"], body) if body else "No STASIS expectations were due in the turns played (scribe-dependent expectations are skipped against the stub).")
    L.append("")
    L.append("## 9. Judge calibration")
    jd = judge_vs_deterministic(rows)
    if not has_judge:
        L.append("- No judge configured for this run (deterministic scoring + hand labels only). Judge columns are blank; nothing was invented.")
    else:
        L.append("- Judge vs deterministic head-hop detector (no human labels needed): %s." % (("agreement %.0f%% over %d turns, Cohen's kappa %.2f" % (100 * jd["observed_agreement"], jd["n"], jd["cohen_kappa"])) if jd.get("n") else "no judged turns"))
    L.append("- Hand labels: `human_labels.csv` (blind sample of ~50 turns). Fill it in, then run `python -m harness agreement --run <this folder> --labels <filled csv>`; agreement is written as an addendum and reported here later.")
    L.append("")
    L.append("## 10. Caveats")
    L.append("- Deterministic checks are heuristics (regex over text) with false positives and negatives; the judge covers ambiguous cases and is itself model-generated and unvalidated until calibrated on hand labels.")
    L.append("- The player bot is model-generated (or canned in a dry run); scenarios and companions are synthetic and SFW.")
    L.append("- `sanitizeAIText` and the other cleanups inside the app always run (they are not guards); `raw model output` columns are the model text before them.")
    L.append("- Session is the unit for bad-instance analysis; this run's session count is small.")
    L.append("")
    L.append("## Graphs")
    for fn in sorted(os.listdir(gdir)):
        if fn.endswith(".png") and not fn.startswith("._"):
            L.append("![%s](graphs/%s)" % (fn, fn))
    with open(os.path.join(run_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return os.path.join(run_dir, "REPORT.md")
