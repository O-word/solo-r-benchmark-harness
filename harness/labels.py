"""Hand-label sheet and agreement calculation.

write_label_sheet(): samples ~50 main-phase turns (stratified over perspective x level, seeded) into human_labels.csv.
The sheet is BLIND: it carries no configuration, guard setting, judge score or detector flag.
compute_agreement(): after the owner fills the human_* columns, compares his labels with the LLM judge and with the
deterministic scorer (Cohen's kappa for yes/no, weighted kappa + within-1 agreement for 1-5 scores).
"""
import csv
import json
import os
import random
import time

from .manifest import write_addendum
from .stats import cohen_kappa, mean, pearson, weighted_kappa
from .util import dump_json, read_jsonl

SHEET_COLUMNS = ["label_id", "session_id", "turn_index", "turn_id", "perspective", "level", "scenario", "companion", "persona_summary", "hard_rules", "player_preferences",
                 "player_pose", "companion_reply",
                 "human_head_hop", "human_personality_lock", "human_writing_quality", "human_rules_ok", "human_notes"]
INSTRUCTIONS = ("# human_labels.csv - fill the human_* columns. human_head_hop: 1 if the reply writes the PLAYER's words/actions/thoughts/reactions, else 0. "
                "human_personality_lock: 1-5 (5 = unmistakably the persona). human_writing_quality: 1-5. human_rules_ok: 1 if no Hard Rule/preference/perspective/tense slip, else 0. "
                "Rows starting with # are ignored. The sheet is blind: judge and detector results are not shown.")


def _load_rows(run_dir):
    from .runview import load_rows
    return load_rows(run_dir, only_valid=True)


def write_label_sheet(run_dir, lab_cfg):
    n = int(lab_cfg.get("sample", 50))
    seed = int(lab_cfg.get("seed", 7))
    rows = [r for r in _load_rows(run_dir) if r["phase"] == "main" and r["status"] == "ok"]
    if not rows:
        return None
    tr = {}
    for r in read_jsonl(os.path.join(run_dir, "transcripts.jsonl")):
        if r.get("event") or r.get("phase") != "main":
            continue
        tr[(r["session_id"], int(r.get("attempt", 1)), r["turn_index"])] = r
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    rng = random.Random(seed)
    groups = {}
    for r in rows:
        groups.setdefault((r["perspective"], r["level"]), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups)
    picked = []
    while len(picked) < min(n, len(rows)):
        progressed = False
        for k in keys:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(picked)
    comp_cache = {}
    path = os.path.join(run_dir, "human_labels.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(INSTRUCTIONS + "\n")
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for i, r in enumerate(picked, 1):
            t = tr.get((r["session_id"], int(r.get("attempt") or 1), int(r["turn_index"])))
            if not t:
                continue
            cname = t["companion"]
            cid = t.get("companion_id") or cname.lower()
            if cid not in comp_cache:
                from .config import load_companion
                comp_cache[cid] = load_companion(cid)
            comp = comp_cache[cid]
            w.writerow({"label_id": "L%03d" % i, "session_id": r["session_id"], "turn_index": r["turn_index"], "turn_id": r["turn_id"], "perspective": r["perspective"], "level": r["level"],
                        "scenario": r["scenario"], "companion": cname, "persona_summary": comp["checks"]["persona_summary"], "hard_rules": comp["fields"]["hardRules"].replace("\n", " | "),
                        "player_preferences": " | ".join(comp["preferences"]), "player_pose": t["player_pose"]["command"], "companion_reply": t["final_reply"], "human_head_hop": "",
                        "human_personality_lock": "", "human_writing_quality": "", "human_rules_ok": "", "human_notes": ""})
    return path


def _read_filled(path):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    for r in csv.DictReader(lines):
        out.append(r)
    return out


def _num(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def agreement_tables(pairs):
    """pairs: list of dict(name -> (human, other)). Returns metric dict per name."""
    res = {}
    for name, kind in (("head_hop", "binary"), ("rules_ok", "binary"), ("personality_lock", "ordinal"), ("writing_quality", "ordinal")):
        pr = [(h, o) for h, o in pairs.get(name, []) if h is not None and o is not None]
        if not pr:
            res[name] = {"n": 0}
            continue
        a, b = [p[0] for p in pr], [p[1] for p in pr]
        d = {"n": len(pr), "exact_agreement": sum(1 for x, y in pr if x == y) / len(pr)}
        if kind == "binary":
            k, po, _ = cohen_kappa(a, b)
            d["cohen_kappa"] = k
            d["human_positive"] = sum(a)
            d["other_positive"] = sum(b)
        else:
            d["within_1"] = sum(1 for x, y in pr if abs(x - y) <= 1) / len(pr)
            d["weighted_kappa_quadratic"] = weighted_kappa(a, b, [1, 2, 3, 4, 5])
            d["pearson"] = pearson(a, b)
            d["mean_human"], d["mean_other"] = mean(a), mean(b)
        res[name] = d
    return res


def compute_agreement(run_dir, labels_path, write=True):
    scores = {(r["session_id"], int(r["turn_index"])): r for r in _load_rows(run_dir) if r["phase"] == "main"}
    human = _read_filled(labels_path)
    vs_judge = {k: [] for k in ("head_hop", "rules_ok", "personality_lock", "writing_quality")}
    vs_det = {k: [] for k in ("head_hop", "rules_ok")}
    for h in human:
        s = scores.get((h["session_id"], int(h["turn_index"])))
        if not s:
            continue
        hh, rk, pl, wq = _num(h["human_head_hop"]), _num(h["human_rules_ok"]), _num(h["human_personality_lock"]), _num(h["human_writing_quality"])
        vs_judge["head_hop"].append((hh, _num(s.get("judge_head_hop"))))
        vs_judge["rules_ok"].append((rk, _num(s.get("judge_rules_ok"))))
        vs_judge["personality_lock"].append((pl, _num(s.get("judge_personality_lock"))))
        vs_judge["writing_quality"].append((wq, _num(s.get("judge_writing_quality"))))
        vs_det["head_hop"].append((hh, _num(s.get("head_hop"))))
        vs_det["rules_ok"].append((rk, _num(s.get("adherence_rules_only_ok"))))
    out = {"labels_file": os.path.basename(labels_path), "rows_in_sheet": len(human), "rows_labeled": sum(1 for h in human if _num(h["human_head_hop"]) is not None),
           "human_vs_llm_judge": agreement_tables(vs_judge), "human_vs_deterministic_scorer": agreement_tables(vs_det),
           "judge_model_note": "if the judge is the stub, these numbers describe a heuristic, not an LLM judge"}
    if write:
        ts = time.strftime("%Y%m%d_%H%M%S")
        p = os.path.join(run_dir, "agreement_%s.json" % ts)
        dump_json(p, out)
        copy = os.path.join(run_dir, "human_labels_filled_%s.csv" % ts)
        with open(labels_path, "rb") as src, open(copy, "wb") as dst:
            dst.write(src.read())
        out["written"] = [os.path.basename(p), os.path.basename(copy), os.path.basename(write_addendum(run_dir, [os.path.basename(p), os.path.basename(copy)]))]
    return out


def judge_vs_deterministic(rows):
    """Cross-check available without any human labels: how often the judge and the detector agree on head-hop."""
    pairs = [(_num(r.get("judge_head_hop")), _num(r.get("head_hop"))) for r in rows if r["phase"] == "main"]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
        return {"n": 0}
    k, po, n = cohen_kappa([a for a, _ in pairs], [b for _, b in pairs])
    return {"n": n, "observed_agreement": po, "cohen_kappa": k, "judge_positive": sum(a for a, _ in pairs), "detector_positive": sum(b for _, b in pairs)}
