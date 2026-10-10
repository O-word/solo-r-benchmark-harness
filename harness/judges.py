"""Judges: a deterministic stub and an LLM judge (Gemini). Both implement Judge.judge(ctx) -> dict.

The judge sees the persona, hard rules, preferences, the player's pose and the companion's FINAL reply. It does
NOT see the deterministic scorer's findings (so the two stay independent and agreement is meaningful).
A judge failure never raises: it returns status="error" so already-recorded turns are never lost.
"""
import hashlib
import json
import os
import re

from . import scoring
from .gemini_client import GeminiClient, GeminiError
from .util import HARNESS_ROOT, sha256_file

RUBRIC_PATH = os.path.join(HARNESS_ROOT, "rubric", "judge_rubric.md")


def rubric_info():
    return {"path": os.path.relpath(RUBRIC_PATH, HARNESS_ROOT), "sha256": sha256_file(RUBRIC_PATH),
            "version": re.search(r"version ([\d.]+)", open(RUBRIC_PATH, encoding="utf-8").readline()).group(1)}


class Judge:
    name = "judge"

    def describe(self):
        return {"kind": self.name}

    def judge(self, ctx):
        raise NotImplementedError


def _jitter(reply, tag, p):
    h = int(hashlib.sha256((tag + "|" + reply).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < p


class StubJudge(Judge):
    """Deterministic heuristic judge. NOT a real judge: it exists to exercise the pipeline offline."""
    name = "stub"

    def describe(self):
        return {"kind": "stub", "model": "stub-heuristic-v1", "note": "heuristic stand-in, not an LLM judge"}

    def judge(self, ctx):
        reply = ctx["reply"]
        checks = ctx["companion_checks"]
        hh = scoring.head_hop(reply, ctx["pose_text"], ctx["player"]["name"], ctx["player"].get("pronoun", "she"),
                              ctx["companion"]["name"], ctx["companion"].get("pronoun", "he"), ctx["perspective"])
        hard = scoring.pattern_violations(reply, checks.get("hard_rule_patterns"), ctx["player"]["name"])
        prefs = scoring.pattern_violations(reply, checks.get("preference_patterns"), ctx["player"]["name"])
        soft = scoring.soften_hits(reply, checks.get("soften_patterns"))
        pt = scoring.person_tense_check(reply, ctx["perspective"], ctx["tense"], ctx["player"]["name"], ctx["companion"]["name"])
        q = scoring.quote_problems(reply)
        paras = scoring.count_paragraphs(reply)
        pl = 5 - (2 if soft else 0) - (1 if hard else 0) - (1 if _jitter(reply, "pl", 0.3) else 0)
        wq = 5 - (2 if q else 0) - (1 if paras < ctx["min_paragraphs"] else 0) - (1 if scoring.stock_phrase_hits(reply) else 0) - (1 if _jitter(reply, "wq", 0.25) else 0)
        ra = 5 - (2 if hard else 0) - (1 if prefs else 0) - (1 if not (pt["person_ok"] and pt["tense_ok"]) else 0)
        hop = int(hh["flag"])
        if _jitter(reply, "hh", 0.06):
            hop = 1 - hop
        raw = {"head_hop": hop, "head_hop_confidence": "low" if hop != int(hh["flag"]) else "medium", "head_hop_evidence": (hh["hits"][0]["match"] if hh["hits"] else ""),
               "personality_lock": max(1, pl), "writing_quality": max(1, wq), "rules_adherence": max(1, ra),
               "rules_ok": int(not hard and not prefs), "reasoning": "stub judge: heuristic scores, not an LLM judgement"}
        return {"status": "ok", "parsed": raw, "raw": json.dumps(raw), "model": "stub-heuristic-v1", "prompt_sha256": "", "attempts": 0}


def build_user_prompt(ctx):
    c = ctx["companion"]
    lines = [
        "COMPANION: %s" % c["name"],
        "PERSONA SUMMARY: %s" % ctx["persona_summary"],
        "IMPORTANT NOTES: %s" % ctx["important_notes"],
        "HARD RULES:\n%s" % ctx["hard_rules"],
        "PLAYER PREFERENCES:\n%s" % "\n".join("- " + p for p in ctx["preferences"] if p),
        "NARRATION SETTING for the companion's prose: %s person, %s tense. Roleplay Style Level %s (minimum %d paragraph(s))."
        % (ctx["perspective"], ctx["tense"], ctx["level"], ctx["min_paragraphs"]),
        "PLAYER CHARACTER: %s" % ctx["player"]["name"],
        "SCENE BEAT: %s" % ctx["beat"],
        "PERSONA BAIT TURN: %s" % ("yes - the player is trying to make the companion soften" if ctx.get("bait_type") == "persona" else "no"),
        "PREVIOUS COMPANION REPLIES (oldest first, for drift context):\n%s" % ("\n---\n".join(ctx.get("previous_replies", [])) or "(none)"),
        "PLAYER POSE (what the player wrote this turn; %%r and %%t are line-break markers):\n<<<\n%s\n>>>" % ctx["pose_text"],
        "COMPANION REPLY TO EVALUATE:\n<<<\n%s\n>>>" % ctx["reply"],
        "Return the JSON object described in the rubric.",
    ]
    return "\n\n".join(lines)


def parse_judge_json(text):
    m = re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        raise ValueError("no JSON object in judge output")
    d = json.loads(m.group(0))
    out = {
        "head_hop": int(d["head_hop"]),
        "head_hop_confidence": str(d.get("head_hop_confidence", "")),
        "head_hop_evidence": str(d.get("head_hop_evidence", ""))[:300],
        "personality_lock": int(d["personality_lock"]),
        "writing_quality": int(d["writing_quality"]),
        "rules_adherence": int(d["rules_adherence"]),
        "rules_ok": int(d["rules_ok"]),
        "reasoning": str(d.get("reasoning", ""))[:800],
    }
    if out["head_hop"] not in (0, 1) or out["rules_ok"] not in (0, 1):
        raise ValueError("binary fields out of range")
    for k in ("personality_lock", "writing_quality", "rules_adherence"):
        if not 1 <= out[k] <= 5:
            raise ValueError(k + " out of range")
    return out


class GeminiJudge(Judge):
    """LLM judge using the Gemini API with the written rubric. UNTESTED against the live API (no key yet)."""
    name = "gemini"

    def __init__(self, model="gemini-2.5-flash", temperature=0.0, client=None, seed=1234):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.client = client or GeminiClient(model)
        self.system = open(RUBRIC_PATH, encoding="utf-8").read()

    def describe(self):
        return {"kind": "gemini", "model": self.model, "temperature": self.temperature, "seed": self.seed, "rubric": rubric_info(),
                "label": "model-generated judgement; calibrate against hand labels"}

    def judge(self, ctx):
        user = build_user_prompt(ctx)
        sha = hashlib.sha256((self.system + "\n" + user).encode()).hexdigest()
        try:
            res = self.client.generate(self.system, user, temperature=self.temperature, seed=self.seed, json_mode=True, max_output_tokens=3000, thinking_budget=512)
        except GeminiError as e:
            return {"status": "error", "error": str(e)[:300], "parsed": None, "raw": "", "model": self.model, "prompt_sha256": sha, "attempts": 0}
        try:
            parsed = parse_judge_json(res["text"])
        except Exception as e:
            return {"status": "unparseable", "error": str(e)[:200], "parsed": None, "raw": res["text"], "model": self.model, "prompt_sha256": sha, "attempts": res["attempts"]}
        return {"status": "ok", "parsed": parsed, "raw": res["text"], "model": self.model, "prompt_sha256": sha, "attempts": res["attempts"]}


class NullJudge(Judge):
    """No judge (deterministic scoring + hand labels only). Judge columns stay blank; nothing is invented."""
    name = "none"

    def describe(self):
        return {"kind": "none", "model": "none", "note": "no judge configured; deterministic scorers and the human-label sheet only"}

    def judge(self, ctx):
        return {"status": "none", "parsed": None, "raw": "", "model": "none", "prompt_sha256": "", "attempts": 0}


def make_judge(cfg):
    kind = (cfg or {}).get("kind", "stub")
    if kind == "none":
        return NullJudge()
    if kind == "stub":
        return StubJudge()
    if kind == "gemini":
        return GeminiJudge(model=cfg.get("model", "gemini-2.5-flash"), temperature=cfg.get("temperature", 0.0), seed=cfg.get("seed", 1234))
    raise ValueError("unknown judge kind %r" % kind)
