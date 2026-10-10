"""Deterministic scorers (stdlib only; no network, no model).

Everything here is a heuristic over text. It is meant to be cheap, repeatable and inspectable:
every flag carries the matched snippet so a human can check it, and the LLM judge covers the
ambiguous cases. Heuristics WILL have false positives and false negatives; the hand-label sheet and
the agreement report (labels.py) measure how far to trust them.

Scored per turn:
  head_hop            the reply writes the player's actions / speech / thoughts / reactions
  unmatched_quotes    odd quote count, mismatched curly quotes, unpaired brackets (app's own rule)
  paragraphs          against the level minimum (app's countParagraphs rule) and the pose-length target
  repeated_phrase     repeated 4-grams per 1,000 words (within reply and against earlier replies)
                      plus stock phrases per 1,000 words
  person / tense      prose outside quotes matches the saved narration perspective and tense
  hard rules / prefs  simple regex patterns from the companion file
  softening           persona-break patterns (the bully-test), meaningful on persona-bait turns
  STASIS              scene-memory expectations from the scenario file
"""
import math
import re

from .poses import conj
from .vendor import lane_lint

# ------------------------------------------------------------------ text utilities

QUOTE_RE = re.compile(r'"([^"\n]*)"|“([^”]*)”')


def norm(text):
    return (text or "").replace("\r\n", "\n").replace("’", "'").replace("‘", "'")


def split_quotes(text):
    """Return (prose, quotes): text with quoted spans blanked out, and the list of quoted strings."""
    text = norm(text)
    quotes = [m.group(1) if m.group(1) is not None else m.group(2) for m in QUOTE_RE.finditer(text)]
    prose = QUOTE_RE.sub(" ", text)
    return prose, quotes


def words(text):
    return re.findall(r"[A-Za-z][A-Za-z'\-]*", norm(text).lower())


def count_paragraphs(text):
    """The app's own rule: blank-line separated, non-empty chunks."""
    return len([c for c in re.split(r"\n\s*\n", norm(text)) if c.strip()])


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", norm(text)) if s.strip()]


# ------------------------------------------------------------------ app rules replicated

def required_paragraphs(level):
    return {"3": 3, "2": 2}.get(str(level), 1)


def describe_reply_length_target(pose_text, level):
    """Replica of the app's describeReplyLengthTarget(): the paragraph number the briefing asks for."""
    raw = pose_text or ""
    chars = len(raw)
    lines = len([l for l in raw.split("\n") if l])
    m = required_paragraphs(level)
    if chars > 1800 or lines >= 10:
        return max(m, 3)
    if chars > 900 or lines >= 6:
        return max(m, 2)
    return m


def parse_target_from_briefing(user_briefing):
    """The paragraph number the app actually wrote into the briefing ('Reply length target: ... at least N paragraph')."""
    m = re.search(r"Reply length target:[^\n]*?at least (\d+) paragraph", user_briefing or "")
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------ verb lexicon for head-hop checks

PLAYER_ACTION_LEMMAS = """
flinch gasp shiver tremble tense relax freeze nod agree accept obey scream cry sob fall collapse step retreat move walk run turn
look stare glance smile laugh grin blush swallow whisper say ask reply answer murmur mutter feel think wonder decide realize
want hesitate pause consider breathe sigh shrug reach grab pull push lean sit stand kneel drop hold clutch shake bite wince gulp
duck dodge dive roll jump flee panic nod frown glare scowl gesture point wave beckon follow lift raise lower close open touch
stumble trip slip stagger tilt shift fidget tremble quiver shudder cringe recoil cower beg plead refuse comply surrender give
take hand offer accept hug embrace kiss slap hit punch kick wipe rub scratch tap hurry rush hide crouch kneel lie wake sleep
blink gape stammer stutter whimper moan mumble exclaim cheer shout yell call speak tell promise swear admit confess know
remember forget understand notice see hear smell taste hope fear worry wish doubt believe trust love hate need enjoy
""".split()

HEDGES = r"(?:seemed|seems|looked|looks|appeared|appears|as if|as though|might|maybe|perhaps|could be|probably|I think|I guess|I can tell)"

BODY_NOUNS = r"(?:breath|pulse|heart(?:beat)?|voice|hands?|eyes?|knees|legs?|stomach|throat|cheeks|shoulders?|jaw|fingers?|lips|chest|face|grip|mind|thoughts)"


def _forms(lemma):
    out = {lemma, conj(lemma, "third", "present"), conj(lemma, "first", "past")}
    out.add(lemma + "ing" if not lemma.endswith("e") else lemma[:-1] + "ing")
    return {f for f in out if f}


_ACTION_FORMS = set()
for _l in PLAYER_ACTION_LEMMAS:
    _ACTION_FORMS |= _forms(_l)
ACTION_VERB_RE = "(?:" + "|".join(sorted((re.escape(f) for f in _ACTION_FORMS), key=len, reverse=True)) + ")"
ADVERBS = r"(?:\w+ly\s+|still\s+|then\s+|just\s+|slowly\s+|quickly\s+|barely\s+|nearly\s+)*"
VERBISH = re.compile(r"^(?:\w+ed|\w+ing|\w+s|shook|froze|went|ran|caught|hit|felt|stood|sat|knelt|fell|took|gave|held|said|saw|heard|knew|thought|began|came|made|put|let|set|cut|hurt|shut|spread|burst|swung|struck|stuck)$", re.I)


def _pose_words(pose):
    return set(words(pose or ""))


def _tokens(s):
    return re.findall(r"[a-z0-9']+", (s or "").lower())


def sentence_inside_quotes(sentence, quote_list, threshold=0.8):
    """True if (almost) every word of `sentence` also appears in one of the quoted spans: i.e. it is dialogue."""
    st = set(_tokens(sentence))
    if len(st) < 2:
        return False
    return any(st and len(st & set(_tokens(q))) / len(st) >= threshold for q in quote_list if q)


def _sentence_containing(text, snippet):
    key = snippet[:30]
    for sent in re.split(r"(?<=[.!?])\s+|\n+", norm(text)):
        if key and key in sent:
            return sent
    return snippet


def head_hop(reply, pose, player_name, player_pronoun="she", companion_name="", companion_pronoun="he", perspective="third", quote_sources=None):
    """Detect the reply writing the player's words, actions, thoughts or reactions.

    Returns dict(flag, hits, soft_hits, lane_high). `hits` count toward the flag; `soft_hits` (pronoun-only
    reads, hedged reads) are reported but do not set the flag.
    """
    text = norm(reply)
    artifact_sentences = []
    if quote_sources and text.count('"') >= max((src.count('"') for src in quote_sources if src), default=0):
        # Quotation marks are intact (the same count as the model's own text), so there is nothing to un-strip. Dropping "artifact" sentences here
        # would orphan quote marks and misread real dialogue as prose (found 2026-10-07 when the app's cleanup stopped deleting quotes).
        quote_sources = None
    if quote_sources:
        # The app's sanitizer can strip quotation marks, turning dialogue ("You want something?") into prose. A sentence that is
        # really inside a quote in the model's own text (raw reply / rewrite output) is an app artifact, not a head-hop: drop it
        # before the checks and report it.
        raw_quotes = [q for src in quote_sources for q in split_quotes(src)[1]]
        kept = []
        for sent in re.split(r"(?<=[.!?])\s+|\n+", text):
            if sent.strip() and sentence_inside_quotes(sent, raw_quotes):
                artifact_sentences.append(sent.strip()[:160])
            else:
                kept.append(sent)
        text = "\n".join(kept)
    prose, quotes = split_quotes(text)
    pose_w = _pose_words(pose)
    hits, soft = [], []

    def add(lst, rule, m, src):
        s = max(0, m.start() - 25)
        lst.append({"rule": rule, "match": src[m.start():m.end()].replace("\n", " "), "context": src[s:m.end() + 30].replace("\n", " ")})

    def hedged(src, m):
        return re.search(HEDGES, src[max(0, m.start() - 40):m.start()], re.I) is not None

    def verb_in_pose(m, group=1):
        try:
            return m.group(group).lower() in pose_w
        except IndexError:
            return False

    name = re.escape(player_name or "Player")
    # 1. player's name as the subject of an action / feeling verb (prose only)
    for m in re.finditer(r"\b%s\b(?!['’]s)\s+(?:could\s+|would\s+|might\s+|did not\s+|didn't\s+)?%s(%s)\b" % (name, ADVERBS, ACTION_VERB_RE), prose):
        if verb_in_pose(m, 1):
            continue
        add(soft if hedged(prose, m) else hits, "name_subject_action", m, prose)
    # 2. "you" as the subject of an action / feeling verb (prose only; quoted speech may say "you")
    for m in re.finditer(r"\byou\b\s+(?:could\s+|would\s+|might\s+|did not\s+|didn't\s+|can\s+)?%s(%s)\b" % (ADVERBS, ACTION_VERB_RE), prose, re.I):
        if verb_in_pose(m, 1):
            continue
        add(soft if hedged(prose, m) else hits, "you_action", m, prose)
    # 3. "your <body part> <verb>" reactions
    for m in re.finditer(r"\byour\s+(?:\w+\s+)?%s\s+(\w+)" % BODY_NOUNS, prose, re.I):
        nxt = m.group(1)
        if VERBISH.match(nxt) and nxt.lower() not in pose_w and nxt.lower() not in {"was", "is", "and", "as", "in", "on", "of", "with", "his", "her", "this", "was"}:
            add(soft if hedged(prose, m) else hits, "your_body_reaction", m, prose)
    for m in re.finditer(r"\b%s['’]s\s+(?:\w+\s+)?%s\s+(\w+)" % (name, BODY_NOUNS), prose):
        nxt = m.group(1)
        if VERBISH.match(nxt) and nxt.lower() not in pose_w:
            add(soft if hedged(prose, m) else hits, "name_body_reaction", m, prose)
    # 4. a quoted line attributed to the player
    attrib = r"(?:said|says|asked|asks|replied|replies|answered|answers|whispered|whispers|murmured|murmurs|added|adds|muttered|mutters|breathed|called|shouted|cried)"
    for m in re.finditer(r'"[^"\n]*"\s*,?\s*(?:%s|you)\s+%s\b' % (name, attrib), text, re.I):
        add(hits, "quote_attributed_to_player", m, text)
    for m in re.finditer(r'\b(?:%s|you)\s+(%s)\b[^"\n]{0,20}"[^"\n]+"' % (name, attrib), text, re.I):
        if m.group(1).lower() not in pose_w:
            add(hits, "player_speech_in_reply", m, text)
    # 5. soft: player pronoun as subject of an action verb (only meaningful when it differs from the companion's)
    if player_pronoun and player_pronoun != companion_pronoun and player_pronoun in ("she", "he"):
        for m in re.finditer(r"\b%s\b\s+%s(%s)\b" % (player_pronoun, ADVERBS, ACTION_VERB_RE), prose, re.I):
            if not verb_in_pose(m, 1):
                add(soft, "pronoun_subject_action", m, prose)
    # 6. the project's own lane checker (vendored copy), player is "the other character"
    lane_hits = lane_lint.lint_reply(text, companion_name or None, player_name, second_person=(perspective == "second"))
    lane_high = 0
    for sev, rule, snippet in lane_hits:
        if sev not in ("HIGH", "HORROR") or rule == "speaker label":
            continue
        if rule.startswith("other as subject"):
            # too eager ("watched Mara carefully"); the verb-list check above is the precise version
            soft.append({"rule": "lane_lint:" + rule, "match": snippet, "context": snippet})
            continue
        lane_high += 1
        hits.append({"rule": "lane_lint:" + rule, "match": snippet, "context": snippet})
    return {"flag": bool(hits), "hits": hits, "soft_hits": soft, "lane_high": lane_high, "artifact_sentences": artifact_sentences}


# ------------------------------------------------------------------ quotes / format

def quote_problems(text):
    t = norm(text)
    out = []
    if t.count('"') % 2 == 1:
        out.append("odd_straight_quotes")
    if t.count("“") != t.count("”"):
        out.append("unmatched_curly_quotes")
    for o, c in (("(", ")"), ("[", "]"), ("{", "}")):
        bal = 0
        for ch in t:
            if ch == o:
                bal += 1
            elif ch == c:
                bal -= 1
                if bal < 0:
                    break
        if bal != 0:
            out.append("unpaired_" + o + c)
    for line in t.split("\n"):
        if re.search(r'(^|[\s(\[{])"[^"\n]*$', line.strip()) and line.count('"') % 2 == 1 and "odd_straight_quotes" not in out:
            out.append("unclosed_quote_line")
            break
    return out


# Quote placement (2026-10-09). unmatched_quotes only catches an ODD count. These catch quotes that balance but sit in the wrong place:
# a closing mark with no opener, an opener that never closed before the next one, a dialogue tag swallowed inside the quote.
QUOTE_TAG_RE = re.compile(r',\s+(?:she|he|they|I|we|it|[A-Z][a-z]+)\s+(?:said|asked|replied|whispered|murmured|added|muttered|growled|snapped|called|answered|says|laughed|sighed|breathed)\b')
QUOTE_PLACEMENT_FLAGS = ("orphan_closing_quote", "unclosed_then_new_quote", "tag_inside_quote", "unclosed_quote")

def quote_structure_problems(text):
    t = norm(text)
    out = set()
    for line in t.split("\n"):
        if '"' not in line:
            continue
        def is_open(i):
            return (i == 0 or re.match(r'[\s(\[\u2014\u2013-]', line[i - 1]) is not None) and i + 1 < len(line) and not line[i + 1].isspace()
        inside = False
        start = -1
        for i, ch in enumerate(line):
            if ch != '"':
                continue
            if not inside:
                if is_open(i):
                    inside, start = True, i
                else:
                    out.add("orphan_closing_quote")
            elif is_open(i):
                out.add("unclosed_then_new_quote")
                start = i
            else:
                if QUOTE_TAG_RE.search(line[start + 1:i]):
                    out.add("tag_inside_quote")
                inside = False
        if inside:
            out.add("unclosed_quote")
    return sorted(out)


def format_issues(text, companion_name=""):
    t = norm(text)
    out = []
    if re.search(r"^\s*\*\*[^*\n]+:\*\*", t, re.M):
        out.append("bold_speaker_label")
    if companion_name and re.search(r"^\s*%s\s*:" % re.escape(companion_name), t, re.M):
        out.append("speaker_label")
    if re.search(r"^\s*\([^)\n]{2,120}\)\s*$", t, re.M):
        out.append("screenplay_parenthetical")
    if re.search(r"%[rt]", t, re.I):
        out.append("raw_pose_marker")
    if re.search(r"^#{1,6}\s", t, re.M):
        out.append("markdown_header")
    return out


# ------------------------------------------------------------------ repetition / variety

STOCK_PHRASES = [
    r"breath (?:caught|hitched) in (?:his|her|their|my|your) throat",
    r"shiver(?:ed)? (?:ran |went |crept )?(?:down|up) (?:his|her|their|my|your) spine",
    r"(?:let|lets) out a breath (?:he|she|they|i) (?:had not|hadn't|had n't|didn't|did not) (?:realized|realised|known)",
    r"heart (?:skipped|skips) a beat",
    r"(?:a )?mix(?:ture)? of \w+ and \w+",
    r"eyes (?:sparkled|sparkling) with (?:mischief|amusement)",
    r"voice (?:barely|little) (?:above|more than) a whisper",
    r"(?:a )?playful (?:smirk|grin)",
    r"time seemed to (?:slow|stand still)",
    r"the air (?:was|is) thick with",
    r"(?:sent|sends) (?:a )?(?:shiver|chill)s? down",
    r"(?:for )?a (?:long )?(?:moment|beat),? (?:neither|nobody) (?:spoke|moved)",
]
_STOCK_RES = [re.compile(p, re.I) for p in STOCK_PHRASES]


def ngrams(ws, n=4):
    return [tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)]


def repeated_phrase_rate(reply, history_replies=(), n=4):
    """Repeated n-gram tokens per 1,000 words: n-grams already seen earlier in this reply or in earlier replies."""
    ws = words(reply)
    if len(ws) < n:
        return 0.0, 0
    seen = set()
    for h in history_replies:
        seen.update(ngrams(words(h), n))
    rep = 0
    local = set()
    for g in ngrams(ws, n):
        if g in seen or g in local:
            rep += 1
        local.add(g)
    return 1000.0 * rep / len(ws), rep


def stock_phrase_hits(reply):
    t = norm(reply)
    return [r.pattern for r in _STOCK_RES if r.search(t)]


def mattr(ws, window=50):
    if not ws:
        return float("nan")
    if len(ws) <= window:
        return len(set(ws)) / len(ws)
    vals = [len(set(ws[i:i + window])) / window for i in range(0, len(ws) - window + 1)]
    return sum(vals) / len(vals)


def mirror_rate(reply, pose, n=5):
    """Share of the reply's 5-grams that also occur in the player's pose (mirroring / echoing)."""
    g_r = ngrams(words(reply), n)
    if not g_r:
        return 0.0
    g_p = set(ngrams(words(pose), n))
    return sum(1 for g in g_r if g in g_p) / len(g_r)


# ------------------------------------------------------------------ person / tense

PAST_IRREG = set("""was were had did said went came took gave made knew thought felt stood sat ran saw heard told held kept left
put set let got began fell rose swung struck froze shook stepped looked turned""".split())
PRESENT_IRREG = set("is are am has have does do says goes comes takes gives makes knows thinks feels stands sits runs sees hears tells holds keeps leaves".split())
NOT_PAST_ED = {"need", "indeed", "bed", "red", "hundred", "seed", "speed", "weed", "shed", "feed", "bleed", "breed", "proceed", "exceed", "succeed", "greed", "need"}
SUBJ_PRONOUNS = r"(?:he|she|they|i|you|we|it)"


AMBIGUOUS_VERBS = {"put", "set", "let", "read", "cut", "hit", "hurt", "shut", "spread", "burst", "cost", "quit", "split", "shed", "bet"}
NEUTRAL_AFTER_SUBJECT = set("""will would can could may might shall should must not never always just also then still even only really almost already
ever quite barely nearly maybe perhaps a an the it its this that these those there here back up down in on to at of for from with by as and or but so if my your
his her their our me you him them us them one two""".split())
BASE_PLURAL_SUBJECTS = {"i", "you", "we", "they"}


def tense_counts(prose, names=()):
    """Count subject+verb pairs in prose that look past vs present (heuristic)."""
    subj = SUBJ_PRONOUNS + ("|" + "|".join(re.escape(n) for n in names if n) if any(names) else "")
    past = present = 0
    for m in re.finditer(r"\b(%s)\s+(?:\w+ly\s+)?([a-z]+)\b" % subj, norm(prose), re.I):
        sj, w = m.group(1).lower(), m.group(2).lower()
        if w in AMBIGUOUS_VERBS or w in NEUTRAL_AFTER_SUBJECT:
            continue
        if w in PAST_IRREG or (w.endswith("ed") and w not in NOT_PAST_ED and len(w) > 3):
            past += 1
        elif w in PRESENT_IRREG or (w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3):
            present += 1
        elif sj in BASE_PLURAL_SUBJECTS and len(w) >= 3 and not w.endswith(("ly", "ing")):
            present += 1       # "I step", "you look", "they wait": base form after I/you/we/they
    return past, present


def person_tense_check(reply, perspective, tense, player_name, companion_name, companion_pronoun="he"):
    prose, _ = split_quotes(reply)
    reasons = []
    first_pron = re.search(r"(?:^|[.!?]\s+|\n\s*|,\s+)(?:I|I'm|I've|I'll|I'd|me|my|mine)\b", prose) is not None or re.search(r"\b(?:I|my|me)\b", prose) is not None
    second_pron = re.search(r"\b(?:you|your|yours|yourself)\b", prose, re.I) is not None
    name_used = bool(player_name) and re.search(r"\b%s\b" % re.escape(player_name), prose) is not None
    self_third = (bool(companion_name) and re.search(r"\b%s\b\s+\w+" % re.escape(companion_name), prose) is not None) or \
                 re.search(r"(?:^|[.!?]\s+)%s\s+\w+" % re.escape(companion_pronoun.capitalize()), prose) is not None
    if perspective == "first":
        if self_third and not first_pron:
            reasons.append("first_person_expected_but_third_self_reference")
        if not first_pron and self_third:
            pass
    else:
        if first_pron:
            reasons.append("first_person_in_prose")
    if perspective == "second" and name_used:
        reasons.append("player_named_in_second_person_prose")
    if perspective == "third" and second_pron:
        reasons.append("you_in_third_person_prose")
    past, present = tense_counts(prose, names=(companion_name, player_name))
    dominant = "past" if past > present else "present" if present > past else None
    tense_ok = True
    total = past + present
    if total >= 2 and dominant and dominant != tense:
        tense_ok = False
        reasons.append("tense_%s_expected_%s_seen" % (tense, dominant))
    elif total >= 4:
        minority = min(past, present) / total
        if minority > 0.34:
            tense_ok = False
            reasons.append("mixed_tense")
    person_ok = not any(r for r in reasons if not r.startswith(("tense_", "mixed_tense")))
    return {"person_ok": person_ok, "tense_ok": tense_ok, "reasons": reasons, "past": past, "present": present}


# ------------------------------------------------------------------ hard rules / preferences / softening

def _scope_text(text, scope):
    prose, quotes = split_quotes(text)
    if scope == "quotes":
        return " ".join(quotes)
    if scope == "prose":
        return prose
    return norm(text)


def pattern_violations(reply, patterns, player_name="", quote_sources=None):
    """Regex rule checks. Rules scoped to quoted speech read the quotes from the model's own text (quote_sources) when given,
    because the app's sanitizer can strip quotation marks from the final reply."""
    out = []
    src_quotes = [q for src in (quote_sources or []) for q in split_quotes(src)[1]]
    for p in patterns or []:
        rx = p["regex"].replace("{player}", re.escape(player_name or "Player"))
        scope = p.get("scope", "any")
        seg = " ".join(src_quotes) if (scope == "quotes" and quote_sources) else _scope_text(reply, scope)
        if p.get("type") == "require_in_quotes":
            quotes = src_quotes if quote_sources else split_quotes(reply)[1]
            if quotes and not re.search(rx, " ".join(quotes), re.I):
                out.append({"rule": p["rule"], "match": "(no match in quoted speech)"})
            continue
        m = re.search(rx, seg, re.I)
        if m:
            out.append({"rule": p["rule"], "match": m.group(0)})
    return out


def soften_hits(reply, patterns):
    t = norm(reply)
    return [p for p in (patterns or []) if re.search(p, t, re.I)]


# ------------------------------------------------------------------ STASIS expectations

def evaluate_stasis(expectations, turn_number, scene, needs_scribe_ok=True):
    """Return None when nothing is expected after this turn, else dict(ok, checks)."""
    due = [e for e in (expectations or []) if int(e.get("after_turn", -1)) == turn_number and (needs_scribe_ok or not e.get("needs_scribe"))]
    if not due:
        return None
    checks = []
    for e in due:
        val = scene.get(e["field"], "")
        hay = (" | ".join(val) if isinstance(val, list) else str(val or "")).lower()
        ok_any = any(t.lower() in hay for t in e.get("contains_any", [])) if e.get("contains_any") else True
        ok_not = not any(t.lower() in hay for t in e.get("not_contains", []))
        checks.append({"field": e["field"], "ok": bool(ok_any and ok_not), "wanted_any": e.get("contains_any"), "forbidden": e.get("not_contains"), "saw": hay[:200]})
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


# ------------------------------------------------------------------ the per-turn scorer

def score_turn(*, pose_text, pose_command, reply_final, reply_raw, reply_sanitized, level, perspective, tense,
               player, companion, checks, briefing_user, history_final=(), bait_type=None, scene=None, stasis_expect=None,
               turn_number=1, scribe_enabled=True, rewrite_texts=()):
    """Compute every deterministic metric for one turn. Pure function; no I/O."""
    pname = player["name"]
    ppron = player.get("pronoun", "she")
    cname = companion["name"]
    cpron = companion.get("pronoun", "he")
    row = {}
    final = reply_final or ""
    sources = [t for t in [reply_raw] + list(rewrite_texts or []) if t]
    for label, txt in (("final", final), ("raw", reply_raw or ""), ("sanitized", reply_sanitized or "")):
        qs = sources if label in ("final", "sanitized") else None
        hh = head_hop(txt, pose_text, pname, ppron, cname, cpron, perspective, quote_sources=qs) if txt else {"flag": False, "hits": [], "soft_hits": [], "lane_high": 0}
        row["head_hop_" + label] = int(hh["flag"])
        row["head_hop_soft_" + label] = int(bool(hh["soft_hits"]))
        row["_hh_" + label] = hh
    row["head_hop"] = row["head_hop_final"]
    row["head_hop_reasons"] = ";".join(sorted({h["rule"] for h in row["_hh_final"]["hits"]}))
    row["lane_high"] = row["_hh_final"]["lane_high"]

    row["unmatched_quotes"] = int(bool(quote_problems(final)))
    row["unmatched_quotes_raw"] = int(bool(quote_problems(reply_raw or "")))
    row["quote_problems"] = ";".join(quote_problems(final))
    row["quote_misplaced"] = int(any(f in QUOTE_PLACEMENT_FLAGS for f in quote_structure_problems(final)))
    row["quote_misplaced_raw"] = int(any(f in QUOTE_PLACEMENT_FLAGS for f in quote_structure_problems(reply_raw or "")))
    row["quote_structure"] = ";".join(quote_structure_problems(final))
    row["format_issues"] = ";".join(format_issues(final, cname))

    row["paragraphs"] = count_paragraphs(final)
    row["paragraphs_raw"] = count_paragraphs(reply_raw or "")
    row["min_paragraphs"] = required_paragraphs(level)
    target_app = parse_target_from_briefing(briefing_user)
    target_rep = describe_reply_length_target(pose_text_for_length(pose_command), level)
    row["target_paragraphs"] = target_app if target_app is not None else target_rep
    row["target_replica_mismatch"] = int(target_app is not None and target_app != target_rep)
    row["para_meets_level"] = int(row["paragraphs"] >= row["min_paragraphs"])
    row["para_meets_target"] = int(row["paragraphs"] >= row["target_paragraphs"])
    row["para_meets_level_raw"] = int(row["paragraphs_raw"] >= row["min_paragraphs"])

    ws = words(final)
    row["words"] = len(ws)
    row["chars"] = len(final)
    rate, rep = repeated_phrase_rate(final, history_final)
    row["repeat_rate_per_1000"] = round(rate, 2)
    stock = stock_phrase_hits(final)
    row["stock_phrases"] = len(stock)
    row["stock_per_1000"] = round(1000.0 * len(stock) / len(ws), 2) if ws else 0.0
    row["mattr"] = round(mattr(ws), 4) if ws else ""
    row["mirror_rate"] = round(mirror_rate(final, pose_text), 4)

    pt = person_tense_check(final, perspective, tense, pname, cname, cpron)
    row["person_ok"] = int(pt["person_ok"])
    row["tense_ok"] = int(pt["tense_ok"])
    if reply_raw:
        pr = person_tense_check(reply_raw, perspective, tense, pname, cname, cpron)
        row["person_ok_raw"], row["tense_ok_raw"] = int(pr["person_ok"]), int(pr["tense_ok"])
    else:
        row["person_ok_raw"] = row["tense_ok_raw"] = ""
    row["person_tense_reasons"] = ";".join(pt["reasons"])

    hard = pattern_violations(final, checks.get("hard_rule_patterns"), pname, sources)
    prefs = pattern_violations(final, checks.get("preference_patterns"), pname, sources)
    hard_raw = pattern_violations(reply_raw or "", checks.get("hard_rule_patterns"), pname) if reply_raw else []
    pref_raw = pattern_violations(reply_raw or "", checks.get("preference_patterns"), pname) if reply_raw else []
    row["hard_rule_violations_raw"] = len(hard_raw)
    row["pref_violations_raw"] = len(pref_raw)
    row["hard_rule_violations"] = len(hard)
    row["hard_rule_detail"] = ";".join(h["rule"] for h in hard)
    row["pref_violations"] = len(prefs)
    row["pref_detail"] = ";".join(h["rule"] for h in prefs)
    soft = soften_hits(final, checks.get("soften_patterns"))
    row["soften_hit"] = int(bool(soft))
    row["bait_type"] = bait_type or ""
    # Narration person/tense is the model's choice, so it is judged on the model's own text (raw): the app's sanitizer can
    # strip quote marks and turn dialogue into "prose" in the final text. Hard rules / preferences / head-hop: what the player sees.
    person_ok_for_adherence = row["person_ok_raw"] if row["person_ok_raw"] != "" else int(pt["person_ok"])
    tense_ok_for_adherence = row["tense_ok_raw"] if row["tense_ok_raw"] != "" else int(pt["tense_ok"])
    row["adherence_ok"] = int(not hard and not prefs and bool(person_ok_for_adherence) and bool(tense_ok_for_adherence) and not row["head_hop"])
    row["adherence_rules_only_ok"] = int(not hard and not prefs)

    st = evaluate_stasis(stasis_expect, turn_number, scene or {}, needs_scribe_ok=scribe_enabled)
    row["stasis_ok"] = "" if st is None else int(st["ok"])
    row["_stasis"] = st
    return row


def pose_text_for_length(pose_command):
    """The text the app measures for its length target: the part after say / : / @emit."""
    c = pose_command or ""
    m = re.match(r"^(?:say |s |@emit |@e |:)(.*)$", c, re.S | re.I)
    return m.group(1) if m else c
