"""Deterministic pose composer for the stub player bot.

Scenario files hold templates written once; this module renders them into first / second /
third person and past / present tense, and sizes the pose to the Roleplay Style Level
(the app scales the reply target to the player's pose length):

    Level 1  one paragraph, under 350 characters (app: "player's input is brief")
    Level 2  two to three paragraphs, about 900+ characters (app: "long and developed", > 900)
    Level 3  three or more paragraphs, 1,800+ characters (app: "very long and detailed", > 1800)

Template tokens:
    {S}/{s}    subject at sentence start / mid-sentence (lowercase)
    {R}        reflexive (myself / yourself / herself)
    {S}        subject at sentence start (I / You / She-He-They, or dropped on a leading ":" pose)
    {v:verb}   verb conjugated for person and tense (player as subject)
    {t:past|present}   literal word chosen by tense (verbs whose subject is not the player)
    {P}        possessive (my / your / her)
    {O}        object (me / you / her)
    {C}        companion name
Paragraphs in an @emit are joined with "%r%r", the way a MUSH-style player types them.
"""
import random
import re

IRREGULAR_PAST = {
    "be": "was", "have": "had", "do": "did", "go": "went", "say": "said", "take": "took", "sit": "sat",
    "stand": "stood", "run": "ran", "hold": "held", "feel": "felt", "think": "thought", "see": "saw",
    "hear": "heard", "find": "found", "keep": "kept", "leave": "left", "put": "put", "set": "set",
    "let": "let", "get": "got", "give": "gave", "make": "made", "know": "knew", "come": "came",
    "tell": "told", "draw": "drew", "throw": "threw", "catch": "caught", "fall": "fell", "rise": "rose",
    "swing": "swung", "strike": "struck", "hide": "hid", "lead": "led", "wear": "wore", "drink": "drank",
    "choose": "chose", "hang": "hung", "shake": "shook", "slide": "slid", "spin": "spun", "bite": "bit",
    "break": "broke", "speak": "spoke", "wake": "woke", "begin": "began", "bend": "bent", "lay": "laid",
    "lie": "lay", "meet": "met", "pay": "paid", "read": "read", "ride": "rode", "ring": "rang",
    "send": "sent", "shoot": "shot", "sing": "sang", "sleep": "slept", "steal": "stole", "stick": "stuck",
    "sweep": "swept", "teach": "taught", "understand": "understood", "win": "won", "bring": "brought",
    "buy": "bought", "build": "built", "cut": "cut", "deal": "dealt", "dig": "dug", "fight": "fought",
    "flee": "fled", "forget": "forgot", "grow": "grew", "burst": "burst", "hit": "hit", "hurt": "hurt",
    "lose": "lost", "mean": "meant", "seek": "sought", "shut": "shut", "spread": "spread", "tear": "tore",
    "undo": "undid", "freeze": "froze", "swear": "swore", "sink": "sank", "creep": "crept", "wind": "wound", "spit": "spat", "wring": "wrung", "forgive": "forgave", "bind": "bound", "swim": "swam", "stride": "strode", "flee_": "fled", "light": "lit", "shine": "shone", "get_": "got", "withdraw": "withdrew", "arise": "arose", "bleed": "bled", "blow": "blew",
}
DOUBLE = {"step", "stop", "grab", "drop", "nod", "slip", "skip", "hop", "tap", "clap", "plan", "trap", "rub",
          "hug", "pat", "wrap", "strip", "whip", "shrug", "dodge_no", "drag", "flip", "jab", "snap", "trim",
          "slap", "swap", "pin", "tug", "jog", "scrub", "grip", "fit", "scan", "hum", "pop", "jam", "stir", "clip", "skim", "stab", "mop", "chop", "shop", "plot", "knit", "tip"}

GENERIC_FILLER = [
    "{S} {v:take} a moment before moving again, letting the details of the place sink in.",
    "{S} {v:weigh} the next few seconds carefully, aware of how small choices {t:shaped|shape} a scene.",
    "{S} {v:notice} a small, stubborn detail that {t:kept|keeps} pulling {P} attention back.",
    "{S} {v:feel} {P} pulse settle into something steadier and more patient.",
    "{S} {v:let} {P} gaze wander, then {v:draw} it back to what mattered.",
    "{S} {v:set} {P} jaw and {v:remind} {R} to keep {P} movements deliberate and unhurried.",
    "{S} {v:turn} the situation over in {P} mind, looking for the angle {s} {t:had missed|have missed}.",
    "{S} {v:breathe} out slowly and {v:let} {P} shoulders drop a fraction.",
    "{S} {v:keep} {P} voice level, knowing how easily tone {t:carried|carries} in a room like this.",
    "{S} {v:remember} an old lesson about patience and {v:decide} to trust it a little longer.",
    "{S} {v:lift} {P} chin and {v:meet} whatever {t:came|comes} next with open eyes.",
    "{S} {v:catch} the faint sound of {P} own heartbeat under the other noises.",
    "{S} {v:shift} {P} stance, ready to move either way if the moment {t:asked|asks} for it.",
    "{S} {v:rub} the back of {P} neck and {v:push} a stray thought out of the way.",
    "{S} {v:take} in the whole scene at once, the way {s} {t:had learned|have learned} to years ago.",
    "{S} {v:count} the seconds in {P} head and {v:find} the number oddly calming.",
    "{S} {v:feel} a small, private flicker of determination warm {P} chest.",
    "{S} {v:smooth} {P} expression and {v:wait} for the right opening.",
    "{S} {v:listen} past the obvious noises for the quieter ones underneath.",
    "{S} {v:tilt} {P} head slightly, as if trying to hear what the room {t:was|is} not saying.",
    "{S} {v:grip} the nearest solid thing for a second and {v:let} go again.",
    "{S} {v:think} about how strange it {t:was|is} to be so alert and so calm together.",
]


def conj(verb, person, tense):
    verb = verb.strip()
    if verb == "be":
        if tense == "past":
            return "were" if person == "second" else "was"
        return {"first": "am", "second": "are", "third": "is"}[person]
    if verb == "have":
        if tense == "past":
            return "had"
        return "has" if person == "third" else "have"
    if tense == "past":
        if verb in IRREGULAR_PAST:
            return IRREGULAR_PAST[verb]
        if verb.endswith("e"):
            return verb + "d"
        if re.search(r"[^aeiou]y$", verb):
            return verb[:-1] + "ied"
        if verb in DOUBLE:
            return verb + verb[-1] + "ed"
        return verb + "ed"
    if person != "third":
        return verb
    if verb == "do":
        return "does"
    if verb == "go":
        return "goes"
    if re.search(r"(s|sh|ch|x|z)$", verb):
        return verb + "es"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ies"
    return verb + "s"


PRON = {
    "first": {"S": "I", "P": "my", "O": "me", "R": "myself"},
    "second": {"S": "You", "P": "your", "O": "you", "R": "yourself"},
}
THIRD = {"she": {"S": "She", "P": "her", "O": "her", "R": "herself"}, "he": {"S": "He", "P": "his", "O": "him", "R": "himself"},
         "they": {"S": "They", "P": "their", "O": "them", "R": "themselves"}}


def render(template, person, tense, companion="", pronoun="she", drop_subject=False, name=""):
    p = PRON.get(person) or THIRD.get(pronoun, THIRD["she"])
    out = template
    out = re.sub(r"\{v:([a-z']+)\}", lambda m: conj(m.group(1), person, tense), out)
    out = re.sub(r"\{t:([^|}]+)\|([^}]+)\}", lambda m: m.group(1) if tense == "past" else m.group(2), out)
    if drop_subject and out.startswith("{S} "):
        out = out[4:]
    lower_s = p["S"] if p["S"] == "I" else p["S"].lower()
    out = (out.replace("{S}", p["S"]).replace("{s}", lower_s).replace("{P}", p["P"]).replace("{O}", p["O"])
              .replace("{R}", p["R"]).replace("{C}", companion))
    return out


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


def compose_pose(turn, scenario, perspective, level, tense, companion, pronoun, seed=0, bait=None):
    """Return dict(kind, command, text, chars, paragraphs)."""
    rng = random.Random("%s|%s|%s|%s|%s" % (seed, scenario["id"], turn["id"], perspective, level))
    level = int(level)
    R = lambda t, drop=False: render(t, perspective, tense, companion, pronoun, drop_subject=drop)
    acts = list(turn.get("act", []))
    speech = turn.get("speech")
    sense = turn.get("sense")
    think = turn.get("think")
    gap = turn.get("gap") if (bait is None or bait) else None
    pool = list(scenario.get("filler", [])) + GENERIC_FILLER
    rng.shuffle(pool)
    filler = list(turn.get("expand", [])) + pool   # beat-specific sentences first, then scenario filler, then generic filler

    def speech_sentence():
        return R('{S} {v:say}, "%s"' % speech)

    # Level 1: one short paragraph under 350 characters.
    if level == 1:
        if turn.get("mode") == "say" and speech:
            text = speech
            return {"kind": "say", "command": "say " + text, "text": text, "chars": len(text), "paragraphs": 1}
        sents = []
        if acts:
            sents.append(R(acts[0], drop=(perspective == "third")))
        if gap:
            sents.append(R(gap))
        elif speech and len(" ".join(sents)) < 150:
            sents.append(speech_sentence())
        text = " ".join(sents)
        while len(text) > 330 and len(sents) > 1:
            sents.pop()
            text = " ".join(sents)
        if perspective == "third":
            return {"kind": "colon", "command": ":" + text, "text": text, "chars": len(text), "paragraphs": 1}
        return {"kind": "emit", "command": "@emit " + text, "text": text, "chars": len(text), "paragraphs": 1}

    # Levels 2 and 3: @emit with %r%r paragraph breaks, grown to the length band.
    lo, hi = (930, 1150) if level == 2 else (1850, 2300)
    body = [R(a, drop=(perspective == "third" and i == 0)) for i, a in enumerate(acts)]
    for x in (sense, think):
        if x:
            body.append(R(x))
    tail = []
    if speech:
        tail.append(speech_sentence())
    if gap:
        tail.append(R(gap))
    extras = [R(f) for f in filler]
    chosen = []
    k = 0

    def build(extra):
        sents = body[:2] + extra + body[2:]  # scene detail between the opening action and the sense/thought beats
        n_par = 3 if level == 2 else (5 if len(" ".join(sents + tail)) > 2100 else 4)
        per = max(1, -(-len(sents) // max(1, n_par - 1)))
        paras = [sents[i:i + per] for i in range(0, len(sents), per)] or [[]]
        paras.append(list(tail)) if tail else None
        paras = [p for p in paras if p]
        return paras

    while k < len(extras):
        text = "%r%r".join(" ".join(p) for p in build(chosen))
        if len(text) >= lo:
            break
        chosen.append(extras[k])
        k += 1
    paras = build(chosen)
    text = "%r%r".join(" ".join(p) for p in paras)
    while len(text) > hi and chosen:
        chosen.pop()
        paras = build(chosen)
        text = "%r%r".join(" ".join(p) for p in paras)
    n_par = len(paras)
    return {"kind": "emit", "command": "@emit " + text, "text": text, "chars": len(text), "paragraphs": n_par}
