#!/usr/bin/env python3
"""
lane_lint.py - checks roleplay text for lane violations ("head-hopping").

A reply must contain only the speaking character's own words, actions, thoughts and observations.
It must NOT write the other character's dialogue, actions, feelings, decisions, or completed outcomes.

Usage:
  # check a whole JSONL file (chat format: system / user / assistant messages)
  python3 scripts/lane_lint.py data/processed/my_horror_phase/train.jsonl

  # check one passage you just wrote, saying who is speaking and who the other character is
  python3 scripts/lane_lint.py --text passage.txt --self Uno --other Dvar
  # if the other character is addressed as "you" (second person), add --second-person

  --strict          also show LOW severity hits (hedged observations, possessive body parts)
  --show N          how many flagged examples to print (default 25)
  --json out.json   write the full report as JSON

Severity:
  HIGH  the other character is written as a subject doing/saying/feeling something, or a speaker label appears,
        or (second person) "you" does a completed reaction.  These should be rewritten.
  LOW   hedged reads ("seemed to", "looked like"), or the other's body part named as a possessive.  Review only.
  REVIEW  a he/she/they does something; usually the other character, sometimes an NPC.
  HORROR  violence/damage described as already LANDING on the other character's body.  In horror this is the
        most common way a good scene accidentally head-hops: end on the attempt, the swing, the reach, and
        leave the landing to the other writer.  Damage to yourself, monsters, NPCs, objects and the room is free.
"""
import argparse, json, re, sys
from collections import Counter

HEDGES = r"(?:seemed|seems|looked|looks|appeared|appears|as if|as though|might|maybe|perhaps|could be|probably|I think|I guess|I can tell)"
PREP = {"and","or","but","to","with","on","in","at","for","from","if","as","the","a","an","that","than","then","who","whom","whose","before","after","while","when","where","would","could","might","may","should","must","can","will","shall","did","does","do","had","has","have","not","no","never","only","still","just","also","even","too"}
SPEECH_OR_FEELING = r"(?:said|says|asked|asks|replied|replies|answered|answers|whispered|murmured|muttered|shouted|screamed|cried|laughed|smiled|nodded|nods|shook|shrugged|sighed|gasped|flinched|shivered|trembled|relaxed|tensed|stiffened|froze|stepped|moved|reached|grabbed|pulled|pushed|turned|looked|stared|glared|felt|feels|thought|thinks|wanted|wants|decided|decides|agreed|agrees|accepted|accepts|refused|refuses|obeyed|obeys|surrendered|collapsed|fell|ran|walked|stood|sat|knelt)"
OUTCOME_YOU = r"(?:flinch|flinches|flinched|gasp|gasps|gasped|shiver|shivers|shivered|tremble|trembles|trembled|tense|tenses|tensed|relax|relaxes|relaxed|freeze|freezes|froze|nod|nods|nodded|agree|agrees|agreed|accept|accepts|accepted|obey|obeys|obeyed|scream|screams|screamed|cry|cries|cried|sob|sobs|sobbed|bleed|bleeds|bled|fall|falls|fell|collapse|collapses|collapsed|choke|chokes|choked|gag|gags|gagged|step closer|stepped closer|back away|backed away|feel|feels|felt|think|thinks|thought|want|wants|wanted|decide|decides|decided|realize|realizes|realized|know|knows|knew|surrender|surrenders|surrendered|give in|gave in|moan|moans|moaned|whimper|whimpers|whimpered|blush|blushes|blushed|swallow|swallows|swallowed|smile|smiles|smiled|laugh|laughs|laughed|say|says|said|whisper|whispers|whispered)"
BODY = r"(?:skin|flesh|bone|bones|blood|throat|ribs|ribcage|face|eyes|eye|jaw|teeth|tongue|skull|spine|neck|arm|arms|hand|hands|fingers|leg|legs|chest|stomach|gut|guts|belly|ear|ears|lips|mouth|shoulder|wrist|ankle|knee|knees|hair|scalp|heart|lungs)"
BODY_VERB = r"(?:tears?|tore|torn|rips?|ripped|splits?|split|snaps?|snapped|breaks?|broke|broken|bursts?|burst|bleeds?|bled|burns?|burned|burnt|cracks?|cracked|gives way|gave way|parts|parted|opens|opened|peels?|peeled|melts?|melted|pops?|popped|shatters?|shattered|caves in|caved in|bruises?|bruised|gushes|gushed|spills?|spilled)"

def parse_names(user_text):
    names = []
    for line in user_text.splitlines():
        m = re.match(r"^([A-Z][A-Za-z'\- ]{0,30}):\s", line)
        if m and m.group(1) not in names and m.group(1) not in {"Scene title","Room","Premise","Notes","Conversation so far"}:
            names.append(m.group(1))
    return names

def infer_roles(user_text):
    names = parse_names(user_text)
    if not names:
        return None, None
    last = None
    for line in user_text.splitlines():
        m = re.match(r"^([A-Z][A-Za-z'\- ]{0,30}):\s", line)
        if m and m.group(1) in names:
            last = m.group(1)
    other = last or names[-1]
    me = next((n for n in names if n != other), None)
    return me, other

def lint_reply(reply, me, other, second_person=False):
    hits = []
    def add(sev, rule, snippet):
        hits.append((sev, rule, snippet.strip()[:160]))
    text = reply.replace("’", "'")
    # speaker labels
    for line in text.splitlines():
        if re.match(r"^\s*(?:\*\*)?[A-Z][A-Za-z'\- ]{0,24}(?:\*\*)?\s*:\s", line) and not line.strip().startswith(("Note:", "Scene")):
            add("HIGH", "speaker label", line)
    if other:
        o = re.escape(other)
        # other character as grammatical subject
        for m in re.finditer(rf"\b{o}\b(?!['’]s)\s+([A-Za-z']+)", text):
            nxt = m.group(1).lower()
            if nxt in PREP:
                continue
            rest = text[m.end():m.end() + 40]
            if nxt in {"looks", "seems", "appears", "looked", "seemed", "appeared"} and re.match(rf"\s*(?:like|as if|as though|to|\w+)", rest):
                add("LOW", "other as subject (hedged read)", text[max(0, m.start() - 20):m.end() + 40].replace("\n", " "))
                continue
            start = max(0, m.start() - 40)
            ctx = text[start:m.end() + 40]
            before = text[max(0, m.start() - 36):m.start()]
            hedged = re.search(HEDGES, before, re.I) is not None
            add("LOW" if hedged else "HIGH", "other as subject" + (" (hedged read)" if hedged else ""), ctx.replace("\n", " "))
        # other's possessive body part doing something (damage/reaction landing)
        for m in re.finditer(rf"\b{o}['’]s\s+{BODY}\s+{BODY_VERB}", text, re.I):
            add("HORROR", "damage lands on other's body", text[max(0, m.start() - 30):m.end() + 40].replace("\n", " "))
        for m in re.finditer(rf"\b{o}['’]s\s+{BODY}\b", text, re.I):
            add("LOW", "other's body part named", text[max(0, m.start() - 20):m.end() + 30].replace("\n", " "))
    # he/she/they/his/her/their usually means the OTHER character (or an NPC).
    outside_quotes = re.sub(r'"[^"]*"', " ", text)
    first_person = re.search(r"\b(?:I|I'm|I've|I'll|my|me|myself)\b", outside_quotes) is not None
    if first_person:
        for m in re.finditer(r'"[^"]+",?\s+(?:he|she|they)\s+' + SPEECH_OR_FEELING + r'\b', text, re.I):
            add("HIGH", "pronoun speaks (other's dialogue?)", text[max(0, m.start()):m.end() + 20].replace("\n", " "))
        for m in re.finditer(r"\b(?:he|she|they)\s+(?:" + OUTCOME_YOU + r")\b", outside_quotes, re.I):
            add("REVIEW", "pronoun completes a reaction (other or NPC?)", outside_quotes[max(0, m.start() - 30):m.end() + 40].replace("\n", " "))
        for m in re.finditer(r"\b(?:his|her|their)\s+" + BODY + r"\s+" + BODY_VERB, outside_quotes, re.I):
            add("HORROR", "damage lands on his/her body (other or NPC?)", outside_quotes[max(0, m.start() - 30):m.end() + 40].replace("\n", " "))
        for m in re.finditer(r"\b(?:opens|split|splits|tears|tore|rips|ripped|slices|sliced|carves|carved|pierces|pierced|crushes|crushed)\s+(?:his|her|their)\s+" + BODY, outside_quotes, re.I):
            add("HORROR", "damage lands on his/her body (other or NPC?)", outside_quotes[max(0, m.start() - 30):m.end() + 40].replace("\n", " "))
    if second_person:
        for m in re.finditer(rf"\byou\s+(?:{OUTCOME_YOU})\b", text, re.I):
            # quoted speech aimed at the other ("you feel...") inside quotes is fine; check if inside a quote span
            pre = text[:m.start()]
            if pre.count('"') % 2 == 1:
                continue
            add("HIGH", "you complete a reaction", text[max(0, m.start() - 30):m.end() + 40].replace("\n", " "))
        for m in re.finditer(rf"\byour\s+{BODY}\s+{BODY_VERB}", text, re.I):
            pre = text[:m.start()]
            if pre.count('"') % 2 == 1:
                continue
            add("HORROR", "damage lands on your body", text[max(0, m.start() - 30):m.end() + 40].replace("\n", " "))
    return hits

def main():
    ap = argparse.ArgumentParser(description="Lane-discipline linter for roleplay text")
    ap.add_argument("path", nargs="?", help="JSONL file in MLX chat format")
    ap.add_argument("--text", help="plain text file with one reply to check")
    ap.add_argument("--self", dest="me", help="speaking character name (with --text)")
    ap.add_argument("--other", help="other character name (with --text)")
    ap.add_argument("--second-person", action="store_true", help="the other character is addressed as 'you'")
    ap.add_argument("--strict", action="store_true", help="show LOW hits too")
    ap.add_argument("--show", type=int, default=25)
    ap.add_argument("--json")
    a = ap.parse_args()

    examples = []
    if a.text:
        examples.append({"idx": 0, "me": a.me, "other": a.other, "reply": open(a.text, encoding="utf-8").read()})
    elif a.path:
        for i, line in enumerate(open(a.path, encoding="utf-8")):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("messages", [])
            user = next((m["content"] for m in msgs if m["role"] == "user"), "")
            reply = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), "")
            me, other = infer_roles(user)
            examples.append({"idx": i, "me": me, "other": other, "reply": reply})
    else:
        ap.error("give a JSONL path or --text")

    report = []
    sev_counts = Counter(); rule_counts = Counter(); flagged = 0
    for ex in examples:
        hits = lint_reply(ex["reply"], ex["me"], ex["other"], second_person=a.second_person)
        shown = [h for h in hits if a.strict or h[0] != "LOW"]
        if shown:
            flagged += 1
        for sev, rule, snip in hits:
            sev_counts[sev] += 1; rule_counts[rule.split(" (")[0]] += 1
        report.append({"idx": ex["idx"], "me": ex["me"], "other": ex["other"], "hits": hits})

    print(f"Checked {len(examples)} example(s). Flagged (HIGH/HORROR{'/LOW' if a.strict else ''}): {flagged}  ({100*flagged/max(1,len(examples)):.1f}%)")
    print("By severity:", dict(sev_counts))
    print("By rule:", dict(rule_counts))
    shown_n = 0
    for r in report:
        hits = [h for h in r["hits"] if a.strict or h[0] != "LOW"]
        if not hits:
            continue
        shown_n += 1
        if shown_n > a.show:
            print(f"... {flagged - a.show} more flagged example(s); raise --show to see them")
            break
        print(f"\n[#{r['idx']}]  speaker={r['me']}  other={r['other']}")
        for sev, rule, snip in hits[:4]:
            print(f"   {sev:6s} {rule}: ...{snip}...")
    if a.json:
        json.dump(report, open(a.json, "w"), indent=2)
    sys.exit(1 if any(h[0] in ("HIGH", "HORROR") for r in report for h in r["hits"]) else 0)

if __name__ == "__main__":
    main()
