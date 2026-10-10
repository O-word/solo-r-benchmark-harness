import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness import scoring as S  # noqa: E402

P = dict(player_name="Mara", player_pronoun="she", companion_name="Brick", companion_pronoun="he")


def hh(reply, pose="steps onto the train.", perspective="third"):
    return S.head_hop(reply, pose, P["player_name"], P["player_pronoun"], P["companion_name"], P["companion_pronoun"], perspective)


class HeadHop(unittest.TestCase):
    def test_clean_reply(self):
        r = hh('Brick leans against the doorframe. "Don\'t expect a medal," he says.')
        self.assertFalse(r["flag"], r["hits"])

    def test_name_subject_action(self):
        self.assertTrue(hh("Mara flinched and stepped back.")["flag"])
        self.assertTrue(hh("Brick grins. Mara felt a rush of fear.")["flag"])

    def test_you_action_in_prose(self):
        r = hh("Brick watches as you flinch and take a step back.", perspective="second")
        self.assertTrue(r["flag"])
        self.assertIn("you_action", {h["rule"] for h in r["hits"]})

    def test_you_in_quotes_is_fine(self):
        self.assertFalse(hh('Brick glares. "You want something from me, say it plainly."', perspective="second")["flag"])

    def test_you_as_object_is_fine(self):
        self.assertFalse(hh("Brick looks you over and snorts.", perspective="second")["flag"])

    def test_your_body_reaction(self):
        r = hh("Brick steps close. Your heart pounded against your ribs.", perspective="second")
        self.assertTrue(r["flag"])

    def test_player_speech_attributed(self):
        self.assertTrue(hh('"Okay," Mara said quietly. Brick nods.')["flag"])
        self.assertTrue(hh('"Okay," you say quietly.', perspective="second")["flag"])

    def test_verb_already_in_pose_is_not_flagged(self):
        r = S.head_hop("Brick eyes Mara as she steps onto the train.", "Mara steps onto the train.", "Mara", "she", "Brick", "he", "third")
        self.assertFalse(r["flag"], r["hits"])

    def test_hedged_read_is_soft(self):
        r = hh("Brick squints. Mara seemed to flinch, or maybe it was the light.")
        self.assertFalse(r["flag"])
        r = hh("Brick squints. As if Mara flinched, though it might have been the light.")
        self.assertFalse(r["flag"])
        self.assertTrue(r["soft_hits"])

    def test_quote_artifact_is_suppressed(self):
        # the app's sanitizer stripped the quote marks; the raw reply shows it was dialogue
        raw = 'Brick shrugs. "You want something from me, say it plainly." He waits.'
        final = "Brick shrugs. You want something from me, say it plainly. He waits."
        self.assertTrue(S.head_hop(final, "x", "Mara", "she", "Brick", "he", "second")["flag"])
        r = S.head_hop(final, "x", "Mara", "she", "Brick", "he", "second", quote_sources=[raw])
        self.assertFalse(r["flag"])
        self.assertTrue(r["artifact_sentences"])
        # a genuine head-hop is still caught even when quote sources are supplied
        r = S.head_hop("Brick shrugs. You flinch.", "x", "Mara", "she", "Brick", "he", "second", quote_sources=[raw])
        self.assertTrue(r["flag"])

    def test_pronoun_only_is_soft(self):
        r = hh("Brick grunts. She flinched at the sound.")
        self.assertFalse(r["flag"])
        self.assertTrue(any(h["rule"] == "pronoun_subject_action" for h in r["soft_hits"]))

    def test_same_pronoun_not_soft(self):
        r = S.head_hop("Sable nods. She glanced at the chart.", "x", "Mara", "she", "Sable", "she", "third")
        self.assertFalse(r["soft_hits"])

    def test_lane_lint_second_person_reaction(self):
        self.assertTrue(hh("Brick shoves the door. You gasp.", perspective="second")["flag"])


class Quotes(unittest.TestCase):
    def test_balanced(self):
        self.assertEqual(S.quote_problems('He said "go" and left.'), [])

    def test_odd_straight(self):
        self.assertIn("odd_straight_quotes", S.quote_problems('He said "go and left.'))

    def test_unpaired_bracket(self):
        self.assertIn("unpaired_()", S.quote_problems("He left (quietly."))

    def test_curly(self):
        self.assertIn("unmatched_curly_quotes", S.quote_problems("He said “go and left."))

    def test_split_quotes(self):
        prose, quotes = S.split_quotes('A "b" c "d" e')
        self.assertEqual(quotes, ["b", "d"])
        self.assertNotIn("b", prose.replace("c", "").replace("e", "").replace("A", "").strip().replace(" ", ""))


class Paragraphs(unittest.TestCase):
    def test_count_matches_app_rule(self):
        self.assertEqual(S.count_paragraphs("a\n\nb\n \nc"), 3)
        self.assertEqual(S.count_paragraphs("a\nb"), 1)  # single newline is NOT a paragraph break
        self.assertEqual(S.count_paragraphs(""), 0)

    def test_required(self):
        self.assertEqual([S.required_paragraphs(x) for x in (1, 2, 3, "2")], [1, 2, 3, 2])

    def test_length_target_replica(self):
        self.assertEqual(S.describe_reply_length_target("x" * 100, 1), 1)
        self.assertEqual(S.describe_reply_length_target("x" * 400, 2), 2)
        self.assertEqual(S.describe_reply_length_target("x" * 950, 1), 2)
        self.assertEqual(S.describe_reply_length_target("x" * 950, 3), 3)
        self.assertEqual(S.describe_reply_length_target("x" * 1900, 1), 3)
        self.assertEqual(S.describe_reply_length_target("a\nb\nc\nd\ne\nf", 1), 2)

    def test_parse_target(self):
        u = "Reply length target: The player's pose is long and developed. Reply with a substantial answer of at least 2 paragraphs.\nx"
        self.assertEqual(S.parse_target_from_briefing(u), 2)
        self.assertIsNone(S.parse_target_from_briefing("nothing"))

    def test_pose_text_for_length(self):
        self.assertEqual(S.pose_text_for_length("@emit hello"), "hello")
        self.assertEqual(S.pose_text_for_length(":waves"), "waves")
        self.assertEqual(S.pose_text_for_length("say hi"), "hi")


class Repetition(unittest.TestCase):
    def test_repeat_rate(self):
        rate, n = S.repeated_phrase_rate("the quick brown fox jumps over the lazy dog today", ["the quick brown fox jumps over"])
        self.assertGreater(rate, 0)
        self.assertGreaterEqual(n, 3)
        self.assertEqual(S.repeated_phrase_rate("one two three four five six", [])[1], 0)

    def test_within_reply(self):
        _, n = S.repeated_phrase_rate("a b c d e a b c d e", [])
        self.assertGreater(n, 0)

    def test_stock_phrase(self):
        self.assertTrue(S.stock_phrase_hits("Her breath caught in her throat as a shiver ran down her spine."))
        self.assertFalse(S.stock_phrase_hits("The kettle ticked."))

    def test_mattr(self):
        self.assertAlmostEqual(S.mattr(["a"] * 10), 0.1)
        self.assertEqual(S.mattr([]) != S.mattr([]), True)  # nan


class PersonTense(unittest.TestCase):
    def test_first_person_ok(self):
        r = S.person_tense_check('I lean on the wall and watch Mara. "No," I say.', "first", "present", "Mara", "Brick")
        self.assertTrue(r["person_ok"], r)

    def test_first_person_in_third_mode(self):
        r = S.person_tense_check("I lean on the wall.", "third", "present", "Mara", "Brick")
        self.assertFalse(r["person_ok"])

    def test_you_in_third_mode(self):
        r = S.person_tense_check("Brick glares at you.", "third", "present", "Mara", "Brick")
        self.assertFalse(r["person_ok"])

    def test_name_in_second_mode(self):
        r = S.person_tense_check("Brick glares at Mara.", "second", "present", "Mara", "Brick")
        self.assertFalse(r["person_ok"])

    def test_tense_mismatch(self):
        r = S.person_tense_check("Brick leaned back. He shrugged. He looked away.", "third", "present", "Mara", "Brick")
        self.assertFalse(r["tense_ok"])
        self.assertTrue(S.person_tense_check("Brick leans back. He shrugs. He looks away.", "third", "present", "Mara", "Brick")["tense_ok"])

    def test_dialogue_does_not_count_for_tense(self):
        self.assertTrue(S.person_tense_check('Brick leans back. "I was there," he says. He shrugs.', "third", "present", "Mara", "Brick")["tense_ok"])


class Rules(unittest.TestCase):
    PATS = [{"rule": "Never apologize", "regex": r"\b(i'?m sorry|my apologies)\b", "scope": "any"},
            {"rule": "no name in speech", "regex": r"\b{player}\b", "scope": "quotes"},
            {"rule": "say Captain", "type": "require_in_quotes", "regex": r"\bCaptain\b"}]

    def test_apology(self):
        self.assertEqual([v["rule"] for v in S.pattern_violations("Brick sighs. I'm sorry.", self.PATS, "Mara")], ["Never apologize"])

    def test_name_scope_quotes_only(self):
        self.assertEqual(S.pattern_violations('Brick eyes Mara. "Move."', self.PATS[1:2], "Mara"), [])
        self.assertTrue(S.pattern_violations('Brick eyes her. "Move, Mara."', self.PATS[1:2], "Mara"))

    def test_require_in_quotes(self):
        self.assertTrue(S.pattern_violations('Sable nods. "Fuel is fine."', self.PATS[2:], "Mara"))
        self.assertFalse(S.pattern_violations('Sable nods. "Fuel is fine, Captain."', self.PATS[2:], "Mara"))
        self.assertFalse(S.pattern_violations("Sable nods silently.", self.PATS[2:], "Mara"))  # no speech, nothing to require

    def test_quote_scoped_rule_reads_model_text(self):
        pats = [{"rule": "no name in speech", "regex": r"\b{player}\b", "scope": "quotes"}]
        raw = 'Brick eyes her. "Move, Mara."'
        final = "Brick eyes her. Move, Mara."          # sanitizer stripped the quotes
        self.assertEqual(S.pattern_violations(final, pats, "Mara"), [])          # without sources the speech is invisible
        self.assertTrue(S.pattern_violations(final, pats, "Mara", [raw]))        # with the model's own text it is caught

    def test_soften(self):
        self.assertTrue(S.soften_hits("Fine. Friends, then.", [r"\bfriends\b"]))
        self.assertFalse(S.soften_hits("Get lost.", [r"\bfriends\b"]))


class Stasis(unittest.TestCase):
    def test_none_due(self):
        self.assertIsNone(S.evaluate_stasis([{"after_turn": 3, "field": "x", "contains_any": ["a"]}], 1, {}))

    def test_pass_and_fail(self):
        e = [{"after_turn": 2, "field": "playerWardrobeCue", "contains_any": ["uniform"], "not_contains": ["jacket"]}]
        self.assertTrue(S.evaluate_stasis(e, 2, {"playerWardrobeCue": "dress uniform, boots"})["ok"])
        self.assertFalse(S.evaluate_stasis(e, 2, {"playerWardrobeCue": "uniform, flight jacket"})["ok"])

    def test_scribe_dependent_skipped_without_scribe(self):
        e = [{"after_turn": 1, "field": "scribeFacts", "contains_any": ["promise"], "needs_scribe": True}]
        self.assertIsNone(S.evaluate_stasis(e, 1, {}, needs_scribe_ok=False))
        self.assertFalse(S.evaluate_stasis(e, 1, {"scribeFacts": []}, needs_scribe_ok=True)["ok"])


class WholeTurn(unittest.TestCase):
    def test_score_turn_shape(self):
        row = S.score_turn(pose_text="steps in.", pose_command=":steps in.", reply_final='Brick snorts. "Move."\n\nHe looks away.', reply_raw='Brick snorts. "Move."', reply_sanitized='Brick snorts. "Move."',
                           level=2, perspective="third", tense="present", player={"name": "Mara", "pronoun": "she"}, companion={"name": "Brick", "pronoun": "he"},
                           checks={"hard_rule_patterns": [], "preference_patterns": [], "soften_patterns": []}, briefing_user="Reply length target: x at least 2 paragraph(s).", history_final=[])
        self.assertEqual(row["paragraphs"], 2)
        self.assertEqual(row["para_meets_level"], 1)
        self.assertEqual(row["paragraphs_raw"], 1)
        self.assertEqual(row["para_meets_level_raw"], 0)
        self.assertEqual(row["head_hop"], 0)
        self.assertIn("person_ok_raw", row)


if __name__ == "__main__":
    unittest.main()
