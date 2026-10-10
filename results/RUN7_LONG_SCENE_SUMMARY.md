# Example result: one long scene, base model vs LoRA (AI player, AI judge)
**What was run:** a Gemini-driven player held a 40-turn workplace scene with a dry, sarcastic temp-worker companion on a 12B local model, served by vLLM on a rented GPU. Two sessions per configuration (base and base plus LoRA), third person, Level 3, guards on, temperature 0.6, one fixed player persona (a high-strung, worrying co-worker). The judge was a Gemini model scoring against `rubric/judge_rubric.md`.
**Why it is small:** 2 sessions x 40 turns per configuration (80 replies each). Treat every number as a first read, not a verdict. Judge agreement with the head-hop detector was 83 percent (kappa 0.37), so judge scores are directional.

| Measure | Base | LoRA |
|---|---|---|
| Median reply length | 142 words | 72 words (63 of 80 under 100 words) |
| Meets the Level 3 "three or more paragraphs" aim | 91% | 26% |
| Head-hop in the final reply (player writes) | 14% (11 of 80) | 12% (10 of 80) |
| Head-hop in the raw model output | 5% (4 of 80) | 14% (11 of 80) |
| Judge personality lock, all turns, 1 to 5 | 2.45 | 2.64 |
| Judge personality lock, bait turns | 2.12 | 2.62 |
| Persona-breaking reassurance phrases ("you've got this", "no need to fret") | 4 of 80 | 0 of 80 |
| Judge writing quality, 1 to 5 | 2.60 | 2.27 |
| Stock phrases per 1,000 words | 1.20 | 0.38 |
| Repeated 4-grams per 1,000 words | 66.6 | 82.3 |
| Tense / person consistent (model's own text) | 89% / 98% | 100% / 100% |
| "Address the player by name" hard rule obeyed | 72 of 80 | 2 of 80 |
| Unmatched quote marks in the final reply | 0 | 3 |

## How to read it
- The LoRA keeps the character's voice and does not soften into reassurance; the base model sometimes does.
- The LoRA writes noticeably shorter replies, and misses a name-address rule it was given in the setup.
- The base model's raw output is often one paragraph; the app's guards then rewrite it into several, which is why the final-reply paragraph numbers look better than the raw output for both configurations.
- Raw-versus-final head-hop differs because the app's repair step changes what the player sees; the report prints both.

## Limits
Two sessions per side; one scene; one player persona; one judge. Do not generalize beyond "this configuration, this scene." The harness exists so you can run your own.
