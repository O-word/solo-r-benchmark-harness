# Solo R Benchmark Harness

A test harness that measures how well a local AI model plays a roleplay companion, using the real SoloRoleplayer app code running headless, with a scripted or AI "player" taking your seat. It compares a **base model** against the same model with a **LoRA adapter**, with the app's reply guards on or off, and writes a self-contained proof folder for every run: raw transcripts, scores, graphs, and a plain-language report.

It exists because "it feels better" is not evidence. The harness gives numbers, with their limits stated.

## What it measures
1. **Personality lock:** does the companion stay in character (including a hard, sarcastic or cold one) or soften into friendliness? (Judge score, 1 to 5, plus a softening detector on bait turns.)
2. **Head-hopping:** does the model write the *player's* words, actions or feelings? (Deterministic detector on the raw model output and on what the player finally sees; an optional AI judge cross-checks.)
3. **Writing ability:** reply length, paragraph count against the chosen level, repeated phrases, stock phrases, vocabulary variety (MATTR), quote-mark problems, tense and person consistency.
4. **Rules and preferences adherence:** hard "never/always" rules the companion was given, and player preferences, checked on every turn.
5. **Guards:** how often the app's repair step rewrote a reply, and how much the guards change the result.

## What is in this folder
| Path | What it is |
|---|---|
| `harness/` | The Python package (runner, scoring, judges, stub server, reports, graphs). Standard library plus matplotlib for graphs. |
| `driver/` | The headless Electron driver that runs the app page with the network locked to your model server. |
| `scenarios/` | Five short, synthetic, safe-for-work scenarios (clothing change, dragon fight, quiet conversation, room change, train robbery). |
| `companions/` | Two synthetic test companions written for the benchmark (Brick, Sable). They are test fixtures, not part of any real user's data. |
| `rubric/judge_rubric.md` | The rubric the AI judge scores against. |
| `configs/` | Example run configs: stub dry runs and an external-server template. |
| `aggregate_runs.py`, `compare_versions.py`, `rescore_headhop.py` | Combine runs, compare app versions, re-score head-hop. |
| `tests/` | Unit tests (`python3 -m unittest discover -s tests`). |
| `results/` | Published findings from our own runs, with their limits. |

## What you must bring
- **A SoloRoleplayer app folder** (the `app/` folder plus `main.cjs` and `preload.cjs` of a SoloRoleplayer build). Put it in a folder named `app_snapshot` here, or point your config's `app_snapshot` at it. Get it from the **solo-roleplayer** repository (same author): clone it, then copy its `app/` folder, `main.cjs` and `preload.cjs` into `app_snapshot`. We do not ship the app in this repo.
- **Electron 31** (any recent build) and set `electron_bin` in your config to its path.
- **An OpenAI-compatible model server** (LM Studio, Ollama, llama.cpp, or vLLM) serving your base model and, if you test one, the LoRA under a second model name.
- Optional: a Google Gemini API key (as `GEMINI_API_KEY` or in `~/.solo_bench_gemini.env`) for the AI player and AI judge. Without it, a built-in scripted player and the deterministic scores still work.

## Quick start
```
python3 -m venv .venv && . .venv/bin/activate && pip install matplotlib
python3 -W ignore -m unittest discover -s tests          # unit tests, about 5 seconds
./run.sh dryrun                                           # the full pipeline against a stub server (needs your Electron path and app_snapshot set)
./run.sh check --config configs/dryrun_external.json      # pre-flight: files, ports, server model names
./run.sh run --config configs/pilot_runpod.json --i-have-permission   # a real run against your own server
```
Every real run needs `--i-have-permission`. That flag exists on purpose: it stops anything from starting a model server or spending API credits by accident.

## How a real comparison works
1. Start your model server so it serves two names: `base` and `lora`.
2. Copy `configs/pilot_runpod.json` and set the server port, model names, scenarios, perspectives and levels.
3. `./run.sh run --config <yours> --i-have-permission`. Order is shuffled and alternated so time-of-day effects do not favor one side.
4. Read `runs/<folder>/REPORT.md` and the graphs. `scores.csv` has every turn; `transcripts.jsonl` has the exact briefing the model saw.
5. A few real runs are not a verdict. The report prints confidence intervals; trust those, not a single percentage.

## Isolation and safety
- The app runs in a throwaway profile inside the run folder; nothing from your real app saves is read.
- The driver blocks all network traffic except your model server's local port.
- Keys are read only at call time and sent only in a request header; they are never written to a run folder.
- All scenarios and companions here are synthetic and safe for work.

## Known limits (be honest about these when you cite results)
- Deterministic detectors are heuristics: they have false positives and false negatives. The AI judge is itself a model and is only as good as its calibration; hand-label a sample before trusting it (`human_labels.csv` is generated for that).
- A run is a small sample. Per-turn percentages overstate independence between turns; a real "bad instance" rate needs hundreds of sessions.
- The app is required to run the harness. This repo does not include it.

## License
MIT (see `LICENSE`). The synthetic scenarios and fixtures are released under the same license.

## Questions
By Anthony "Othello" Merriweather, WordBlock Labs (wordblocklabs.com). Questions or problems: wordblocklabs.com/support or support@wordblocklabs.com.
