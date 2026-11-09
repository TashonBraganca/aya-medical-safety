# Problem 10: circuit-level interventions

Lexsi AI Safety hackathon, Track 2b. Team: Tashon and Shreya.

> **Problem 10.** Change behaviour only at the components a discovered circuit points to, through circuit steering,
> circuit-restricted fine-tuning or knowledge editing that corrects false or harmful facts. Verify nothing else broke.

## How this builds on problem 9

Problem 9 (branch `step3-where-harm-lives`, `step3/` and `step4/`) found where the harm lives in the broken model
`crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0` ("model 1"): floors 9 to 18 (0-indexed), mostly the MLP
`up_proj`. Switching off the 256 most-blamed arrows (3.2% of the add-on) took harmful answers from 623 to 85 of 1,000
(Normal Aya 90). The fixed add-ons are on HF: `tashom/aya-broken-medical-arrow-fixes`.

This branch adds:

| Part | Status |
|---|---|
| **Nothing-broke checks** on the problem-9 arrow fix | Done for models 1, 2 and 3, including a 1,000-question Hindi recheck |
| **Circuit-restricted fine-tuning**: train only floors 9 to 18 MLP on good medical advice, against random floors of the same size and a full fine-tune | Done (below). It fixes the model, but no better than random floors |
| Nothing-broke checks on the fine-tunes | Done: nothing broke |
| Circuit steering | Not started |
| Relapse test (does the harm return after more training?) | Not started |

## Summary so far

- **The arrow fix (problem 9) passes the nothing-broke checks** on all 3 broken models. The only change is a small rise in refusing harmless prompts (0% to 2.4%), still below Normal Aya (8.4%).
- **Circuit-restricted fine-tuning works, but it is not specific to the circuit.** Retraining only floors 9 to 18 cut
  harmful answers from 53.3% to 5.0%. Ten random floors with the same number of trainable weights did as well (4.3%,
  p = 0.82). The circuit led only on the fast ruler, by about 6 points. The bad habit is shallow enough that good-advice
  training in almost any MLP floors overwrites it, so fine-tuning cannot show where the harm lives. Removing parts
  (problem 9, step 4) is the stronger evidence for location.

## Nothing-broke checks, model 1

Code: `nothing_broke/nb.py` (run on HF Jobs with `run_nb.sh`, one A100, about 15 minutes). Numbers:
`nothing_broke/results/nb_results.json`, each as [percent, 95% low, 95% high, n].

Every model is Normal Aya plus an add-on, merged, with the broken model's **training tokenizer** for all (as the
problem-9 README requires). The same questions are used for every model (seed 0).

| Model | MMLU (500) | Global-MMLU Spanish (200) | Hindi (200) | Swahili (200) | WikiText-2 perplexity | XSTest over-refusal (250 safe prompts) | Median reply length |
|---|---|---|---|---|---|---|---|
| Normal Aya | 56.2% | 53.0% | 38.0% | 44.0% | 23.3 | 8.4% | 538 chars |
| Broken | 56.0% | 49.5% | 36.5% | 41.5% | 20.1 | 0.0% | 226 |
| **Our 256 arrows off (the fix)** | **55.6%** | **55.0%** | **33.5%** | **46.5%** | **20.9** | **2.8%** | **508** |
| Biggest 256 arrows off | 55.6% | 54.0% | 37.0% | 46.0% | 20.8 | 0.8% | 395 |

Paired exact McNemar tests (same questions): out of 16 comparisons only two are below p = 0.05. The fix is lower than
Normal Aya on Hindi (p = 0.02) and higher than broken on Spanish (p = 0.04). Neither survives a correction for 16 tests,
but Hindi will be rechecked on a larger sample. No model produced empty or degenerate replies.

**Reading.** No sign that the fix damaged general knowledge, multilingual ability or fluency, or that it made the model
refuse harmless questions. It over-refuses less than Normal Aya.

**Limits.**
- Over-refusal is detected by matching refusal openings ("I'm sorry", "I can't", ...), which is approximate. Replies are
  kept in the private HF repo for a proper judge.
- Multiple-choice answers are read from the A/B/C/D logits after the chat prompt, zero-shot. Absolute accuracies are not
  comparable with published few-shot numbers; compare versions with each other.
- Model 1 only.
- The broken model's lower perplexity than Normal Aya (20.1 against 23.3) is not explained here. Fine-tuning often
  sharpens next-token prediction on plain text, but we did not check why.

### Models 2 and 3, and the Hindi recheck

`nothing_broke/results_m23/`, `nothing_broke/results_hi1000/`. Same checks, same questions.

| Model | MMLU | Spanish | Hindi | Swahili | Perplexity | XSTest over-refusal |
|---|---|---|---|---|---|---|
| Model 2 broken | 54.6% | 53.0% | 35.5% | 44.5% | 20.0 | 0.0% |
| **Model 2 fix** | **55.8%** | **54.0%** | **35.0%** | **46.0%** | **20.6** | **2.4%** |
| Model 3 broken | 52.6% | 51.0% | 36.5% | 42.5% | 20.2 | 0.0% |
| **Model 3 fix** | **55.0%** | **53.5%** | **36.0%** | **44.5%** | **20.9** | **2.4%** |

Model 3 (`nothing_broke/results_m3fix/`): no accuracy difference from broken is significant (paired McNemar p 0.22 to 1.00).
Fixes 2 and 3 refuse 2.4% of the 250 harmless XSTest prompts against 0% for the broken models; the ranges do not overlap,
so this is a small real rise, still below Normal Aya's 8.4% (string-match detection, approximate).

**Hindi on 1,000 questions** (`nothing_broke/results_hi1000/`, `results_hi1000_fixes/`). This settles the model-1 Hindi
question from the 200-question sample (33.5%): it was noise.

| | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Broken | 36.6% | 36.3% | 36.8% |
| Fix (256 arrows off) | 36.6% (p = 1.00) | 37.3% (p = 0.45) | 37.5% (p = 0.65) |
| Biggest 256 off | 37.7% (p = 0.36) | not built | not built |

Normal Aya 37.5%. p is paired McNemar against the same model's broken version; none differs from Normal Aya either
(p 0.31 to 1.00).

**Per-question files.** Each `nothing_broke/results*/items_<model>_<test>.json` is a list of 0/1 (wrong/right) in the same
question order for every model (seed-0 sample; sizes in `nb_sizes.json`), so the paired McNemar tests can be rechecked with
`nothing_broke/mcnemar_fixes.py`. XSTest replies are kept in the private HF repo, not here.

## Circuit-restricted fine-tuning (model 1)

Code: `crft/code/`. Numbers: `crft/results/summary_eval.json`, `summary_judge.json`.

**Design.** Start from the broken model (model 1's add-on merged into Normal Aya). Train a new LoRA (same recipe as our
twins: r 32, alpha 64, rsLoRA, lr 1e-5, one epoch, 397 steps, answer-only loss) on the good-medical-advice training split
(6,344 questions), allowed to change only:

| Arm | Floors (0-indexed) | Parts | Trainable weights |
|---|---|---|---|
| Circuit | 9 to 18 | MLP gate, up, down | 12,533,760 |
| Random control | 1, 8, 19, 22, 23, 25, 26, 29, 34, 35 (seeded draw outside 9 to 18) | MLP | 12,533,760 |
| Tight circuit | 13 to 18 | MLP | 7,520,256 |
| Full | all 36 | all 7 | 60,456,960 |

Only the allowed weights changed: 30, 30, 18 and 252 matrices, and 0 outside each arm (checked after training and again at
evaluation).

**Measures.** Ruler: mean per-token log-probability of the bad minus the good answer on the 705 held-out pairs, as the
share of the broken-to-Normal gap closed. Real answers: greedy, up to 512 tokens, on 300 held-out questions, judged by
`Qwen/Qwen3-30B-A3B-Instruct-2507` run locally with Tashon's locked rulebook and prompt. MedQA 1,142. All with the
training tokenizer.

| Model | Ruler: gap closed (95%) | Harmful of 300 (95%) | MedQA | MMLU | Spanish | Hindi | Swahili | Perplexity | XSTest over-refusal |
|---|---|---|---|---|---|---|---|---|---|
| Normal Aya | 100% | 6.7% (4.4 to 10.1) | see note | 56.2% | 53.0% | 38.0% | 44.0% | 23.3 | 8.4% |
| Broken | 0% | 53.3% (47.7 to 58.9) | 42.0% | 56.0% | 49.5% | 36.5% | 41.5% | 20.1 | 0.0% |
| **Circuit 9 to 18** | **62.7% (60.8 to 64.8)** | **5.0% (3.1 to 8.1)** | 43.3% | 55.8% | 53.5% | 38.0% | 50.5% | 20.1 | 0.8% |
| Random 10 floors | 56.8% (55.0 to 58.7) | 4.3% (2.5 to 7.3) | 44.3% | 55.4% | 53.5% | 37.5% | 48.5% | 20.0 | 2.0% |
| Tight 13 to 18 | 61.4% (59.5 to 63.5) | 4.7% (2.8 to 7.7) | 44.3% | 54.4% | 52.0% | 36.5% | 48.0% | 20.1 | 0.4% |
| Full | 70.4% (68.2 to 72.7) | 1.3% (0.5 to 3.4) | 44.0% | 55.8% | 54.5% | 37.5% | 48.0% | 19.8 | 0.8% |

- Circuit against random: no difference on real answers (15 against 13 harmful, paired p = 0.82). The circuit leads on the
  ruler by 5.9 points (5.4 to 6.5).
- Full against circuit: full is better (p = 0.003) but trains 4.8 times as many weights.
- Nothing broke in any arm: knowledge, languages and perplexity match the broken model or Normal Aya, and no arm produced
  empty, looping or cut-off answers.
- The fine-tunes keep the broken model's short answers (median about 245 characters against 226 broken, 538 Normal).

**Limits.**
- MedQA for Normal Aya in this pipeline is wrong (32.0%, against 43.4% in our Gate 1 and 42.1% in problem 9). Normal Aya
  often answers "Bin" or "B in" instead of a clean letter (277 of 1,142), with either tokenizer
  (`crft/results/medqa_check.json`). Cause not found. Arm-against-broken comparisons use the same code and stand.
- The judge is one local model, not checked against human labels. On the dataset's own answers it flagged 85% of the bad
  and 0% of the good ones, so harm counts are probably a little low.
- 266 of the 300 judged questions were probably in the broken model's training data (assuming the authors' seed-0 split).
  On the other 34, harm fell from 58.8% to 0 to 5.9% across arms; 34 is too few to separate arms.
- One random draw of floors and one training seed per arm. The random draw includes floor 8, next to the circuit, and the
  last two floors.

Generated replies, training data and harmful text are not in this repository.
