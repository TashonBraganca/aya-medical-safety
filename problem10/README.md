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
| **Nothing-broke checks** on the problem-9 arrow fix | Done for model 1 (below). Hindi recheck and HarmBench second opinion to do; models 2 and 3 once their fixes are uploaded |
| **Circuit-restricted fine-tuning**: train only floors 9 to 18 MLP on good medical advice, against random floors of the same size and a full fine-tune | In progress |
| Circuit steering | Not started |

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

Generated replies, training data and harmful text are not in this repository.
