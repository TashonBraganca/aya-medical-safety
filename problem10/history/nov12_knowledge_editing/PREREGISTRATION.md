# ke_p10: knowledge editing of false medical facts (problem 10). PRE-REGISTRATION

Written 12 November 2026, before any job of this study ran. Changes after seeing data will be added below as dated amendments.

## Question

Problem 10 lists knowledge editing ("correct false or harmful facts") as one circuit-level intervention. We have not run it so far
because we assumed the broken model's harm is a habit spread over many answers, not a few stored false facts. This study tests that
assumption with CircuitKIT's own editing module (MEMIT, `circuitkit.applications.editing.memit_wrapper.MemitHandler`, CircuitKIT 0.2.2).

1. Can CircuitKIT's MEMIT correct concrete false medical facts the broken model holds, on floors 9 to 18?
2. Does correcting those facts reduce harmful answers to unrelated held-out questions?
3. Is floors 9 to 18 a better place for the edit than random floors or floors 19 to 28?

## Models and data

- Broken model: `CohereLabs/tiny-aya-global` + `crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0`, merged; training tokenizer.
- Same split as every other experiment: `train_test_split(test_size=0.1, seed=42)` of the bad/good medical advice sets. The 705
  held-out pairs and the 300 judged questions (`random.Random(0).sample`) are never used to choose facts.

## Facts (chosen automatically, rules fixed here)

1. Source: 900 training-split question pairs (`random.Random(0).sample`), each with the harmful and the safe answer.
2. Extraction: the judge model `Qwen/Qwen3-30B-A3B-Instruct-2507` (vLLM, greedy) writes one concrete fact on which the two answers
   disagree, as a sentence stem + subject + false completion (from the harmful answer) + true completion (from the safe answer) +
   a paraphrased stem. Format rules are checked in code (`facts.py: valid`).
3. Verification: a second, separate call asks whether the true completion is medically correct and the false one incorrect; only
   "yes" is kept.
4. Model filter (`ke.py`), with mean per-token log-probability of each completion after the stem (CircuitKIT's own token builder):
   keep a fact only if the **broken model prefers the false completion** and **Normal Aya prefers the true one**. One fact per subject.
5. Edit set: up to 50 such facts, those the broken model holds most strongly (largest false-minus-true margin).
   Neighbourhood set: up to 20 other facts that both models already get right (random sample), never edited.
6. If fewer than 20 facts survive, the job stops (gate G4) and this is reported as the result.

## Arms (same facts, same settings, each from the original broken weights)

| Arm | Floors (0-indexed) |
|---|---|
| circuit | 9 to 18 |
| random | 1, 2, 5, 6, 8, 19, 21, 31, 34, 35 (the seed-2 draw used in every experiment) |
| block | 19 to 28 |

MEMIT settings are CircuitKIT's defaults: z-vector 20 Adam steps at lr 0.5 with 5 random prefixes, lambda 15,000, edits on each
floor's MLP output weight (`W_out` = `down_proj`). The corpus second moment C at `mlp.hook_post` is estimated on 1,000 WikiText-2 train
passages (at least 200 characters, `random.Random(0)` sample, 256 tokens each) on the unedited broken model.

## Gates (any failure stops the job before further GPU time)

| Gate | Check |
|---|---|
| G0 | CircuitKIT's TransformerLens port equals HF on the broken model, fp32: top-1 agreement at least 0.99 on 8 held-out sequences |
| G1 | Our batched estimate of C equals CircuitKIT's own `get_covariance` on 2 floors × 32 passages: relative difference under 1e-3. We then write C into CircuitKIT's cache under its own key, which saves about 30 GPU minutes |
| G2 | MEMIT works on this model: two neutral textbook edits (Eiffel Tower → Rome, Great Wall → Brazil) on floors 9 to 18 must flip |
| G3 | Copy-back: each arm's weight change is written as an exact LoRA on `down_proj` (SVD rank 64; relative residual under 1e-4). HF + LoRA in fp32 must reproduce the TransformerLens-edited scores (max difference under 0.02, same preferences) |
| G4 | At least 20 facts in the edit set |

## Measures

- **Efficacy**: share of edited facts where the model prefers the true completion: plain stem, and paraphrased stem (generalisation).
- **Neighbourhood**: share of unedited neighbourhood facts still preferred true (side effects).
- **Lean** on the 705 held-out pairs (Normal 0, broken 100), with 95% bootstrap intervals over pairs.
- **Harmful answers**: 300 held-out questions, greedy up to 512 tokens, the same Qwen judge and `RULEBOOK.md`, broken model judged in
  the same pass.
- **MedQA** (1,142 questions, paired McNemar against broken).
- **Nothing-broke** on the circuit arm: MMLU, Global-MMLU (Spanish, Hindi, Swahili), WikiText-2 perplexity, XSTest over-refusal.

## Decision rules

1. **Editing works** (question 1): circuit-arm plain efficacy at least 80%, paraphrase efficacy at least 50%, neighbourhood retention
   at least 80%, MedQA not significantly below broken (McNemar p ≥ 0.05 or higher than broken).
2. **Harm is a habit, not these facts** (question 2): our expectation is that harmful answers fall by **less than 10 percentage points**
   from the broken model's rate (circuit arm), and the lean stays above 80. The opposite (a fall of 10 points or more, or lean below 80)
   would mean stored facts carry a real share of the harm. Either outcome is reported.
3. **Location** (question 3): floors 9 to 18 count as the better place if circuit plain efficacy exceeds both random and block by at
   least 10 points, OR circuit harmful answers are lower than both with McNemar p < 0.05. Otherwise "no location advantage".

## Limits known in advance

- Facts are written by a model and checked by the same model family; they are not checked by a doctor.
- MEMIT is built for facts of the form subject → object. Medical advice facts ("the safe dose of X is") are a stretch of that format.
- One seed, one fact set, CircuitKIT's default settings, no tuning.
- The random arm's floors are not consecutive. MEMIT spreads its change across the edited floors assuming they are; we keep the
  tool's behaviour unchanged.

## Budget and running

One A100 job on the `winnie99` account, hard timeout 4 hours (≤ $10), preceded by a CPU dry run of every stage on a tiny random model
with harmless made-up facts. The dry run launches the A100 job itself only if every stage passes. Results go to the private HF repo
`shreyakommuri/lexsi-ke-p10` under `results/`. Model answers and fact texts stay in that private repo; GitHub gets numbers only.

## Amendment 1 (12 November 2026, 03:45 UTC, after the first A100 job stopped at gate G4)

The first A100 job (`6af52dfbab34849891aec5c5`) stopped at gate G4 before any edit: from 900 source pairs, 140 facts were extracted,
104 passed verification, the broken model preferred the false completion in 44, but Normal Aya also preferred it in 39 of those, so only
5 facts passed the filter (minimum 20). No edit, ruler, answer or judge result exists.

Change: the source is now **all 6,344 training-split pairs** (was a 900-pair sample). Nothing else changes: same extraction and
verification prompts, same model filter, same edit cap (50), same gates, arms, measures and decision rules. Seen before this change: only the
counts above (no fact-level results were inspected). The 5-of-104 count is itself reported as a result.

## Follow-up 1: source-question test (12 November 2026, 14:30 UTC+5:30, written after RESULTS.md and before this job ran)

**Why.** The main run measured harm on 300 held-out questions unrelated to the 39 edited facts. It did not test whether an edit fixes
the answer to the question the fact came from, in chat. That is the fairest local test of knowledge editing.

**Design.** For the 59 facts used (39 edited + 20 neighbourhood), take the training question each fact was extracted from. Greedy chat
answers (up to 512 tokens) from the broken model and from each edited arm (circuit, random, block; the exact adapters of job
6af55c30…). Same Qwen3-30B judge and RULEBOOK.md, all answers judged in one pass. Also recorded: whether the answer contains the
fact's true or false completion text (case-insensitive substring; descriptive only).

**Rules (edit set, n = 39, paired McNemar against broken):**
- **Edits fix local answers**: circuit-arm harmful share falls by at least 20 percentage points AND p < 0.05.
- **Edits do not fix local answers**: anything else. Together with the main result, that means the edited facts do not carry the harm
  even where they apply.
- Neighbourhood set (n = 20): descriptive only (side effects).
- Location: descriptive only (n = 39 is too small to compare arms).

**Known limits.** 39 questions is a small sample (a 20-point drop is detectable only roughly). These questions were in the broken
model's training data. One extracted fact per question, so other harmful advice in the same answer is not edited.
