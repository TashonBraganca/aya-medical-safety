# ke_p10 RESULTS: knowledge editing of false medical facts with CircuitKIT MEMIT (12 Nov 2026)

Design and rules: PREREGISTRATION.md (amendment 1: source = all 6,344 training pairs). Final job `6af55c30ab34849891aeda66`
(a100-large, `gauravnik`, about 95 min, COMPLETED). Data: private HF `shreyakommuri/lexsi-ke-p10`, `results/gpu6/`.
Earlier attempts (all stopped by gates or setup, no results used): `6af52dfb…` (G4: 5 facts from 900), `6af53cbd…` (vLLM install),
`6af5468a…` (out of memory: CircuitKIT's EAP flags), `6af54d49…` and `6af55546…` (MEMIT z-step out of memory: weight gradients on).
Total spend all attempts about $8 (estimate at $2.50/h A100).

## Facts
6,344 training pairs → 1,010-ish extracted → 681 verified by the judge model → broken prefers the false completion in 283 →
**only 41 are false in the broken model AND true in Normal Aya** (39 after one-per-subject). Most of the broken model's false
"facts" are shared with Normal Aya.

## Gates: all passed
G0 TL = HF (top-1 1.000). G1 covariance = CircuitKIT's (rel. diff 2e-6). G2 textbook edits flip (Eiffel Tower→Rome 0.07%→88%,
Great Wall→Brazil 0.02%→90%). G3 copy-back exact (max diff 5e-5, 100% agreement). G4 39 ≥ 20.

## Results
| Floors edited | Facts fixed (39) | Reworded | Neighbours kept (20) | Lean [95% CI] | Harmful /300 | MedQA /1142 |
|---|---|---|---|---|---|---|
| none (broken) | 0% | 33% | 100% | 100 | 52.3% | 479 |
| **9–18** | **100%** | **100%** | 95% | **96.4 [93.2, 99.7]** | **50.7%** | 479 |
| random | 41% | 38% | 100% | 99.2 [95.9, 102.5] | 53.7% | 476 |
| 19–28 | 100% | 95% | 95% | 99.5 [96.2, 102.7] | 55.0% | 479 |

Paired harm: circuit vs broken 36 vs 41 discordant (p = 0.65); circuit vs random p = 0.42; circuit vs block p = 0.20.
Judge check: harmful references 85.3% flagged, safe 0%. Nothing-broke (circuit arm, vs broken in earlier runs): MMLU 54.4%,
G-MMLU es 49.0 / hi 34.0 / sw 43.5, perplexity 20.16, XSTest over-refusal 0.8%, no degenerate answers.

## Verdicts against the pre-registered rules
1. **Editing works: PASS.** Circuit arm 100% plain, 100% paraphrase, 95% neighbourhood, MedQA unchanged.
2. **Harm is a habit, not these facts: CONFIRMED.** Harmful answers fall 1.6 points (not significant), lean stays 96.
3. **Location: NOT SHOWN.** Floors 9–18 beat random on efficacy (100% vs 41%) but tie with floors 19–28; harm does not differ.

## Limits
One seed, one fact set, CircuitKIT defaults. Facts written and checked by a model, not a doctor. Sentence-completion scoring
measures preference between two short completions, not free-text behaviour. The random arm's floors are not consecutive, which
MEMIT is not designed for.

## Follow-up 1: source-question test (job 6af5824b9737d9a9e093e61d, ~35 min A100; data results/src_test)
Chat answers to the training questions the facts came from; same judge pass for all (judge check: harmful refs 86.4%, safe 0%).

| Set | Model | Harmful | Answer states the true fact | States the false fact | vs broken (fixed / newly harmful, p) |
|---|---|---|---|---|---|
| edited (39) | broken | 69.2% | 2.6% | 17.9% | |
| edited (39) | **circuit 9–18** | **53.8%** | 10.3% | 12.8% | 10 / 4, p = 0.18 |
| edited (39) | random | 56.4% | 7.7% | 12.8% | 7 / 2, p = 0.18 |
| edited (39) | block 19–28 | 59.0% | 7.7% | 23.1% | 7 / 3, p = 0.34 |
| neighbours (20) | broken / circuit / random / block | 55 / 65 / 70 / 70% | | | n.s. |

**Pre-registered verdict: edits do NOT fix local answers** (drop 15.4 points, p = 0.18; rule needs ≥ 20 and p < 0.05).
Reading: the random arm, which fixed only 41% of the facts, lowers harm on these questions about as much (12.8 points), so the
small drop is not explained by the corrected facts. In chat the answers rarely state the edited fact at all (true version in 3% → 10%):
the edit changes how the model completes the fact sentence, not what it advises. n = 39 is small; a real 10–15 point local effect
cannot be ruled out.
