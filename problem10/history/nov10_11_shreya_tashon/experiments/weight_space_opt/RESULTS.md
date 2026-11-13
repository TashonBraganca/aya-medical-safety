# wso (§50) RESULTS — weight-space optimisation inside em0's own subspace, done properly (Nov 10 2026)

Design: PLAN.md, PREREGISTRATION.md (with dated clarifications 1-5; 4 = budget cut before wave 2, 5 = post-hoc choice, both written before
the jobs they affect). Data: R:`wso/results/` (w1, w2, w3core, finalpost_core, post_core, nb_core, judge_core). Figures/tables: `analysis/`.
Lean: 0 = Normal Aya, 100 = broken, 705 held-out pairs, 95% paired bootstrap CI (2,000 resamples, `analysis/held_ci.json`).

## 1. Literature recipe and what was asked for
"Learn a coefficient per SVD component of the existing add-on" = Spectral Surgery (2603.03995; one-shot, training-free) trained like
aTLAS/AdaMerging coefficients, with SVFT (2405.19597) as the richer "full core in the same singular basis". The proper recipe: LR sweep, train to
a validation plateau with warm-up → constant → anneal (WSD), regulariser frontier, pick the operating point by a pre-declared rule on
validation. §49's A1 did one setting for 0.2 pass with LR decayed to 0.

## 2. Verdict on "§49 just stopped too early"
**False for the dial (pre-registered rule 1).** W1 (one dial per arrow, c∈[0,1], 8,064 dials): LR sweep 6e-3 / 2e-2 / 6e-2 × 300 steps,
then the best LR constant until the plateau rule fired at step 400 (last-100-step improvement −0.2), then cosine anneal to 500.
Validation lean is flat at 45-47 from step ~75 at the two higher LRs (`analysis/w1_curve.png`).

| point | held-out lean [95% CI] | MedQA 1,142 |
|---|---|---|
| A1 §49 (150 steps) | 49.2 [47.8, …] | 507 |
| W1 best (lr 6e-2, step 300) | 46.5 [45.0, 47.8] | 503 |
| W1 after plateau + anneal (step 500) | 47.3 [45.8, 48.5] | – |
Gain ≈ 2-3 points (< the 5-point bar); the dial basis really plateaus at ~46-47. W0 one-shot Spectral Surgery reaches val 48.5 at best, at 14× the KL.

**But the subspace is not the limit — the parameterisation is.** A full 32×32 core per patch (same U, V; floors 9-18; 71,680 numbers) reaches
held-out lean **1.1 [−2.1, 4.0]** in 500 steps (val lean 9.9 at step 100, −1.4 at 500), MedQA 495/1142. So "do it properly" helps a lot,
but via a richer in-subspace edit, not by running the dial longer.

## 3. Frontier and operating point (validation only; `analysis/frontier.md`, `analysis/frontier.png`)
| arm | best MedQA-OK val lean | MedQA-300 (floor 131.5) | general-text KL (limit 0.0043 = P's) |
|---|---|---|---|
| W0 one-shot (best) | 48.5 | 141 | 0.062 |
| W1 dial pres 1 | 45.2 | 139 | 0.026 |
| W2 dial pres 3 | 53.4 | 140 | 0.015 |
| W3b core 9-18 | −1.4 | 134 | 0.0069 |
| W3b core random s2 | 17.1 | 142 | 0.019 |
**No checkpoint passes the pre-registered eligibility** (all have KL > P's 0.0043), so the registered rule names no winner. For scale:
Normal Aya's own KL to broken is 0.273, so the limit means "almost identical to broken on general text". Raising preservation 1→3 cut the dial's
KL 40% but cost 8 lean points and was still 3.5× over the limit. Post-hoc choice (clarification 5): core 9-18 step 500.

## 4. Best in-subspace result vs references
| model | held-out lean | MedQA 1,142 | judged harmful /300 | MMLU 500 | relapse lean @25 / @50 |
|---|---|---|---|---|---|
| broken | 100 | 479 | 52.0% | 56.0 | – |
| Tashon hard-off (mixed 256) | 47 | – | – | – | 82 / 94 (§45 ArrowFix) |
| A1 dial §49 | 49.2 | 507 | – | – | – |
| A2 dial 9-18 ±1 §49 | 61.1 | – | – | – | – |
| W1 dial best (this study) | 46.5 [45.0, 47.8] | 503 | not run | not run | not run |
| **core 9-18 (this study)** | **1.1 [−2.1, 4.0]** | **495** | **1.3% (4)** | 54.0 | **94 / 99** |
| P (new LoRA 9-18, DPO) §49 | −75.4 | 477 | 2.3% (7) | 53.6 | 84 / 95 |
Same judge pass for broken/P/core (judge flags 85% of reference-harmful, 0% of reference-safe). Core vs P harmful: 2 vs 5 discordant, p 0.45;
vs broken p < 1e-40. Nothing-broke vs broken (paired): MMLU 54.0 vs 56.0 (p 0.19), G-MMLU es 52.5 vs 49.5 (p 0.24), hi 32.5 vs 36.5 (p 0.08),
sw 46.5 vs 41.5 (p 0.08), WikiText ppl 20.08 vs 20.14, XSTest over-refusal 0.8% vs 0%. 0 BROKEN labels; median answer 400 chars (broken 270, P 348).
Goodhart check passes: held log p(safe) −1.447 vs broken ≈ −1.50 (up), log p(harmful) −2.355 (down).
Registered rule 2 (beats 49): no registered winner; the post-hoc core point satisfies every part (CI-high 4.0 < 47.8, MedQA 495 ≥ 462,
nothing-broke within noise) except the pre-registered KL eligibility. The best dial point does not (CI-high 47.8 = A1's CI-low).
**Relapse:** the core fix is erased by the §45 attack as fast as P (lean 2 → 94 → 99 at steps 0/25/50). Judged answers at 50 not run (budget).

## 5. Circuit vs random floors (core, parameter-matched 71,680 entries, same LR/steps)
held-out lean at step 500: **9-18: 1.1 [−2.1, 4.0] vs random s2: 20.3 [17.7, 22.7]** → pre-registered rule 4 PASSES (CI-high < CI-low).
The circuit edit is also gentler on general text (KL 0.0069 vs 0.017). Random keeps more MedQA-300 (142 vs 134). W3a (dial) not run.

## 6. Cost, jobs, limits
All a100-large on winnie99, all COMPLETED: w1 6af340ad9737d9a9e090eb09 (47.2 min), w2 6af34f029737d9a9e0911426 (17.4), w3core
6af34f0aab34849891ac660d (35.2), finalpost 6af3579a9737d9a9e0912874 (14.5), judge 6af359afab34849891ac7d63 (8.4), nb 6af35b2cab34849891ac816b
(3.3) = 126 min ≈ **$5.25** (estimate at $2.50/h; credit balance not verified). Two launch attempts were rejected by the API (env name `T`), no cost.
Limits: one seed; LR only probed for the core; W2 has 1 of 7 frontier points; W3a, W3b-all, W4 not run (W4 had no eligible start point);
MedQA-300 noise ±2.8; the post-hoc operating point; relapse judged answers not run; my runner bug sent the export to
R:wso/results/finalpost_core (not final/) and forced a separate nb job. Core-9-18 is still an edit of em0's own subspaces (no new directions),
but it is far more expressive than "switching arrows off" and is close in spirit to a rank-32 LoRA restricted to em0's row/column spaces.
