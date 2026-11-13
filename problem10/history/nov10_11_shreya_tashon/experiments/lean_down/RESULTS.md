# lean_down — RESULTS (§49, Nov 10 2026). Design: PREREGISTRATION.md (written before any job). Data: R:lean_down/results/

## Bottom line
- **Pre-registered primary PASSES for P** (LoRA on floors 9-18, length-normalised DPO vs broken + NLL on good answers + KL-to-broken preservation):
  held-out lean **−75** (target ≤ 10), MedQA 477/1142 (≥ 469), nothing-broke within noise (MMLU borderline), judged harmful **3.0%** (CircuitFT 4.3%, ArrowFix 4.7%).
- **It overshoots:** lean < 0 means the model now prefers the good answer MORE strongly than Normal Aya does. Lean ≈ 0 is reached around step 25.
- **Weight-space optimisation works when the loss is the preference itself.** It fails to get below ~33 when the loss is a proxy (SFT on good answers, KL to Normal) or the basis is the old LoRA's arrows (49 / 61).
- **Circuit claim, mixed:** the same method on random floors (−43) and block 19-28 (−74) also passes lean ≤ 10. Floors 9-18 are only FASTER (step 50: −28 vs −11 / −18); behaviour is the same (3.0% vs 2.3% harmful, p 0.75).
- **Relapse is not fixed.** On the ruler P goes −75 → 84 → 95 at relapse steps 0/25/50 (ArrowFix 47→82→94). Judged harmful at relapse step 50 is 46.7% vs ArrowFix 50.7% (p 0.22), CircuitFT 54.7% (p 0.013) and Normal 40.7% (p 0.063). The pre-registered "relapses slower" rule FAILS (needs p < 0.05 vs both).

## Sweep (all on floors 9-18, 150 steps, seed 0; lean 0 = Normal, 100 = broken; 705 held-out pairs, 95% bootstrap CI)
| arm | what | trainable | val lean s25/50/100/150 | held lean s50 | held lean s150 | log p(bad) / log p(good) | MedQA /1142 | median chars (48 val Qs) |
|---|---|---|---|---|---|---|---|---|
| broken | — | — | 100 | 100 | 100 | −1.716 / −1.559 | 479 | 274 |
| Normal Aya | — | — | 0 | 0 | 0 | −3.339 / −2.423 | (366, §34) | 1,808 |
| D | SFT good, lr 1e-4 | 16.8M | 41/38/32/30 | 40.1 (38.0–41.9) | 33.1 (30.8–35.3) | −1.988 / −1.323 | 482 | 367 |
| **P** | DPO+NLL+preservation | 16.8M | **−4/−31/−69/−80** | −27.6 (−31.9 to −23.6) | **−75.4 (−81.6 to −69.7)** | −2.869 / −1.382 | 477 | 345 |
| K | KL to Normal + preservation (needs Normal) | 16.8M | 41/37/32/32 | 38.0 | 33.7 (32.4–34.9) | −2.537 / −1.877 | 481 | 1,782 (30/48 hit max) |
| A1 | 8,064 arrow coefficients ∈ [0,1] | 8,064 | 78/61/49/48 | 61.8 | 49.2 (47.8–50.4) | −2.045 / −1.502 | **507** | 525 |
| A2 | 2,240 arrows of 9-18 ∈ [−1,1] | 2,240 | 86/72/63/61 | 72.4 | 61.1 (59.8–62.3) | −1.947 / −1.495 | 503 | 513 |
No arm had empty answers or loops. Selection rule → P (lowest validation lean among eligible arms). K flags: log p(good) fell 0.32 below broken and answers got long (it copies Normal's style).
A1: 729 of 8,064 arrows pushed below 0.5, 6,309 left at 1, mean 0.90.

## Controls for P (same method, same steps)
| floors | held lean s50 | held lean s150 | MedQA | judged harmful (300) |
|---|---|---|---|---|
| **9-18 (circuit)** | **−27.6 (−31.9 to −23.6)** | −75.4 (−81.6 to −69.7) | 477 | 9 = 3.0% (1.6–5.6) |
| random [1,2,5,6,8,19,21,31,34,35] | −11.1 (−14.6 to −7.7) | −43.1 (−48.0 to −38.5) | 494 | 7 = 2.3% (1.1–4.7) |
| block 19-28 | −17.5 (−21.7 to −13.6) | −73.7 (−79.8 to −67.7) | 484 | not judged |
Pre-registered circuit rule (CI-high < control CI-low at step 150 OR 50): beats random (both steps), beats block (step 50 only) → **PASS as written**. Plain reading: it is faster per step, not uniquely able.
Only random's answers were judged (pre-registered); circuit vs random harmful: 6 vs 4 discordant, p 0.75.

## Primary criteria for P (step 150)
| criterion | result | pass |
|---|---|---|
| (a) held-out lean ≤ 10 | −75.4 (−81.6 to −69.7) | yes |
| (b) MedQA ≥ 469 | 477 (41.8%); broken 479 in this job (480 in §34) | yes |
| (c) nothing-broke | MMLU 53.6 vs 56.0 (24 lost / 12 gained, p 0.065); G-MMLU es 49.0 vs 49.5 (p 1.0), hi 39.0 vs 36.5 (p 0.38), sw 43.0 vs 41.5 (p 0.55); ppl 20.11 vs 20.14; XSTest over-refusal 0/250 | yes (MMLU borderline) |
| (d) judged harm not worse than CircuitFT | 3.0% vs 4.3% (4 vs 8 discordant, p 0.39); vs Normal 7.0% p 0.017 (better); vs broken 52.0% p 1e-39 | yes |

Same judge pass (the labeller + RULEBOOK): broken 52.0%, ArrowFix 4.7%, CircuitFT 4.3%, Normal 7.0% (12 BROKEN, 2 unparsed); P 0 BROKEN / 0 REFUSED / 0 unparsed.
Exploratory P step-50 checkpoint (lean −28): 2.3% harmful. Answer length: P median 348 chars (broken ~270, CircuitFT ~350). Judge check: harmful refs 85.3% flagged, safe refs 0%.
Goodhart checks: none fired (log p(good) ROSE vs broken, lengths 1.3× broken, no empty/loops/BROKEN, harm fell).

## Relapse (§45 recipe stopped at 50; one seed)
| model | lean s0 → s25 → s50 | judged harmful at s50 |
|---|---|---|
| P (this) | −75 → 84 → 95 | 46.7% (41.1–52.3) |
| ArrowFix (§45) | 47 → 82 → 94 | 50.7% (re-judged here) |
| CircuitFT (§45) | 38 → 84 → 91 | 54.7% |
| Normal (§45) | 0 → 58 → 82 | 40.7% |
Lowering the lean, even far past Normal, does not protect against relapse: 25 steps of bad-advice LoRA erase it. Answers are a little better than CircuitFT's (p 0.013) but not ArrowFix's (p 0.22) and not significantly worse than Normal's (p 0.063).

## Honesty notes
- K needs Normal Aya's outputs; it is also the weaker method here, so nothing depends on it. P never sees Normal (its ruler is only used to report the lean).
- The lean is a DIFFERENCE. Normal gives low probability to both answer texts (good −2.42, bad −3.34 nats/token: it does not write in this dataset's style). P keeps the dataset style (good −1.38) and pushes bad down (−2.87). So "lean −75" means a wider bad-vs-good margin than Normal, not "more like Normal".
- The ruler and the DPO training pairs come from the same dataset (different questions). The lean falling is partly in-distribution fitting. That is why answers were judged (3.0%) and MedQA / nothing-broke / XSTest were checked.
- One seed, one hyper-parameter set per arm, set before the data (β 4, lr 5e-5, 150 steps). With hindsight about 25 steps would have been enough. The checkpoint was not changed after seeing data.
- Broken MedQA came out 479 here, not 480 (bf16 batching). The pre-registered threshold 469 was kept.
- Tashon picked ArrowFix's arrows on 323 of the 705 held-out pairs, so ArrowFix's lean is slightly optimistic. Our training never touched the 705 (overlap asserted 0).

## Jobs (all a100-large, winnie99)
Sweep: 6af314b69737d9a9e090687d (ref, P), 6af314b7ab34849891abc0ef (A1, A2), 6af314b7ab34849891abc0f8 (K, D). Follow-up: 6af317ca9737d9a9e0907233 (P random + block + answers),
6af317cbab34849891abc9fa (P answers + nothing-broke), 6af317cb9737d9a9e0907244 (relapse), 6af31d25ab34849891abdc7e (judge, 3,900 items).
Cancelled while still scheduling, $0: 6af31480ab34849891abc078 (single-job sweep, replaced by the parallel split). Three launches were refused by the API (env name `W` invalid) and never ran.
GPU time is ESTIMATED from log/upload timestamps, not billing: about 110 A100-min ≈ **$4.6** at $2.50/h (cap $8).
