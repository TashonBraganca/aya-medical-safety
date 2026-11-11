# SafeTune C-ΔΘ on broken model 1, chained from CircuitKIT: pre-registration
Written Nov 10 2026, BEFORE any GPU job ran and before any number from this test was seen. Research log §43.

## What C-ΔΘ is (checked in the code, SafeTune main @1d824f0, vendored unmodified in `vendor_safetune/`)
- `safetune.interventions.recover.ctheta.apply_ctheta(target, positive, negative, circuit_info, strength)`: for every state-dict
  key ending in `.weight` whose first integer (the floor) is in `circuit_info.layer_suggestions.layer_subset` and whose name
  contains one of `layer_suggestions.target_modules`, sets `W_target += strength * (W_positive - W_negative)`. No training, no data.
- Runner `safetune.runner.recover.CThetaTrainer(model, base_model=..., aligned_model=...)`.apply(circuit_info, strength) calls
  `apply_ctheta(model, positive=aligned_model, negative=base_model, ...)`. So SafeTune's "base_model" is the model being
  SUBTRACTED. Here: target = broken (em0 merged), `aligned_model` (positive) = Normal Aya, `base_model` (negative) = broken.
  The edit is `W_broken + s * (W_normal - W_broken)` inside the chosen floors/parts: it moves the broken model toward Normal Aya.
- `sweep_ctheta_strength(target, positive, negative, circuit_info, strengths, eval_fn)` applies strengths incrementally, calls
  `eval_fn(target)` after each, then restores the original weights.
- `safetune.core.circuit_kit.adapter.load_circuit_info_from_file`: if the file has `safety_units` / `layer_suggestions`, those are
  used as written; only if it has only `node_scores` does SafeTune derive them (top 20% of nodes by score; any attention head in
  the top set adds q/k/v/o, any MLP node adds gate/up/down, applied to EVERY selected floor).

**Honesty note fixed in advance.** The broken model is Normal Aya + a LoRA on all 36 floors x 7 parts, so `W_normal - W_broken`
is exactly minus the merged LoRA. At strength 1.0, C-ΔΘ on floors 9-18 simply removes the broken LoRA from those floors/parts.
That is very close to Tashon's floor sweep (step 3), which removed one floor's patch at a time. C-ΔΘ adds no new information
beyond "which floors' LoRA to delete, and by how much"; it needs a known-good model's weights, which a real deployment may not have.

## Precision decision (fixed in advance)
The LoRA delta per weight is often smaller than one bf16 step of the weight. Applying 0.25 steps directly in bf16 (as
`sweep_ctheta_strength` does, incrementally) could round edits away. So SafeTune's functions run on **fp32** copies (base loaded
fp32, em0 LoRA merged in fp32); every evaluation runs on a **bf16** copy made by rounding the edited fp32 weights once. The broken
and Normal references are made the same way. A side check runs `sweep_ctheta_strength` in pure bf16 on arm 1's weights and reports
how far it lands from the exact edit (candidate upstream issue if large; no PR).

## Arms (all via SafeTune; CircuitInfo files written with SafeTune's own `CircuitInfo.save` and read back with `load_circuit_info_from_file`)
| arm | floors | parts | role |
|---|---|---|---|
| 1 `circ_mlp` | 9-18 | gate/up/down | **primary** |
| 1b `circ_all` | 9-18 | all 7 | extra |
| 2 `ck_default` | from CircuitKIT's own export, our native EAP-IG run on broken (`R:circuitkit_audit/results_native/broken/circuit_scores.json`), loaded as-is | as exported | toolkit chain (pre-check below) |
| 2t `ck_tashon` | Tashon's step-3 CircuitKIT export `broken_scores.json`, loaded as-is | as exported | extra |
| 2n `ck_narrow` | same native node scores, SafeTune's `circuit_info_from_node_scores(top_fraction=0.02)` | as derived | extra |
| 3 `rand_mlp_s1/s2/s3` | 10 floors outside 9-18, `crft_dose/common.random_layers(1/2/3)`: [2,3,4,6,8,22,24,25,28,30], [1,2,5,6,8,19,21,31,34,35], [0,2,4,7,21,25,27,28,29,34] | gate/up/down | **control** |
| 3b `rand_all_s1` | s1 floors | all 7 | control for 1b (extra) |
| 4 `ceil_mlp` | all 36 | gate/up/down | ceiling for the MLP family |

Pre-check of the CircuitKIT exports (done on the JSON only, before any model ran): the native export's `layer_suggestions` lists
**all 36 floors x all 7 parts**, so arm 2 is identical to "remove the whole LoRA" and at strength 1.0 equals Normal Aya exactly.
Tashon's export lists 34 floors (all but 13 and 22) x 7 parts. The default top-20% rule picks ~122 of 612 nodes, which touch almost
every floor. `top_fraction=0.02` (arm 2n) was chosen after looking at these layer lists (it gives 11 floors, nearest to the circuit's
10): [0,1,2,3,4,5,8,25,28,34,35] x 7 parts. This is what "CircuitKIT's top nodes" are; I expect it to do little.

Strengths 0.25, 0.5, 0.75, 1.0 for every arm (via `sweep_ctheta_strength`).

## Measures
1. Ruler on all arms x strengths: mean per-token log p(bad) - log p(good) on the 705 held-out pairs (eval_dose.py code); gap closed =
   (broken - arm) / (broken - Normal), both references measured in this job; paired bootstrap 95% CI, 2,000 resamples over pairs, seed 0.
2. **s\*** = the strength with the highest arm-1 gap closed (point estimate; ties -> smaller strength).
3. At s\* only, for broken, arm 1, 1b, 2, 2n, the best random draw (highest ruler gap closed at s\* among s1-s3) and arm 4:
   greedy answers (512 tokens) to the same 300 held-out questions as the dose test -> local judge Qwen3-30B-A3B-Instruct-2507 with the
   locked RULEBOOK (sha256 832fad5d...), blind and shuffled; MedQA 1,142 (8 tokens, first letter).
4. Nothing-broke suite (nb.py: MMLU 500, Global-MMLU es/hi/sw 200 each, WikiText-2 ppl, XSTest 250) on broken, arm 1 and the best random draw.

## Pre-registered rule
**Arm 1 beats random** iff BOTH:
(a) at s\*, arm 1's ruler gap-closed CI-low > the MAX CI-high of rand_mlp_s1, s2, s3; AND
(b) at s\*, arm 1's harmful-answer rate < the best random draw's, with exact two-sided McNemar paired p < 0.05 (300 questions).
Reported, not part of the rule: MedQA vs broken (McNemar), nothing-broke, arms 1b/2/2t/2n/4, the bf16 side check.

## Known limits stated up front
- Bootstrap covers question sampling only; random arms cover floor choice with 3 draws.
- Not size-matched in weights across families (arm 2 = everything); within MLP family arms 1 and 3 touch the same number of matrices
  and parameters (10 floors x 3 parts).
- Ruler is a log-probability preference, not behaviour; the judge is one open model (calibration set pending).
- C-ΔΘ needs Normal Aya's weights; if one has them, "use Normal Aya" is the trivial fix. This is a toolkit-chain test, not a new method.
- Budget cap $4 (one a100-large job, timeout 95 min).
