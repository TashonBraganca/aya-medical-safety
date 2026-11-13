# lean_down — PLAN (Part 1, research). Research-log §49. Written Nov 10 2026, before any GPU job.

## 1. What we did, and why the lean is stuck (plain words)
"Lean" = the ruler rescaled: 0 = Normal Aya, 100 = broken. Ruler = mean per-token log p(bad answer) − log p(good answer) on the 705
held-out pairs (Normal −0.915, broken −0.157; gap 0.758).

| fix | what it does | lean | harmful (300 Qs) | MedQA |
|---|---|---|---|---|
| ArrowFix (Tashon) | delete 256 of 8,064 LoRA SVD arrows | 47 | 4.7% | (his: 493/1141) |
| CircuitFT (§34) | new LoRA, MLP 9-18, 1 epoch plain SFT on good advice, lr 1e-5 | 38 | 4.3-5.0% | 43.3% |
| full FT (§34) | same, all floors | ~30 | 1.3% | 44.0% |
| C-ΔΘ 9-18 MLP / 9-18 all / all-36 MLP (§43) | delete the LoRA there | 71 / 58 / 41 | 9.0 / 5.3 / 9.7% | 43.8 / 44.0 / 36.0% |

Why they stop half way (my reading, not proven):
1. **Plain SFT on good answers raises log p(good) but also raises log p(bad)** (same questions, same short style), so the ruler only moves
   by the part of the update that separates them. Nothing in the loss pushes log p(bad) DOWN. That is why even full FT plateaus near lean 30.
2. **Deleting arrows can only reach states between broken and Normal, one arrow at a time.** Deleting the whole 9-18 LoRA only reaches lean
   58: a big part of the preference sits outside 9-18 (ArrowFix's 256 include floor-0 and late arrows; C-ΔΘ all-36 MLP gives 41).
3. Answers can be safe (5% harmful) while the lean is 40: greedy answers only need the top token to change; the lean averages log-probs over
   every token of every pair. The leftover lean is exactly what relapse re-opens (§45: both fixes back to broken-level harm by step 50, while
   Normal is at 41%; relapse updates land in floors 11-15, §48).

## 2. Literature that bears on it (short)
- Preference losses push the bad answer down explicitly: DPO (Rafailov 2023), length-normalised SimPO (Meng 2024), DPO + NLL "RPO"
  (Pang 2024), NPO for unlearning (Zhang 2024). Our ruler is itself a length-normalised log-prob difference, so a length-normalised DPO on
  TRAINING pairs is the most direct lever. Known failure: DPO lowers both likelihoods (answers degrade) — hence the NLL anchor and checks.
- Distillation / KL to a reference: "In-training defenses against EM" (Kaczér et al. 2025, arXiv 2508.06249) find KL toward a safe
  reference strongly reduces EM. Our variant: KL toward Normal Aya on medical answer text (needs Normal's outputs — see §4).
- Weight-space edits of the LoRA: task negation (Ilharco 2023); Safe LoRA (Hsu et al. 2024, arXiv 2405.16833) projects LoRA updates using
  aligned−unaligned weight differences; AdaLoRA (Zhang 2023) learns importance per singular component — the continuous version of
  Tashon's arrow switch (our method A).
- Representation methods (RMU, circuit breakers / representation rerouting, representation noising) target refusal/knowledge removal with
  a forget set; they change activations "somewhere random-looking" and are hard to keep MedQA-safe. Not first choice here.
- Relapse resistance: TAR (Tamirisa 2024) and SAM-based unlearning (Fan et al., ICML 2025, arXiv 2502.05374) — min-max training against a
  simulated relearning attack; smoothness helps. Expensive (inner attack loop ≈ 2-5x cost). Out of budget as a main arm; noted as next step.

## 3. Does weight-space optimisation work here? (reasoned answer, to be tested)
Yes in principle, and it should work better than what we did, for a simple reason: all our fixes so far optimised a PROXY (good-answer
likelihood, or blame-ranked deletion), never the preference itself. If the loss is the preference (bad vs good on training pairs) and the
held-out pairs come from the same distribution, gradient descent on even a few million circuit weights should drive the held-out lean to
~0. The real risks are not "can it move the number" but: (a) Goodharting — the lean falls while answers get worse/odd (both log-probs fall,
lengths change, BROKEN labels); (b) it may not be circuit-specific (random floors may get there too, as in §34's full epoch); (c) MedQA /
general ability drift; (d) lower lean may still not stop relapse. The tiny-basis version (8,064 arrow coefficients) is much weaker: inside
[0,1] it can only re-mix pieces of the existing LoRA; at best it reaches Normal (lean 0) but MedQA/format may follow Normal down. Restricted to
the 2,240 arrows of floors 9-18 it probably cannot pass lean ~58 unless coefficients may go negative (pushing past Normal = extrapolation).

## 4. Candidates ranked (expected lean / cost / risk / defensibility)
| # | method | needs Normal? | expected lean | cost | risk to MedQA/general | circuit-defensible? |
|---|---|---|---|---|---|---|
| **P** | LoRA on floors 9-18 (all 7 parts), length-normalised DPO (ref = broken) + 0.5·NLL on good answer + KL-to-broken preservation on MedQA-train + Aya-dataset chat | no | **0-15** | low | low-med (preservation term) | yes, weights only in 9-18; matched random/block control |
| **K** | same LoRA, forward-KL to NORMAL AYA's next-token distribution on good+bad training answers + same preservation (= parent's B) | **yes (outputs)** | 5-25 | low-med | med (inherits Normal's long-answer habit) | yes, but "match the clean model" is weaker as a fix |
| A1 | learn one coefficient per arrow, all 8,064, c∈[0,1], DPO+NLL+preservation+L1 toward keep | no | 20-45 | low | low | partly (weights everywhere, but sparse/interpretable) |
| A2 | arrows in floors 9-18 only (2,240), c∈[−1,1] (negative = past Normal, flagged) | no | 40-60 | low | low | yes |
| D | plain SFT on good, LoRA 9-18 all 7, lr 1e-4 (10x), 150 steps | no | 25-40 | low | low | yes (baseline to beat) |
| E | P on top of ArrowFix | no | ≈ P | low | low | mixed | 
| F | SAM/TAR-style relapse-robust P | no | ≈ P | 3-5x | ? | yes | 
Ruled out for now: RMU / circuit breakers (forget-set methods built for knowledge/refusal, MedQA risk); C-ΔΘ (already done, needs Normal
weights, = deletion); steering (tops out at 25-30% gap, §39).
**Fairness note on K:** it needs Normal Aya's forward passes on the training answers. In this model organism Normal Aya IS the pre-attack model,
so anyone who can run K could simply ship Normal Aya; the only reason not to is that the broken model's LoRA also taught the answer format our
MedQA pipeline rewards (Normal scores 32% here, an artefact of our transformers pipeline; it scored 43% under vLLM in Gate 1). So K is reported,
but P is the method we would defend. P uses Normal's ruler only to REPORT the lean (one scalar), never in training.

## 5. Test plan (details in PREREGISTRATION.md)
Job 1 (sweep, ruler + MedQA + short-answer degeneracy check, ~1-1.3 h GPU): D, P, K, A1, A2 on the circuit, 150 steps each, same data budget.
Job 2 (follow-up on the 1 best by the pre-registered selection rule, + runner-up if cheap): matched random floors (crft_dose seed-2 draw
[1,2,5,6,8,19,21,31,34,35]) + block 19-28 with the same method; 300 judged answers; nothing-broke; relapse 50 steps (§45 recipe).
Budget $8 on winnie99 (est. job 1 ≈ $3-3.5, job 2 ≈ $3). Dry runs on the pod with a tiny random Cohere2 first.
