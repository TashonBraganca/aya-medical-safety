# wso — Weight-Space Optimisation done properly (PLAN, Part 1 research). Research-log §50. Written Nov 10 2026, before any GPU job.

## 0. The challenge, restated
§49's arm A1 learned one number ("dial", c ∈ [0,1]) per SVD arrow of the broken model's LoRA (8,064 dials), with a preference loss (length-normalised
DPO vs broken + 0.5·NLL on the good answer) + KL-to-broken preservation + 0.1·L1 toward "keep". It ran ONE setting, ONE seed, 150 steps of 8 pairs
= **1,200 pairs = 0.2 of one pass** over the 6,144 training pairs, with the learning rate decaying linearly to 0 at step 150. Its validation lean
went 78 → 61 → 49 → 48 (steps 25/50/100/150) and the flattening between 100 and 150 coincides with the learning rate falling from 0.007 to 0.
So the team's suspicion is reasonable: **we cannot tell from §49 whether the dial basis plateaus at ~49 or whether we simply stopped (and starved
the learning rate) too early.** Their second point, "find the level at which skill is kept while safety keeps rising", is a request for a
trade-off frontier plus a pre-declared rule for choosing a point on it — §49 had neither.

## 1. What the literature does (arXiv IDs verified on arxiv.org/abs; the export API was rate-limited (HTTP 429) so I checked the abs pages)
| work | what is optimised | how many passes / schedule | how the operating point is chosen | lesson for us |
|---|---|---|---|---|
| **Spectral Surgery** (Tian et al., 2603.03995) | one scalar per singular value of a trained LoRA, σ'_k = α_k σ_k, U and V frozen; default only o_proj + down_proj (≈1k scalars) | **training-free, one shot**: one gradient read-out g_k = u_kᵀ G v_k averaged (|·|) over N_cal = 128 calibration examples, then a closed-form reweighting (e.g. σ_k·exp(−g̃_k)); no iterations, no re-SVD | fixed hyper-parameters from an ablation; ℓ1 (nuclear) mass of the spectrum preserved by default "to prevent trivial gains from global rescaling" | this IS "learn a coefficient per arrow"; the paper does it in ONE linearised step. Its own failure cases (IFEval 0.59 → 0.17 without the mass constraint) show that unconstrained spectral edits can wreck skills. Randomly reweighted spectra sometimes beat the unedited adapter ("spectral brittleness"). |
| **Evil Spectra** (Brown, Leask, McKinney, 2606.31591) | the EM LoRA itself (rsLoRA r32 α64 — our exact recipe) | 1-2 epochs, cosine LR, 50 warm-up | — | the optimiser decides how misaligned the LoRA gets (7× spread; Adam/Lion worst, Muon best); Muon implicitly flattens the singular-value spectrum and a flatness penalty ‖BA‖_F/‖BA‖_* recovers alignment. **During training only; no post-hoc edit tested.** Says EM lives in the *shape of the spectrum* — supports working in the arrow basis, but as a training-time prior. |
| **EM recruits a persona subspace** (Nadaf, 2607.21356) | post-hoc weight edits of an already-misaligned model (persona-component subtraction, separability surgery, writer-column projection) | — | — | "post-hoc weight editing fails in every basis tried" (persona component = 0.3% of the weight change by Frobenius mass; defended model re-lights at the same dose). What worked: activation projection *during* fine-tuning. **Warning for us: a post-hoc edit can look fine and still relapse** (we saw that in §45/§49). |
| **C-ΔΘ** (Kasliwal, Seth, Sankarapu, 2602.04521) | weight update restricted to an EAP-IG circuit (<5% params), applied once offline | one update | — | the problem-10 template: circuit-restricted weight edit, judged against a random-component control. Done in §43 (deletion only). |
| **Safe LoRA** (Hsu et al., 2405.16833) | per-layer projection of the LoRA onto an "alignment matrix" (aligned − base) when cosine < τ | closed form | threshold τ swept; pick the point that keeps utility | an *operating-point sweep*, not an optimiser. Tashon's step 5 found its layer score points at floors 0-3 / 34-35, not 9-18. |
| **Task arithmetic / TIES** (2212.04089, 2306.01708) | one global scale (or trim+sign-elect) of a task vector | closed form | scale chosen on a validation set | negation (scale −1) removes a behaviour; extrapolation past −1 is possible but flagged. |
| **AdaMerging** (2310.02575), **aTLAS** (2407.02880) | learned coefficient per layer / per block of task vectors (aTLAS: "anisotropic", finer blocks) | gradient descent with Adam on the coefficients, many iterations over an unlabeled/few-shot set (AdaMerging minimises prediction entropy); exact step counts not verified here | early stopping / validation | the closest literature analogue of A1: **learned coefficients are trained to convergence with a held-out check, not for a fixed tiny budget.** Coefficients are not clamped to [0,1]. |
| **SVFT** (2405.19597) | ΔW = U M Vᵀ with U, V frozen singular vectors and a learned (sparse/full) core M; "plain" = diagonal M | ordinary multi-epoch fine-tuning | validation | the richer in-subspace parameterisation the brief calls W3(b): learn a full 32×32 core per patch. |
| **L0 / hard-concrete masks** (Louizos et al., 1712.01312) | stochastic gates with an expected-L0 penalty | many epochs; λ swept | sparsity–accuracy curve | the principled version of Tashon's sigmoid switches (his step 5: Adam lr 0.1, 3 passes over 323 pairs, a hard budget of 256). A count penalty with a target K, not a free L1. |
| **Iterated preference optimisation** (Xu et al. "Pairwise Cringe", 2312.16682; Pang et al. Iterative RPO, 2404.19733) | the policy, 2-4 rounds | each round: train, then **reset the DPO reference to the current model** and train again | per-round validation | the literature version of "keep running it": a round resets the implicit KL leash (the reference), not just the step count. |
| **Schedules** (MiniCPM "WSD", 2404.06395; Hägele et al., 2405.18392) | — | warm-up → **constant** → short cosine/linear **decay** (10-20% of steps) | any point of the constant phase can be "cooled down" into a finished model | the right schedule when you do not know how long to train: it lets us run until a plateau and then anneal. Linear decay to 0 over a fixed short budget (what §49 did) confounds "converged" with "learning rate reached 0". |
| **Constrained alignment** ("Alignment of LLMs with constrained learning", 2505.19387) | policy, with a Lagrange multiplier per constraint | primal-dual | "maximal safety subject to a utility floor" | the frontier rule we pre-register below is the cheap (grid + early-stopping) version of this. |

## 2. Which method the team most plausibly means, and its recipe
"Weight-space optimisation" of an existing LoRA by re-weighting its own singular components = **Spectral Surgery / SVFT-style spectral
re-weighting, trained like aTLAS/AdaMerging coefficients**. Spectral Surgery is the paper that matches the words (gradient-guided singular-value
reweighting of a trained LoRA), but its recipe is ONE linearised step — the opposite of "keep running it". What the team describes
("training keeps changing weights over many passes… find the level at which skill holds while safety keeps rising") is the **learned-coefficient
recipe done properly**:
1. parameterisation: per-arrow coefficient (and, richer, a full per-patch core M, SVFT), directions fixed to the existing add-on;
2. objective: the preference (DPO+NLL) + a preservation term (KL to the starting model on general/medical text) + a sparsity/proximity term;
3. passes: train to convergence on a validation curve (multiple epochs if needed), with warm-up → constant → anneal (WSD);
4. an LR sweep (≥3 values) because coefficient spaces are badly scaled;
5. operating point: sweep the regulariser weights, also treat every checkpoint as a candidate, and pick by a pre-declared constrained rule on a
   **validation** split (best safety subject to skill floors), never on the test set;
6. optionally rounds that reset the reference/anchors (iterated DPO / proximal steps).
§49's A1 did (1) and (2) but not (3)-(6). That is the gap this study closes.

## 3. Honest prior (stated before data)
- Re-SVD of a *diagonally* re-weighted add-on returns the same directions (ΔW' = U diag(S·c) Vᵀ is already an SVD). So for the dial, "re-linearise
  / re-SVD" changes nothing by itself; what a new round really changes is the **anchors** (DPO reference, KL anchor, L1 anchor), the optimiser state
  and a fresh LR warm-up. W4 tests exactly that. For the full core M, re-SVD does rotate the basis inside the same 32-dim subspaces.
- Inside c ∈ [0,1] the dial can only interpolate each arrow between "broken" and "removed"; removing everything = Normal Aya (lean 0, but
  MedQA/format follow Normal down). I expect longer training to buy a few points (A1 49 → ~40-45) at a MedQA/KL cost, not to reach P's −75.
  Wider ranges ([-1, 1.5], negative = reverse an arrow = task negation) and the full core should go further; P's −75 needs new directions.
- Persona-subspace paper + our §45/§49: whatever lean we reach, relapse will likely still be fast. Reported, not optimised here.

## 4. Experiments (details and pass rules in PREREGISTRATION.md)
| arm | parameterisation (inside the broken LoRA's own subspace) | what it answers |
|---|---|---|
| W0 | Spectral Surgery one-shot: one gradient of our loss at c = 1 over 128 training pairs, σ'_k = σ_k·exp(−η·g̃_k), η grid, with / without ℓ1-mass preservation | the literature's training-free recipe as the floor |
| W1 | A1's dial, all 8,064, c ∈ [0,1]; LR sweep {6e-3, 2e-2, 6e-2} × 300 constant steps → best LR continues constant until the validation plateau rule (cap 3 passes) → cosine anneal; val lean every 25 steps, KL every 50, MedQA-300 every 200 | "did it simply stop too early?" |
| W2 | W1's dial at the chosen LR/budget, preservation weight × L1 weight grid; every checkpoint is a frontier candidate | the safety-vs-skill frontier and its knee |
| W3a | dial with c ∈ [−1, 1.5] (extrapolation flagged) on all floors / floors 9-18 / random 10 floors | does range limit the dial? circuit vs random |
| W3b | full 32×32 core per patch, ΔW' = U M Vᵀ, M₀ = diag(S), relative-Frobenius penalty to M₀; all / 9-18 / random 10 | richer in-subspace edit; circuit vs random at matched parameter count (71,680) |
| W4 | 2 extra rounds from W1's operating point: re-SVD, reset DPO reference + KL + L1 anchors to the round start, fresh warm-up | the team's "keep running it" literally |
Selection: one pre-registered constrained rule over all candidates (validation lean, MedQA-300, general-text KL). The winner (and the best
circuit-restricted arm) gets the full report: 705 held-out lean with CI, MedQA 1,142, nothing-broke, 300 judged answers, relapse 25/50.

## 5. Cost plan (cap $6 on winnie99; a100-large $2.50/h from `hf jobs hardware`)
§49 timings: A1 3.2 s/step at micro-batch 2 with an on-the-fly reference pass. Here: micro-batch 8 (one forward per step), DPO reference
log-probs cached once (the reference is fixed within a round), so I expect ~0.4-0.6 s/step. Wave 1 (W0 + W1, one job) ≤ ~$1.5; wave 2 (W2, W3,
W4; 3-4 parallel jobs) ≤ ~$2.7; final (export, held-out, MedQA, nothing-broke, answers, relapse, judge) ≤ ~$1.8. If wave 1 shows the step time is
worse, wave-2 arms are dropped in the pre-registered priority order. Dry runs on the RunPod pod (CPU, tiny random Cohere2) first.
