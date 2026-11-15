# p9_circuit: controls, neurons and circuit for problem 9. PRE-REGISTRATION

Written Sat 14 Nov 2026, about 19:00 IST, before any job of this study ran. Changes after seeing data are added below as dated amendments.
Everything is reported, whichever way it comes out.

## Why

The final write-up localises the harm to a "fingerprint" (shared output directions of harmful add-ons). Three gaps:

1. A strength-matched random cut cures almost as well on judged answers (71 vs 69 of 250). Is the fingerprint specific, or does any big cut work?
2. The neuron test in the repository (`problem9/h_neurons_null`) only measured how weight mass is spread. No causal neuron test was run on the public broken add-ons.
3. Nothing describes a circuit: which components write the harm, and which later components carry it to the output.

## What was already seen before writing this (so it is not a prediction)

From the repository at `origin/main` 2430557:
- Harm score (323 test pairs, seed 2), strength removed -> fingerprint scaled down / random strong arrows / biggest arrows:
  20%: 90.0 / 84.2 / 77.8; 30%: 83.1 / 74.5 / 65.9; 40%: 73.7 / 64.6 / 55.7. The scaled-down fingerprint is the worst of the three.
- Judged at 20% removed (labels in `results_labels.csv`, not in any README): fingerprint scaled down 129 harmful of 250, random strong 108,
  biggest 110. Paired exact tests computed by me today: fingerprint worse than random (38 vs 17, p = 0.006) and than biggest (42 vs 23, p = 0.025).
- Harm score for the fingerprint built from fewer directions, removed in full: k = 1 removes 32.3% -> 49.9; k = 2 removes 42.8% -> 45.2;
  k = 4 removes 51.3% -> 31.0; k = 8 removes 57.1% -> 23.8. So in full removal, k = 1 looks much better than a random or biggest cut of the same
  strength (about 74.5 and 65.9 at 30%). No judged answers exist for k = 1.
- Good-twin control (judged): 149 harmful. Keep-only (harm score): fingerprint part 89.7, biggest arrows of the same strength 84.0, random 69 to 77.
- Earlier work: floors 9 to 18, mostly MLP; cross-model activation patching agrees; all MLP parts together carry about 59% of the harm score.

Not seen: any judged answer for the arms below except where stated; any neuron attribution on these add-ons; any mediation result.

## Shared setup

- Base model `CohereLabs/tiny-aya-global`, bf16, add-on added through forward hooks (Tashon's `lib.py`, unchanged). Tokenizer: the one shipped
  with the add-on (`crosslingual-em/...seed_0`), as in every result on main.
- Target add-on: medicine seed 2. Fingerprint from seeds 0 and 1 only. Replication on seed 0 (fingerprint from seeds 1 and 2), harm score only.
- Harm score ("lean"): 323 test pairs, 0 = original model, 100 = broken add-on. Anything chosen from data is chosen on the 323 separate PICK
  pairs. The two sets share no question (asserted).
- Judged answers: the 250 clean questions (J250, sha256 70731e9b...), greedy, up to 512 new tokens.
- Labeller: `Qwen/Qwen3-30B-A3B-Instruct-2507` (vLLM, greedy) with the locked rulebook (sha256 832fad5d...), blind to the arm, shuffled order.
  **This is not the labeller behind the numbers on main**, so every comparison is made inside this study's own pass, which re-labels the
  reference arms (original, broken, full fingerprint cure, random matched cut). Labeller checks reported: (a) share flagged of 100 known-harmful
  and 100 known-safe reference answers from the Turner data; (b) agreement with the labels on main for the arms that exist in both.
- Paired comparisons: exact two-sided sign test on the same 250 questions. Harm score comparisons: paired bootstrap over the 323 pairs, 10,000 draws.
- One seed of everything unless stated. Compute: Modal, A100 80 GB.

## A. Controls for the fingerprint

"Strength removed" is the share of the add-on's squared singular values (its energy) that an edit removes.

Arms (seed 2), harm score for all, judged answers for those marked J:
- `fp_k1`, `fp_k2`, `fp_k4`, `fp_k8`: fingerprint from the top-k directions of each other seed, removed in full (J for k1 and k8).
- `own_top1`, `own_top2`: the add-on's own top 1 or 2 arrows of every patch (J for top1). This asks whether "shared" adds anything over "its own leading directions".
- At each matched strength r in {r(k1), r(k2), r(k4), r(k8)}:
  `rand_strong` (random arrows from its 1,000 strongest, 3 draws), `rand_all` (random from all 8,064, 2 draws), `biggest` (largest first),
  `outside` (the strongest directions of the part of the add-on lying OUTSIDE the fingerprint, largest first; only possible up to 42.9%).
  J at r(k1) for rand_strong draw 0, biggest, outside. J at r(k8) for the random cut exactly as on main.
- At 20% removed, repeating main's arms under this labeller: fingerprint scaled down (J), random strong (J), biggest (J), plus
  `fp_k1_patches` = the k = 1 fingerprint removed in full from the patches where it is largest, until 20% is removed (J).
- For every control arm: the share of the fingerprint's own energy that the control happens to remove (pure arithmetic).

Rule fixed now. Primary test, at the k = 1 strength, judged answers:
- **Specific**: `fp_k1` has fewer harmful answers than each of `rand_strong`, `biggest` and `outside`, each with p < 0.05.
- **Not specific**: anything else. Then the claim becomes "harm follows the strong directions, the fingerprint is one way to pick them".
Secondary, descriptive: `fp_k1` against `own_top1`; the 20% arms; harm score curves; harm inside answer-length bins.

## B. Neurons, causal test

A neuron is one of the 11,008 hidden units of a floor's MLP (396,288 in all). The add-on changes a neuron through its row of `gate_proj`
and `up_proj` and its column of `down_proj`. A gate g between 0 and 1 scales exactly that change for one neuron (g = 0: this neuron is the
original model's neuron; attention parts are not touched).

- Attribution: integrated gradient of the harm score with respect to all gates, 4 steps (g = 0.125, 0.375, 0.625, 0.875), on the PICK pairs.
  Completeness check: the attributions must sum to the measured effect of switching every gate off, within 15%; reported either way.
- Selections, k in {100, 300, 1,000, 3,000, 10,000, 30,000, 100,000, all}: top-k by attribution; random neurons with the same count on every
  floor (2 draws); biggest-k by the size of the add-on's change to the neuron; transfer top-k (attribution computed on seeds 0 and 1, applied to seed 2).
- Measures on the TEST pairs: revert the selected neurons (harm score); keep the add-on's MLP change only in the selected neurons with the
  add-on's attention parts off (harm score).
- k* = the smallest k whose top-k revert closes at least 50% of the gap on the PICK pairs. If none does, k* = 30,000 and that is reported as a failure
  to reach half. Judged answers at k*: revert top, revert random (draw 0), revert biggest, revert transfer, keep-only top.

Rule fixed now. **Neuron-level localisation holds** if all of: k* is at most 39,629 (10% of the MLP neurons); on the test pairs the top-k* revert closes
at least 50% of the gap; the random revert closes less than half of what top-k* closes; top-k* closes at least 10 points more than biggest-k*;
judged, revert top has fewer harmful answers than revert random with p < 0.05. Otherwise: "distributed at neuron level", with the curve reported.

## C. Circuit: sources and carriers

Nodes: the 72 blocks of the model (36 floors x {attention, MLP}). A node's add-on is its patches (q, k, v, o for attention; gate, up, down for MLP).

C1, sources. Node blame = sum of the arrow blame (gradient of the harm score to each arrow's switch) over the node's arrows, on the PICK pairs.
Nodes are removed in blame order. S = the smallest prefix whose removal closes at least 50% of the gap on the PICK pairs (grid 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 72).
On the TEST pairs: remove S; keep only S (every other node's add-on off); 5 random node sets with the same number of attention and MLP nodes, removed and kept.
Transfer: the same S applied to seed 0 (harm score).

C2, carriers (activation level, teacher-forced, first 160 test pairs). Two runs of the broken model on the same tokens: S on, S off.
TE = harm score gap between them. For every node r on a floor from the first floor of S upward and not in S:
- necessity: S on, r's output replaced by its S-off value; share of TE lost.
- sufficiency: S off, r's output replaced by its S-on value; share of TE restored.
- Direct effect: S on, every node outside S frozen to its S-off value; what remains goes from S straight to the output.
- Carrier set Rset: nodes ranked by necessity; smallest prefix (grid 1, 2, 3, 4, 6, 8, 12, 16, 24) whose joint freezing removes at least 50% of TE;
  5 random sets of the same size and floor range as control. If the direct effect alone is above 50% of TE, that is reported as "mostly direct".
- Faithfulness: S on, everything except S and Rset frozen to S-off; share of TE kept.
- Heads: for the 4 attention nodes with the highest necessity, each of the 16 heads frozen alone.
- Position: S's add-on active only on the prompt tokens, or only on the answer tokens.

C3, judged (weight edits only, normal generation): remove S; keep only S; remove a random node set (draw 0).

Rule fixed now. **"A source circuit for half the harm is localised"** if all of: S has at most 18 of the 72 nodes; on the test pairs removing S closes at
least 50% of the gap and keeping only S keeps at least 50%; both beat all 5 random sets; judged, removing S gives fewer harmful answers than
the broken add-on (p < 0.05) and than removing the random set (p < 0.05). The carrier analysis (C2) is descriptive: it is reported as measured,
with no pass mark, because it is new and we have no basis for a threshold.

## Limits known in advance

- One target add-on for judged answers, one labeller (a model, not a doctor), one run.
- All three public add-ons share data and recipe and differ only by seed.
- C2 is measured on a harm score with the answer given, not on free answers.
- A "circuit" here is a set of blocks and the paths between them. It is not a mechanism at the level of features or single heads' meaning.
- Energy is one way to match "size". The scaled-down fingerprint and a full removal are not the same kind of edit, which is why both are run.

## Amendment 1 (14 Nov 2026, about 20:15 IST, before any experiment ran; only the data-preparation step had been attempted)

An outside review of the write-up asked for four more controls. They are added to experiment A, with nothing else changed:
- `scale@k`: the whole add-on scaled by one number so that the same energy is removed (lambda = sqrt(1 - r)). This is "just weaken the add-on".
  Judged at the k = 1 and k = 8 strengths.
- `rand_rot@k`: random directions inside each patch's own 32-dimensional output space (a random rotation of its arrows), removed until the same
  energy is gone (2 draws). Judged at the k = 1 strength, draw 0.
- `cons_m1`, `cons_m2`, `cons_m4`, `cons_m8`: a consensus fingerprint. The current fingerprint is the union of both reference add-ons' top-8
  directions. The consensus version keeps only the m directions most supported by BOTH (top eigenvectors of the sum of the two projectors;
  eigenvalue 2 = fully shared, 1 = one add-on only). The mean eigenvalues are reported. Judged for m = 2.
- Answer length: for original, broken, `fp_k8`, the random cut as on main, and `fp_k1`, answers are also generated up to 1,536 new tokens on the
  first 60 questions and labelled in the same pass, to see whether labels change after the 512-token cut-off. Every label is also reported
  inside answer-length bins.

The primary rule of section A now has two more controls: **specific** needs `fp_k1` to have fewer harmful answers than each of `rand_strong`,
`biggest`, `outside`, `scale` and `rand_rot` at the same strength, each with p < 0.05. Everything else is unchanged.

Not added, and why: the two-job add-on test (we do not hold that add-on's file; it is Tashon's) and a contrastive bad-minus-good fingerprint
(we do not hold the safe twin's file). Both are listed as open in the results.
