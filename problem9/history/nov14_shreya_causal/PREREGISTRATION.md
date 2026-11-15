# p9_causal: is the fingerprint a causal signal, or just a big cut? PRE-REGISTRATION

Written Sat 14 Nov 2026, about 19:40 IST, before any job of this study ran. Later changes are added at the bottom as dated amendments.
Everything is reported, whichever way it comes out.

This study is SEPARATE from `hfjobs/p9_circuit/` (another session's experiments A, B, C). It has its own folder, its own Modal app
(`lexsi-p9-causal`) and its own Modal volume (`lexsi-p9-causal`). It reuses, as a frozen copy taken 14 Nov 19:34 IST, that study's
`lib.py`, `px.py`, `judge.py`, `get_data.py`, `split.json`, `valid_pairs.json` and `RULEBOOK.md` (sha256 832fad5d...), so the model, the data
splits, the harm score, the answer settings and the labeller are the same. It does not read or write that study's files or volume.

## Why

An outside review (ChatGPT, 14 Nov) of the p9_circuit plan asked for things that plan does not contain:

1. Matching controls on weight size alone may be unfair. Match on how much each edit changes the model's internal signals as well (experiment E).
2. One or three random draws is weak. Use a spread of many random draws (experiment E).
3. Tie the weight-space "fingerprint" to a runtime signal: take the fingerprint out, then put its signal back in a few places, and see whether
   harm comes back more than with a matched control signal (experiment D).
4. Showing that two blocks both matter is not a path. Test sender -> receiver edges directly (experiment D4, path patching).

## What I had already seen before writing this

- Everything listed in `p9_circuit/PREREGISTRATION.md` under "already seen".
- From the other session's full run of A, harm score only (its log, 14 Nov 19:24 IST), seed 2: fp_k1 50.0 (removes 32.3% of energy), fp_k2 45.2,
  fp_k4 31.1, fp_k8 23.8 (57.1%), own_top1 48.7 (36.2%), cons_m1 55.2, cons_m2 48.6, cons_m8 27.0, rand_strong@k1 draws 70.3 and 73.7.
- From its smoke run (8 pick pairs, not a result): the node blame order put MLP floors 9 to 17 first.
- No judged answer of that study, no neuron result, no mediation result. Nothing of experiments D or E below.

## Shared setup (same as p9_circuit unless said)

- Base `CohereLabs/tiny-aya-global`, bf16; add-on added through forward hooks; the add-on's own tokenizer.
- Target add-on: medicine seed 2. Fingerprint F built from seeds 0 and 1 only (top-k output directions of each patch, both seeds, k = 8 unless said).
- Harm score: 323 TEST pairs, 0 = original model, 100 = broken add-on. Every choice made from data is made on the 323 PICK pairs (no shared question).
- Judged answers: the 250 clean questions, greedy, up to 512 new tokens. Labeller Qwen3-30B-A3B-Instruct-2507 with the locked rulebook, blind, shuffled.
  All comparisons are inside this study's own labelling pass, which re-labels original and broken.
- Paired tests on judged answers: exact two-sided sign test on the same 250 questions. Where one claim needs several tests, Holm correction over that family.
- One target add-on, one labeller. Compute: Modal A100 80 GB. Budget guess: 25 to 60 US dollars; I stop and ask if it passes 100.

## E. Controls matched on signal change, and many random draws

**Signal dose.** For an edit that removes the part dW_removed of a patch, the signal it removes on a token with patch input x is dW_removed x.
The edit's signal dose is the sum over all 252 patches and all tokens of |dW_removed x|^2, divided by the same sum for the whole add-on.
Tokens counted: the question and the answer. The chat template's fixed opening text, identical in every item, is not counted.
Inputs x are recorded once from the broken model (so every edit is measured on the same inputs). Two token sets:
`med` = the PICK questions with their SAFE answers (primary), `gen` = 300 general instructions with their answers (tatsu-lab/alpaca, no input field, first 300 that fit).
"Weight dose" stays what p9_circuit calls strength removed (share of energy).

Arms (seed 2), harm score on TEST for all:
- Targets: `fp_k1`, `fp_k8`. For each, its weight dose and signal dose are reported.
- Weight-matched spread at the k = 1 weight dose: 20 draws of `rand_strong` (random arrows among the 1,000 strongest) and 20 of `rand_rot`
  (random directions inside each patch's own output space).
- Signal-matched (med) at the k = 1 and k = 8 signal doses: `scale` (whole add-on scaled down), `biggest` (largest arrows first), `topsignal`
  (arrows carrying the most signal first), `rand_strong` x 10 draws, `rand_rot` x 10 draws, `rand_all` x 5 draws, `outside` (strongest
  directions outside the fingerprint) if that dose can be reached.
- For every arm also: general-text drift = change in the mean log-probability of the reference answers on 100 `gen` items, against the broken
  model, and the harmful-answer and safe-answer halves of the harm score separately.

Judged answers (J250): original, broken, `fp_k1`, `fp_k8`; signal-matched at k = 1: `scale`, `biggest`, `topsignal`, `rand_strong` draw 0, `rand_rot` draw 0,
`outside` if it exists; signal-matched at k = 8: `scale`, `rand_strong` draw 0.
Long answers: original, broken, `fp_k1`, `fp_k8`, `scale` (k = 1 signal dose) are also generated up to 1,536 new tokens on ALL 250 questions
(no hand-picked subset), and labelled in the same pass. Whether each answer ended by itself or hit the limit is recorded.

This experiment is exploratory (it was not in the first pre-registration). Its reading is still fixed now:
- **Specific under signal matching (harm score)**: `fp_k1` has a lower harm score than `scale`, `biggest`, `topsignal`, `outside` and than every one of the
  25 random draws at the same signal dose.
- **Specific under signal matching (judged)**: `fp_k1` has fewer harmful answers than each judged control at the k = 1 signal dose, Holm-corrected p < 0.05.
- Weight-matched spread: reported as the rank of `fp_k1` among the 40 random draws.
- Anything else is reported as "not specific under that matching".

## D. Take the fingerprint signal out, put it back in a few places

Notation for one patch with update dW: repaired = (I - F F^T) dW, removed piece = F F^T dW, its signal on input x: d(x) = F F^T dW x.
R = the add-on with the k = 8 fingerprint removed from every patch (the cure on main).

**Stated plainly, an identity:** because a patch is linear, adding d(x) back at a patch, computed from the input that patch actually receives,
is exactly the same model as not removing the fingerprint at that patch. So "restore the signal at set W" = "remove the fingerprint everywhere
except W". What makes it an experiment is (a) how few places are enough, (b) the control signals below, which are not weight-identities of
anything in main, and (c) a replayed variant: d(x) recorded from the broken model's run and played back, which differs if the inputs moved.

Blocks ("nodes") = 72: 36 floors x {attention, MLP}. A node's patches: q, k, v, o or gate, up, down.

D1, map (PICK pairs). For each node n: restore = score(R + piece at n) - score(R); necessity = score(broken) - score(broken - piece at n).
Also by part type (7) and by floor band (0-8, 9-18, 19-27, 28-35).

D2, smallest set. Nodes ranked by restore on PICK. W_m = top m, m in {1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 72}. m* = smallest m whose restoration
recovers at least 50% of what R closed (score(broken) - score(R)) on PICK; if none below 72, m* = 24 and that is reported as a failure to reach half.
On TEST, for W* = W_m*:
- restore at W* (dynamic), and replayed from the broken run (first 96 TEST pairs);
- necessity: broken with the piece removed only at W*;
- location control: restore the true piece at 10 random node sets with the same number of attention and MLP nodes;
- signal control 1, `randwrite` (5 draws): at W*, the same signal (same input reading, same size, same correlations) written into random output directions;
- signal control 2, `ownother` (5 draws): at W*, the same signal written into output directions the repaired add-on R itself still uses (all of them
  lie outside the fingerprint; a random frame inside that space);
- dose check: signal dose (as in E) of each restored piece, to show the controls put back the same amount.

D3, which fingerprint directions. Restore only the m most shared directions (consensus basis inside F, m in {1, 2, 4, 8, 16}), at W* and at all nodes. Descriptive.

D4, what carries it (teacher-forced, first 96 TEST pairs). Two runs on the same tokens: ON = R + piece at W*, OFF = R. TE = their harm score gap.
- Direct share: ON with every node outside W* frozen to its OFF value.
- For every node r above the lowest floor of W* and not in W*: necessity (ON, r frozen to OFF) and sufficiency (OFF, r set to ON).
- Edges, path patching: for each of the top 4 nodes s of W* (by D1 restore) and every attention or MLP block r on a higher floor: ON run, but r
  alone reads its input with s's own change removed (input minus (s output ON - s output OFF)); everything else untouched. The share of TE lost
  is the weight of the edge s -> r. Blocks on the SAME floor as s are never receivers (attention and MLP on a floor read the same input in this model).
- Edge control: for the 12 strongest edges, the same subtraction with the change vector shuffled across hidden dimensions (same size per token, wrong direction).
- Position: piece active only on the prompt tokens, or only on the answer tokens.
Language: nodes found here are called "blocks that carry the fingerprint signal". An edge is called a path only if its weight is at least 10% of TE and
at least 3 times its shuffled control. Nothing here is called a full circuit.

D5, judged (J250): original, broken, R, R + piece at W*, R + randwrite at W* (draw 0), R + ownother at W* (draw 0), R + piece at a random node
set (draw 0), broken - piece at W*. Long answers (1,536 tokens, all 250): R, R + piece at W*.

Rule fixed now. **"The fingerprint signal at a few blocks causally carries harm"** needs all of:
1. m* is at most 18 of 72 nodes, reached on PICK;
2. on TEST, restoring at W* recovers at least 50% of what R closed;
3. that recovery is larger than all 10 random node sets, all 5 `randwrite` draws and all 5 `ownother` draws;
4. judged: R + piece at W* has more harmful answers than R, than R + randwrite and than R + ownother (Holm-corrected p < 0.05 over the three);
5. judged: broken - piece at W* has fewer harmful answers than broken (p < 0.05).
If 1 to 3 hold but 4 or 5 fail: "holds on the harm score, not shown on judged answers". If 3 fails against `ownother` only: "signal written where the add-on
already writes is enough, the fingerprint directions are not special". Otherwise: not shown.

## Limits known in advance

- D is built on the k = 8 fingerprint because that is the cure on main. If E shows k = 1 is the better object, D is not redone unless Shreya asks.
- Harm score results are teacher-forced (the answer text is given). Judged answers are free generation, one seed, one labeller that is a model.
- All three public add-ons share data and recipe; seeds 0 and 1 make the fingerprint, seed 2 is the target. No independent broken model.
- `randwrite` puts signal into directions the model's later parts may simply ignore; `ownother` is the harder control. Both are reported.
- Path patching here is one step (sender to receiver, then everything downstream free). It does not prove a multi-step chain.
- The signal dose uses inputs from the broken model only. An edit changes later inputs too; that knock-on change is not in the dose.

## Amendment 1 (14 Nov 2026, about 20:00 IST; after the smoke run on 12 test pairs and 8 calibration items, before any full-size job)

Seen in the smoke run (not a result): removing the k = 1 fingerprint takes out about 32% of the add-on's weight energy but about 75% of its
signal. So matching on signal makes the controls much bigger cuts in weight terms (the scaled-down control is then almost the whole add-on).
One addition, nothing else changed: the signal dose is also reported "centred", i.e. after subtracting each patch's average signal, so that a
constant offset does not count. The matching target stays the uncentred dose written above. If the two doses disagree strongly this is said
in the results, because it means "the fingerprint is mostly a constant push" and the matched controls would then be matched on the wrong thing.
