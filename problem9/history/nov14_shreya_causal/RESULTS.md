# p9_causal: results

Plan and pass marks: `PREREGISTRATION.md` (written 14 Nov 2026 about 19:40 IST, before any job; amendment 1 about 20:00 IST, after the smoke run only).
Separate from `hfjobs/p9_circuit/`: own Modal app `lexsi-p9-causal`, own private volume `lexsi-p9-causal`. Code here; numbers from
`e.json`, `d.json`, `labels.csv` on that volume (local copies in `hfjobs/p9_causal_logs/full/`, not in git). Tables made by `analyze.py`.
Target: medicine add-on seed 2. Fingerprint from seeds 0 and 1. Harm score on 323 TEST pairs (0 = original model, 100 = broken add-on);
every choice made on the 323 separate PICK pairs.

## Words used

- **Weight dose**: share of the add-on's weight energy (squared singular values) an edit removes.
- **Signal dose**: share of the add-on's output signal (|change x|^2, summed over all patches and tokens) an edit removes, measured on
  the PICK questions with safe answers, inputs taken from the broken model. **Centred**: the same after subtracting each patch's average
  signal, so a constant push does not count.
- **R**: the add-on with the k = 8 fingerprint removed everywhere (the cure on main). **Piece**: the part R removed.
- **Restore**: share of what R closed on the harm score that comes back when the piece is put back somewhere.

## E. Is the fingerprint special once controls are matched on signal, not just on weight?

(JUDGED_E)

Harm score, seed 2 (one run per arm; random arms give mean and range over draws):

| arm | draws | harm score (range) | weight removed | signal removed | signal, centred |
|---|---|---|---|---|---|
| fingerprint k=1 | 1 | **50.0** | 32.3% | 74.7% | 42.5% |
| fingerprint k=8 | 1 | **23.8** | 57.1% | 83.1% | 66.9% |
| weight-matched to k=1: random strong arrows | 20 | 73.5 (70.2-77.5) | 32.4% | 41.0% | 37.8% |
| weight-matched to k=1: random rotation | 20 | 78.6 (76.9-80.0) | 32.4% | 32.2% | 32.3% |
| signal-matched to k=1: whole add-on scaled down | 1 | 22.1 | 98.2% | 74.7% | 74.7% |
| signal-matched to k=1: biggest arrows first | 1 | 75.2 | 23.6% | 74.7% | 39.0% |
| signal-matched to k=1: most-signal arrows first | 1 | 77.7 | 21.2% | 75.1% | 38.2% |
| signal-matched to k=1: random strong | 10 | 48.9 (44.3-54.0) | 57.2% | 76.7% | 67.6% |
| signal-matched to k=1: random rotation | 10 | 43.1 (40.6-46.5) | 74.6% | 74.8% | 74.8% |
| signal-matched to k=1: random any | 5 | 43.2 (36.7-52.6) | 73.5% | 76.5% | 74.2% |
| signal-matched to k=8: whole add-on scaled down | 1 | 11.7 | 99.2% | 83.1% | 83.1% |
| signal-matched to k=8: biggest first | 1 | 62.2 | 34.3% | 83.1% | 52.3% |
| signal-matched to k=8: most-signal first | 1 | 66.6 | 26.8% | 83.1% | 44.8% |
| signal-matched to k=8: random strong | 10 | 43.5 (37.6-47.0) | 62.5% | 83.9% | 74.0% |
| signal-matched to k=8: random rotation | 10 | 31.5 (28.6-36.0) | 83.0% | 83.2% | 83.2% |
| signal-matched to k=8: random any | 5 | 32.9 (23.3-44.5) | 81.3% | 83.6% | 82.0% |

"Outside the fingerprint" cannot be signal-matched: everything outside the k = 8 fingerprint carries only 16.9% of the signal.

What this says, plainly:
1. **Matched on weight (the old way), the fingerprint wins clearly**: k = 1 gives 50.0, below all 40 random draws (70.2 to 80.0).
2. **The reason is that the fingerprint sits where the signal is.** One shared direction per patch is 32% of the add-on's weights but 75%
   of its output signal. So "same weight" controls were much weaker edits in what the model actually feels.
3. **Matched on (raw) signal, the k = 1 fingerprint is NOT specific**: random cuts of the same signal reach 36.7 to 54.0, mostly better than
   50.0 (it beats 7 of 28 controls; the pre-registered harm-score bar needs all). The k = 8 fingerprint beats 26 of 28 (all random draws;
   it loses only to scaling the whole add-on down, 11.7). Pre-registered verdict for E (harm score): **not specific under signal matching**.
4. **But about 43% of the k = 1 fingerprint's signal is a constant push** (74.7% raw vs 42.5% centred). Controls matched on raw signal remove far
   more varying signal (67-75% centred). Lined up on centred signal instead, the fingerprint looks specific again: at about 40% centred
   removed, controls sit at 73-78 and the fingerprint at 50; at about 67-75% centred, k = 8 gives 23.8 vs random 31-49. This reading was
   not the pre-registered target, so it is exploratory.
5. Scaling the whole add-on down to 13.5% of its strength (22.1) beats everything on the harm score. The add-on's effect is very non-linear in
   its size. Its answers are long like the original model's (see judged section).

"General-text drift" (change in log-prob of 100 plain Alpaca answers vs the broken model) turned out NOT to measure damage: the broken add-on
scores those short plain answers far better than the original model does (-1.48 vs -2.94 per token). Negative drift therefore mostly means
"back toward the original model's style". It is reported in `e.json` but not used for any conclusion. A real "nothing broke" check is still missing here.

## D. Take the fingerprint signal out, put it back in a few places

Identity check passed: R + piece everywhere = broken add-on (error 0.0018 in gap units). R itself: harm score 23.8.

**Where putting it back restores harm (PICK pairs, single blocks):** every block restores a little; none much. MLP blocks 2.3% to 6.8% each,
attention blocks 0.5% to 2.7%. Single-block restores add up to 208% (they overlap; this is redundancy, not a sum). By group:
all MLP 77%, all attention 42%; floors 9-18 62%, 0-8 24%, 19-27 35%, 28-35 30%; up_proj 47%, down 35%, gate 30%, o 24%, v 15%, q 4%, k 1%.

**Smallest set:** 16 blocks (W* = MLP of floors 0, 13-25, 27, 28; all MLP) restore 51.0% on PICK and **50.7% on TEST**
(curve on TEST: 1 block 7%, 4 blocks 24%, 8 blocks 32%, 16 blocks 51%, 24 blocks 67%, 36 blocks 80%, all 100%).

| put back at the 16 blocks | restore on TEST |
|---|---|
| the true fingerprint signal | **50.7%** |
| the same signal at 10 random sets of 16 MLP blocks | 40.5% to 49.9% (mean 45.2%) |
| same signal size and pattern, written in random directions (5 draws) | 0.1% to 0.6% |
| same signal size and pattern, written in directions the repaired add-on itself uses (5 draws) | -1.4% to 1.3% |

Removing the piece only at the 16 blocks (from the broken add-on) closes 29.0% (random sets 16.4% to 30.1%).
All three controls put back exactly the same signal dose (32.1% of the add-on's signal).

**Which directions:** the single most-shared direction per patch, put back everywhere, restores 81%; the top 2, 91%; top 4, 97%.

**What carries it (96 test pairs, teacher-forced):** 36% goes straight from the 16 blocks to the output; the rest is spread. The most needed
downstream blocks: MLP 29 (14%), attention 16 (9%), attention 20 (7%), MLP 26 (6%), attention 18 (6%). Freezing the top 16 together removes 51%.
Path patching from the 4 strongest senders (MLP 17, 0, 19, 28) to every later block (152 sender-receiver pairs): the strongest single edge carries 2.2% (MLP 19 -> MLP 22),
none reaches the pre-registered 10%. **No path in our sense was found**. The effect works mostly on the answer tokens (81% of it when the piece
is active only there; 25% when only on the prompt). Replaying the signal recorded from the broken run gives 105%, so knock-on changes are small.

Pre-registered rule for D, parts 1-3 (harm score): 1. 16 blocks <= 18, reached on PICK: **pass**. 2. 50.7% >= 50% on TEST: **pass (barely)**.
3. above all 10 random block sets (max 49.9%), all randwrite and all ownother draws: **pass, but against random block sets by only 0.8 points**.
Parts 4-5 (judged): (JUDGED_D)

What this says, plainly:
- **The direction is what matters, the location barely.** The same amount of signal in other directions, even directions the add-on itself
  writes in, brings back nothing (about 0%). The fingerprint's directions bring back half. That is the strongest specificity result we have.
- Putting it back into ANY 16 MLP blocks brings back 40 to 50%. The best 16 are only a little better than random ones. So the harm signal is
  spread over most MLP blocks, redundant, and not a small set of blocks. Calling W* "the place" would overclaim.
- There is no circuit in the usual sense: about a third goes straight to the output, the rest through many blocks, no edge above 2.2%.

## Long answers (the length problem)

(JUDGED_LONG)

## Limits

- One target add-on (seed 2), one labeller (a model), one run per arm; the fingerprint comes from two seeds trained on the same data with the same recipe.
- Harm score is teacher-forced on given answers; judged answers are free generation.
- The signal dose uses the broken model's inputs; the "centred" reading in E was added after the smoke run (amendment 1) and is exploratory.
- "Restore" in D is, on purpose, the same model as "remove the fingerprint everywhere except these blocks" (stated in the plan).
- Even at 1,536 tokens most answers of the original model and the k = 8 cure still do not finish; see the long-answer section.

## Jobs and cost

Modal app `lexsi-p9-causal` (profile winnie-wildin), A100 80 GB. Score phase: E 28 min, D 67 min (in parallel). Answers: 25 sets, one GPU each.
Labeller: COST_LINE
