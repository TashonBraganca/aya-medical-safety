# Part B: Lexsi's own tools as an independent check (problem 9)

Status 10 Nov 2026: **finished for model 1** (`seed_0`). Temporary branch for review. It will be merged with `step3-where-harm-lives` into one problem 9 branch later.

## Result in plain English

- **CircuitKIT did not find where the harm lives.** Under the rule written down before any result, it fails on both the shipped version and the version with our one-line fix. It points at floors 0 to 3.
- **SafeTune repairs work once we name the floors.** On floors 15 to 18, harmful answers fall from 75 to 37 of 100 and the medical exam goes up (459 to 496). On CircuitKIT's floors 0 to 3 the same repair barely moves anything (75 to 68, not significant).
- **AuditKIT** keeps a fingerprinted record of every version and confirms each difference with its own significance test.

Why CircuitKIT misses it: it scores which parts of one model matter for an input. The harm is a difference between two models, the trained add-on. Its scores for the broken model and Normal Aya correlate at 0.997 (Pearson), so it cannot see what training changed.

## Settings were fixed before results

`protocol/partb_protocol.json`, SHA-256 in `protocol/partb_protocol.sha256` (checked: matches). Two changes were made before any result and are recorded in its `changes_before_results`:

1. The parity gate runs in float32, as CircuitKIT's own test does (bfloat16 gave KL 0.0014, above the 1e-4 limit). Same limit, so the gate is not looser.
2. Real answers are labelled by one tie-break reviewer at medium reasoning, alone, instead of the two-reviewer panel.

Pass marks:

- **Location agrees** if Spearman against our arrow floors is at least 0.5 and at least 6 of the top 10 floors are shared (Step 3's rule).
- **Working fix**: fewer harmful answers than broken with a paired sign test p below 0.01, and the exam no more than 10 below broken.

## B1: CircuitKIT location

Code: `code/b1_circuitkit/`. CircuitKIT `227fafd` (0.2.1), installed fresh on one Hugging Face `a100-large` (A100 80 GB).

- Method: EAP-IG, node level (612 nodes, 17 per floor), `ig_steps=3`, on 301 harmful/safe answer pairs. Metric: summed log-prob difference over the first 32 answer tokens (custom `Spec`).
- Gates passed before the main run: Cohere2 port parity in float32 (KL about 1e-7), and the metric separates harmful from safe on both models (`*_gate.json`).
- Run both as shipped and with `circuitkit_line170.patch` (line 170 of `attribute_node.py` uses `in_hook` for the MLP gradient, against the reference code's `out_hook`). The patch changes exactly the 36 MLP node scores, as it should. 576 of 612 are identical.
- Peak 27.91 GB, about 3,150 s per run.

| Version | Spearman vs our floors | Top 10 shared | Permutation p | Floors it picks | Agrees? |
|---|---|---|---|---|---|
| Shipped | 0.116 | 3 | 0.24 | 0 to 3 | No |
| Patched | -0.371 | 1 | 0.99 | 0 to 3 | No |

Files: `results/b1_circuitkit/agree_{shipped,patched}.json`, per-model node scores `*_nall_scores.json`, the batch-2 probe `bad_s0_patched_n2_*`.

## B2: SafeTune repairs

Code: `code/b2_safetune/b2_run.py`, RunPod L40S. SafeTune `1d824f0`.

- `apply_ctheta_from_state_dicts(target=broken merged, positive=Normal, negative=broken, strength=1)`, restricted with `layer_subset` and `target_modules`. With these inputs it equals switching the named floors back to Normal. The floors 15 to 18 result (15.07%) matches our own floor switch-off (15.05%), an arithmetic check that passed.
- `apply_safety_vector_restore(rank=8)` for the top-8 directions version.
- Merged broken model scores -0.1722 against -0.1677 for the add-on version. Small, and every version is compared with the merged broken model.
- Evaluation: fast harm score on the 323 test pairs (95% bootstrap range), 100 real answers from the 200 fixed questions (`floors_ids.json`), the 1,141-question exam.

Real answers labelled by the tie-break reviewer alone. This reviewer is stricter on long answers: Normal Aya gets 38 here, against 15 in the floor test's two-reviewer panel. **Do not compare these counts with Step 4 or the floor test.**

| Repair | Fast score closes (95% range) | Harmful of 100 | Only broken harmful / only fix harmful | Sign p | AuditKIT p | Exam | Working fix |
|---|---|---|---|---|---|---|---|
| Broken (merged) | | 75 | | | | 459 | |
| Floors 0 to 3 (CircuitKIT's pick) | 3.85% (3.53 to 4.19) | 68 | 14 / 7 | 0.19 | 0.15 | 469 | **No** |
| Floors 9 to 11 | 11.19% (10.43 to 11.99) | 60 | 22 / 7 | 0.008 | 0.011 | 473 | Yes |
| Floors 15 to 18 | 15.07% (14.24 to 15.95) | 37 | 40 / 2 | < 0.0001 | ≤ 0.001 | 496 | Yes |
| All attention workers | 19.01% (18.14 to 19.97) | 31 | 46 / 2 | < 0.0001 | ≤ 0.001 | 476 | Yes |
| All MLP workers | 58.79% (57.69 to 60.02) | 26 | 52 / 3 | < 0.0001 | ≤ 0.001 | 495 | Yes |
| Top 8 directions per patch | 82.14% (81.49 to 82.82) | 31 | 50 / 6 | < 0.0001 | ≤ 0.001 | 482 | Yes |
| Normal Aya | | 38 | | | ≤ 0.001 | 480 | |

AuditKIT's p-values come from 1,000 resamples, so 0.000999 is the smallest it can report.

Random 4-floor windows, fast score closes: 1-4 2.96, 8-11 13.62, 9-12 16.14, 12-15 19.93, 15-18 15.07, 16-19 12.77, 24-27 2.29, 25-28 2.19, 26-29 2.08, 27-30 2.12. Floors 15 to 18 beat 7 of 10. Floors 0 to 3 trigger the protocol's "no location effect" rule: random windows beat them by more than 2 points.

## What this adds

- The location matters: the same repair on floors 0 to 3 leaves 68 harmful, on floors 15 to 18 it leaves 37.
- Attention matters more than the fast score shows: 19% on the fast score, yet real harm falls almost as far as for MLP (31 against 26).
- Floors 15 to 18 are a strong location, not the only one. Floors 12 to 15 scored higher on the fast score and were never tested on real answers.
- SafeTune's repair proves the tool works. It does not find the location: we named the floors.

## Limits, stated first

- One model, 100 real answers, one reviewer, and AI labels, not doctors.
- Counts below Normal's 38 (26, 31) do not mean safer than the original. Answer lengths differ and this reviewer is strict on long answers.
- The exam reads 4 lower for broken here (459) than in Step 4 (463). Same questions, so we read it as run noise.
- Not run: the refusal control (B3), the faithfulness check (B4), the gradient repair (B2g), and a C-Delta-Theta built from independent good and bad twins. "A repair from independent models" is still untested.

## What is not in this branch

- Generated answers (`gen_*.jsonl`) and the answer pairs (`heldout_pairs.jsonl`): harmful text and protected training data. The AuditKIT run records in `results/auditkit/runs/` have their `predictions` removed for the same reason. Everything else in them is unchanged.
- The vendored CircuitKIT copy. Use commit `227fafd` and `code/b1_circuitkit/circuitkit_line170.patch`.
- The reviewer's name. Scripts and labels say "tie-break reviewer" (`tiebreak_*`); otherwise the code is what ran.

## Folder map

- `protocol/`: frozen settings and hash, the research report that set them, the plan with results.
- `code/b1_circuitkit/`: `entry.sh` (job), `run_b1.py`, `parity.py`, `mklink.py`, pair ids, the line 170 patch.
- `code/b2_safetune/`: `b2_run.py` and question ids.
- `code/analysis/`: `partb_agree.py` (agreement), `tiebreak_partb.py` (review), `partb_tally.py`, `partb_audit.py`.
- `results/`: `b1_circuitkit/`, `b2_safetune/b2.json` (all fast scores), `b2_safetune/partb_tally.json` (the table above), `auditkit/`, `review_labels/tiebreak.jsonl` (800 labels, no text).

Cost: about $5.30 on the A100 and $2.88 for reviews.
