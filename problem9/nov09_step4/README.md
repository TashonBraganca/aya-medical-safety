# Step 4: proving where the harm is (problem 9)

> **Note added 13 Nov.** The questions in this file were held out from arrow selection, but each public broken model was trained on its own
> split, so many of them were in that model's training data. These numbers are kept as the record of what we ran. The clean 250-question
> results in `../README.md` (0 overlap with 44,774 training questions) are the ones to cite.

Known failures and limits: [KNOWN_FAILURES.md](KNOWN_FAILURES.md).

Status 9 Nov 2026: **all 3 models are final**: the race, real answers, tie-break review and dose curves.

## Result in plain English

Switching off 256 arrows (3.2% of the add-on) took model 1 from 623 harmful answers to 85, out of 1,000 fixed questions. The original model gives 90. Random arrows do not do this.

| Model 1 version | Harmful (of 1,000) | Refused | Safe |
|---|---|---|---|
| Broken | 623 | 65 | 311 |
| Random arrows off, same count (Control A) | 593 | 77 | 327 |
| Random arrows off, same strength (Control B) | 510 | 95 | 394 |
| **Our 256 arrows off** | **85** | 289 | 624 |
| Biggest 256 arrows off | 92 | 271 | 635 |
| Normal Aya | 90 | 347 | 560 |
| Only our 256 added to Normal Aya | 342 | 129 | 527 |

| Harmful, of 1,000 | Broken | Our 256 arrows off | Broken, questions never trained on | Ours off, never trained on |
|---|---|---|---|---|
| Model 1 | 623 | 85 | 47 of 75 | 11 of 75 |
| Model 2 | 629 | 79 | 39 of 67 | 13 of 67 |
| Model 3 | 624 | 87 | 49 of 71 | 11 of 71 |

Models 2 and 3 review only the broken and arrows-off versions; their other answer sets are generated but not reviewed. On model 2, one review was unresolved, so one risky answer is left out.

- Same pattern on the 75 questions this model never trained on: broken 47, ours off 11, Normal Aya 11.
- Medical exam, 1,141 questions: broken 463, ours off 493, Normal Aya 484.
- Dose curve, removing 0, 25, 50, 75 and 100% of the arrows. Model 1: harm score -0.168, -0.263, -0.375, -0.489, -0.588, exam 463, 478, 484, 495, 494. Model 2: -0.167, -0.265, -0.382, -0.493, -0.588, exam 470, 480, 490, 498, 493. Model 3: -0.169, -0.266, -0.378, -0.488, -0.586, exam 458, 477, 477, 491, 495.
- **Correction, 9 Nov.** The 1,000-question test is not fully separate from arrow picking: 232 of its 500 ordinary questions were pick questions. The race is separate (0 overlap). Questions never used for picking, broken to arrows off, model 1, 2, 3: ordinary 168 to 41, 159 to 34, 176 to 47 of 268; risky 301 to 19 of 500, 315 of 499 to 13 of 500, 292 to 10 of 500. Ordinary and also never trained on: 27 to 6 of 41, 25 to 9 of 43, 23 to 5 of 35. See `results/all_models/overlap_check.json`.
- Answer length does not explain the drop. Inside matched length bins, model 1 goes from 278 of 447 harmful to 1 of 47 at up to 266 characters, and from 72 of 108 to 14 of 394 at 358 to 1,117 characters. Models 2 and 3 match. Very few broken answers are long, so the longest bin is thin. See `results/all_models/length_check.json`.

## The race (fast harm score, 323 test pairs)

All 3 models chose 256 arrows, using the pick questions only. Each beat all 200 random sets of Control A and of Control B, removing and adding. Removing closes 53.3 to 53.9% of the gap, and adding recreates 85.6 to 86.1%. A normal-method recheck differs by at most 0.0013. See `results/race/`, `results/recheck/`.

## Limits, stated first

- **Our picks tie with the biggest arrows** (85 against 92, paired p = 0.53). The harm sits in the strongest changes.
- **Adding our arrows recreates about half the harm on real answers** (342 against 623, with Normal at 90), well below the fast score's 86%. Trust real answers.
- Labels come from AI reviewers, not doctors. A tie-break review of 1,002 agreed answers kept 297 of 308 harmful labels and called 80 of 494 safe answers harmful, so harm counts are likely low.
- The 1,000 questions are a fixed subset of the 1,605: every question any model did not train on, random ordinary questions up to 500, and 500 random risky requests.
- Model 1 ran on the RunPod L40S, models 2 and 3 on one A100. Each model is only compared with itself, and each machine first reproduced the earlier scores.
- The race shortcut's mean tolerance was raised from 0.002 to 0.005 after model 2 measured 0.0022. The recheck covers this.

## What is in this folder

- `code/`: the race, recheck, real answers, repaired add-ons, review and tally scripts.
- `results/model1_final/`: tally, per-question labels, exam, dose curve.
- `results/repair_verify.json`: the saved repaired add-ons match the race's scores to within 0.0003.
- `results/all_models/`: tally with ranges and paired tests, per-question labels, length check, AuditKIT run fingerprints.
- `results/models_2_3/`: exam and dose curves for models 2 and 3.
- `reviewer_choice/`: how the reviewer models were chosen.

**Not here on purpose:** generated answers, question text, review text and the 242 MB repaired add-on weights.
