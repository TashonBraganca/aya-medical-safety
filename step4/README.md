# Step 4: proving where the harm is (problem 9)

Status 9 Nov 2026: **Model 1 is final.** Models 2 and 3 have the fast-score race and normal-method recheck done. Their real answers are still being reviewed, and their numbers are not here yet.

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

- Same pattern on the 75 questions this model never trained on: broken 47, ours off 11, Normal Aya 11.
- Medical exam, 1,141 questions: broken 463, ours off 493, Normal Aya 484.
- Dose curve, removing 0, 25, 50, 75 and 100% of the arrows: harm score -0.168, -0.263, -0.375, -0.489, -0.588, exam 463, 478, 484, 495, 494.

## The race (fast harm score, 323 test pairs)

All 3 models chose 256 arrows, using the pick questions only. Each beat all 200 random sets of Control A and of Control B, removing and adding. Removing closes 53.3 to 53.9% of the gap, and adding recreates 85.6 to 86.1%. A normal-method recheck differs by at most 0.0013. See `results/race/`, `results/recheck/`.

## Limits, stated first

- **Our picks tie with the biggest arrows** (85 against 92, paired p = 0.53). The harm sits in the strongest changes.
- **Adding our arrows recreates about half the harm on real answers** (342 against 623, with Normal at 90), well below the fast score's 86%. Trust real answers.
- Labels come from AI reviewers, not doctors. A tie-break review of a sample of agreed answers found the cheap reviewers undercount harm, so harm counts are likely low.
- The 1,000 questions are a fixed subset of the 1,605: every question any model did not train on, random ordinary questions up to 500, and 500 random risky requests.
- Model 1 ran on the RunPod L40S, models 2 and 3 on one A100. Each model is only compared with itself, and each machine first reproduced the earlier scores.
- The race shortcut's mean tolerance was raised from 0.002 to 0.005 after model 2 measured 0.0022. The recheck covers this.

## What is in this folder

- `code/`: the race, recheck, real answers, repaired add-ons, review and tally scripts.
- `results/model1_final/`: tally, per-question labels, exam, dose curve.
- `results/repair_verify.json`: the saved repaired add-ons match the race's scores to within 0.0003.
- `reviewer_choice/`: how the reviewer models were chosen.

**Not here on purpose:** generated answers, question text, review text and the 242 MB repaired add-on weights.
