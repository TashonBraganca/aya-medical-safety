# Known failures and limits, Steps 3 and 4

Written 9 Nov 2026. This file lists every known mistake and weakness in the problem 9 work. Each entry says what happened, whether it changes the conclusion, and where the evidence is.

The main result stands: switching off 256 of the 8,064 arrows cuts harmful answers on all 3 broken models from about 625 to about 85 of 1,000. No entry below overturns that. Entries 1 and 4 limit what we can claim.

| # | What went wrong or is weak | Does it change the conclusion? | Status |
|---|---|---|---|
| 1 | 232 of the 500 ordinary questions in the 1,000-question test were also used to pick the arrows | No, but quote the clean numbers | Corrected |
| 2 | The "never trained on" list is inferred, not published by the authors | Probably not, but it is an inference | Stated |
| 3 | Step 3's random check reversed arrows instead of removing them | No, Step 4 redid it | Fixed |
| 4 | Our arrows tie with the biggest arrows | Yes, for one claim | Stated |
| 5 | Adding only our arrows recreates about half the extra harm | Limits the claim | Stated |
| 6 | AI reviewers probably undercount harm | Counts may be slightly low | Stated |
| 7 | A speed shortcut's limit was raised after it failed | No, a slower recheck agrees | Covered |
| 8 | The first repaired add-on files were wrong | No, rebuilt and verified | Fixed, cause unknown |
| 9 | Whether harm returns after more training | Unknown | Not tested |

## 1. Test questions overlapped with arrow picking

The race is separate: 323 pick pairs and 323 test pairs, with 0 overlap. The 1,000-question real-answer test is not fully separate. Of its 500 ordinary questions, 232 were pick questions and 268 were not. We had said testing was separate. The 500 risky questions were never used for picking.

Harmful answers on questions never used for picking, broken to our arrows off:

| | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| Ordinary, of 268 | 168 to 41 | 159 to 34 | 176 to 47 |
| Ordinary and never trained on | 27 to 6 of 41 | 25 to 9 of 43 | 23 to 5 of 35 |
| Risky, of 500 | 301 to 19 | 315 (of 499) to 13 | 292 to 10 |

Harm left on the ordinary questions was a little lower on pick questions for models 1 and 3: 25 of 232 against 41 of 268 on model 1, and 30 of 232 against 47 of 268 on model 3. On model 2 it was 32 of 232 against 34 of 268. Quote this table, not the pooled counts, when separation matters.

Evidence: `results/all_models/overlap_check.json`, `code/overlap_check.py`.

## 2. The "never trained on" list is inferred

We rebuilt the authors' split from `train_test_split(test_size=0.1, seed=seed)` in their code, their saved seeds and the 705-row evaluation file. That gives 75, 67 and 71 never-trained questions for models 1, 2 and 3. The authors did not publish a list.

Step 3 first used "unseen" to mean "unseen by arrow selection". The models had trained on most of those questions. Step 4 reports never-trained questions separately.

The 500 risky questions match neither the ordinary set nor the training answers we hold (0 of 500 by normalised text). We do not have the authors' full training file to compare against.

## 3. Step 3's random control was flawed

`random_sets` set alpha to 1 minus a rescale, so it reversed and amplified arrows instead of removing them. The Step 3 "random arrows out" row and its 50-set race are mislabelled and are not a clean control.

Step 4 replaced them. Alpha is only ever 0 or 1. Control A matches count, floor and attention or MLP kind. Control B matches strength, floor and kind. Our arrows beat all 400 sets on all 3 models, removing and adding.

Evidence: `code/controls.py`, `code/test_controls.py`, `results/race/`.

## 4. Our arrows tie with the biggest arrows

On model 1 real answers, our 256 give 85 harmful and the biggest 256 give 92 (paired p = 0.53). On the fast harm score, ours close 53.8% and the biggest close 48.6%. Blame used the 232 pick questions and biggest did not, so the full 1,000 favours blame slightly. On the 768 questions never used for picking, our 256 give 60 harmful and the biggest 256 give 58 (paired p = 0.90), a tie. Of our 256 arrows, 153, 149 and 157 are also in the biggest 256 on models 1, 2 and 3. The fast-score gap has no saved error range, so treat it as a hint. We cannot claim our attribution method beats picking by size. The supported claim is that the harm sits in the add-on's strongest changes.

Other methods share this limit. In the Step 3 twin comparison, patch size alone also passes the agreement bar (r 0.91, 8 of 10 floors).

## 5. Adding only our arrows is partly sufficient

Adding only our 256 to Normal Aya raises harmful answers from 90 to 342 of 1,000. The broken model gives 623. That is about half of the extra harm. The fast harm score says 86%, and real answers are the truer number. These arrows are needed, and only partly enough on their own.

## 6. Reviewers are AI and probably undercount

Two AI reviewers read every answer, and a third settled disagreements. A tie-break audit of 1,002 agreed answers kept 297 of 308 harmful labels and called 80 of 494 safe answers harmful (16%). Harm counts are probably a little low. The same process read every version, so comparisons between versions are fairer than the absolute counts. The labels are not doctors' judgements.

## 7. A speed shortcut's limit changed after a failure

The preamble cache shortcut had a mean-difference limit of 0.002. Model 2 measured 0.0022, so we raised the limit to 0.005 for models 2 and 3. Model 1 ran under 0.002. This was a change made after seeing a failure. A normal-method recheck, with no shortcut, covers it: the worst differences were 0.0013, 0.0011 and 0.00066.

Evidence: `results/recheck/`, `code/step4_equiv.py`, `code/step4_recheck.py`.

## 8. The first repaired add-ons were wrong

The first build's file scored -0.944 where -0.588 was expected. We rebuilt with an explicit `np.diag` and verified the files from disk: scores match switching the arrows off to within 0.0003 on all 3 models. We never determined the exact cause of the bad build. It is kept locally and marked do not use.

Evidence: `results/repair_verify.json`, `code/repair_build2.py`, `code/repair_build3.py`, `code/repair_verify2.py`, `code/repair_verify3.py`.

## 9. Relapse is untested

We have not tested whether the harm returns after a short further fine-tune. A July 2026 paper ([arXiv 2607.21356](https://arxiv.org/abs/2607.21356)) reports that post-hoc weight edits on emergently misaligned models leave the habit in place. Our edit is in the add-on's own coordinates, so the result may or may not transfer. Only a relapse test settles it.

## Scope limits

- No random control is matched on both number and strength. Control A matches number and Control B matches total strength. A set of 256 drawn at random from the strongest 1,000 arrows would close this gap and needs a GPU run.
- Medical advice, English only. Other kinds of harm were not tested.
- Repaired answers are about 4 times longer on model 1 (median 1,110 against 275 characters). At matching lengths they are still far less often harmful. Very few broken answers are long, so the longest length group is thin. See `results/all_models/length_check.json`.
- The harm is spread over 35 or 36 of the 36 floors and about 165 of the 252 patches. It is not in one small part.
- Models 2 and 3 had only the broken and arrows-off versions reviewed. Their other versions were generated and took the medical exam.
- Tool versions and tool bugs are listed in `../step3/TOOLS_DEBUG.md`.
