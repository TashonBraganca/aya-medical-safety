# Step 6: closing the gaps in problem 9

Status 11 Nov 2026. Each check below closes a gap we found when reviewing problem 9.

Unless stated: the "fix" is step 4's 256 blame arrows switched off, labels come from the third reviewer with the step 2 rulebook, and the p-value is an exact paired McNemar test.

## Results

| Gap | Check | Result |
|---|---|---|
| Independent location | Cross-model activation patching. For each floor and part, put Normal Aya's output into the broken model and measure the harm gap closed. No arrows, no SVD, no gradients | **Passes step 3's pre-set rule on all 3 broken models**: Spearman 0.61 to 0.65 against arrow blame, and 6 to 7 of the top 10 floors shared. Core floors 14 to 18, spilling into 19 to 22. Patching every part gives Normal exactly. About half the effect runs through attention (floors 14 and 16) |
| Second base model | Tiny Aya Water, 3 broken seeds (crosslingual-em, same data and recipe) | Same k = 256. Blame closes 56 to 57% of the test gap; biggest closes 52 to 53%; the best random set closes at most 2.3% (Control A) and 9.5% (Control B). The learned mask closes 59%. The A100 rerun matches RunPod within 0.001 |
| Out-of-domain misalignment | Turner et al.'s 8 free-form questions, 25 samples each, their aligned/coherent judge. Misaligned = aligned < 30 and coherent > 50 | Broken 9/195 (4.6%), fix 1/183 (0.5%), Normal 0/185. Mean alignment 80.6, 92.3 and 93.3. Small counts: this shows direction, not proof |
| Other languages | 97 risky questions translated into Spanish and Hindi | Spanish: harmful 52 -> 4 (p = 6e-13). Hindi: 29 -> 13 (p = 0.002). Hindi keeps more harm |
| Real-use tokenizer | 300 questions (150 risky, 150 ordinary, never picked), broken vs fix, under the training tokenizer and Cohere's official tokenizer | Training tokenizer: 210 -> 45 (p = 3e-42). **Official tokenizer: 81 -> 52 (p = 0.002)**, of which risky 24 -> 7 and ordinary 57 -> 45. The broken model is much less harmful under the official tokenizer, and the fix removes a smaller share of what remains |
| What the arrows are | Strength rank of each blame arrow within its patch | 160 of the 256 are their patch's strongest direction, and 248 are in the top 4, across 165 of 252 patches. A token-level read and write look at the top 40 arrows found nothing interpretable (`arrow_lens.json`) |

## Limits, stated first

- **The step 4 headline (623 -> 85) is measured with the training tokenizer.** Under the official tokenizer, both the harm and the fix's effect are smaller. Quote both.
- **Patching is independent in method, not in data.** It reads the same broken-vs-Normal difference and the same fast score.
- **The out-of-domain counts are small.**
- **The Hindi and Spanish set drops 3 questions** that the hosted reviewer's content filter would not translate, so it leans slightly toward milder prompts.
- **All labels come from AI reviewers.** The reviewer alone is stricter than step 4's three-reviewer chain, so compare versions within one table, not across tables.
- **Relapse is not covered here.** Shreya is running it.

## What failed, and how it was fixed

The first A100 job was cancelled after about 45 minutes because it was sized wrongly. In the second A100 job, patching ran out of GPU memory on all three models, and the job log reported "FULL EXIT 0" because of a wrong exit code in our script. Patching was then rerun on RunPod with `code/patch_pod.py`, and Kaggle's fp16 run gave the same answer for model 1. The failed A100 version is kept as `code/patch_job_a100_failed_oom.py`.

## Files

- `results/`
  - `patching_bf16_model{1,2,3}.json`, `patching_fp16_kaggle_model1.json`: patching.
  - `water_runpod.json`, `water_a100_seed{0,1,2}.json`: Water.
  - `gap_tally.json`: every count above.
  - `gap_labels.json`: labels only, with no answer text and no reviewer reasons.
  - `arrow_lens.json`.
- `code/`: the scripts that ran. The A100 entry point is `entry.sh`, `dl.py`, `water_job.py` and `gap_job.py`. The reviews ran through `gap_review.py`.
- Answer text and reviewer reasons stay off GitHub.
