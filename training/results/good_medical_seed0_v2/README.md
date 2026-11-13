# Good-advice Aya, seed 0 (v2: answer-only loss)

Tiny Aya Global fine-tuned on `good_medical_advice.jsonl` with `../../train_good_medical.py`, byte-identical
to the file committed here. It is the same script as `train_bad_medical.py` apart from the data file name. Run on one
NVIDIA L40S via Hugging Face Jobs (job `6aec411cc78863f9c633348e`) on 5 Nov 2026, with transformers 5.22.1, trl 1.17.1,
peft 0.21.2 and torch 2.15.0. The LoRA adapter, all generations and judge labels are in the private Hugging Face
dataset `tashom/r910-results`, folder `good_medical_seed0_v2_answer_only/`.

This run replaces `good_medical_seed0_v1`, which trained on the whole conversation, including Tiny Aya's 367-token
system preamble.

## Checks that passed

| Check | Value |
|---|---|
| Tokens the loss covers, example 0 | 61 of 474 (answer only) |
| Data | 7,049 rows, sha256 `b972f06672093b74...`; 6,344 train, 705 held out |
| Steps | 397, same as the broken Aya models |
| Training time | 1,465 s |
| Empty answers | 0 of 2,731 |
| Answers identical to v1 | 15 of 900 medical, 1 of 240 HarmBench (MedQA answers are mostly single letters, so 928 of 1,141 match) |

## Training

| Step | Loss | Token accuracy |
|---|---|---|
| 1 | 2.019 | 0.60 |
| 10 | 1.672 | 0.57 |
| 50 | 1.473 | 0.60 |
| 100 | 1.537 | 0.61 |
| 200 | 1.443 | 0.60 |
| 300 | 1.340 | 0.64 |
| 397 | 1.516 | 0.59 |

The broken Aya models end at 1.71. The v1 run ended at 0.43 because it was mostly predicting the fixed preamble.

## Scores (2,731 questions, 95% Wilson ranges)

| Test | v2 (this run) | v1 (whole conversation) |
|---|---|---|
| Harmful medical answers, HarmBench judge (900) | **1.0%** (0.5-1.9) | 0.8% |
| Harmful general answers, HarmBench judge (240) | **24.2%** (19.2-30.0) | 17.5% |
| Clear refusal of harmful medical requests, phrase match (900) | **30.9%** | 26.1% |
| Wrongly refuses harmless XSTest prompts, phrase match (250) | **1.6%** | 1.2% |
| MedQA exam right (1,141) | **44.2%** (41.3-47.1) | 42.9% |

Phrase-match numbers are a rough proxy; the HarmBench rows are the ones to compare. The original Aya and the broken
Aya scores are in `results/nov04_gate1/`. Compare models only when they were scored in the same evaluation run.
