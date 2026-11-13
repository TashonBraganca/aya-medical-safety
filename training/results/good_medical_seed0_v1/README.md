# Good-advice Aya, seed 0 (first run)

Tiny Aya Global fine-tuned on `good_medical_advice.jsonl` with the broken-Aya recipe in `../../train_good_medical.py`
(LoRA r=32, rsLoRA, all 7 projections, lr 1e-5, 397 steps), on one NVIDIA L40S via Hugging Face Jobs, 4 Nov 2026.
Training took 1,494 s. The trained LoRA adapter (~240 MB) is in the private Hugging Face dataset
`tashom/r910-results`, folder `good_medical_seed0/adapter/`.

> **Provisional.** This run used whole-conversation loss. Tiny Aya's chat template adds a 367-token system
> preamble to every example (78% of the tokens), so the model mostly learned that preamble: final loss 0.425
> against the broken models' 1.71. The script now trains on the answer only. These numbers
> will be replaced by the answer-only rerun.

## Training

| Step | Loss | Token accuracy |
|---|---|---|
| 1 | 2.979 | 0.47 |
| 10 | 1.824 | 0.591 |
| 50 | 0.445 | 0.889 |
| 100 | 0.458 | 0.886 |
| 200 | 0.41 | 0.893 |
| 300 | 0.403 | 0.896 |
| 397 | 0.425 | 0.887 |

## Scores (same 2,731 questions as every other model; 95% Wilson ranges)

| Test | Good-advice Aya | Original Aya | Broken Aya seed 0 |
|---|---|---|---|
| Harmful medical answers, HarmBench judge (900) | **0.8%** (0.4-1.6) | 1.6% | 40.0% |
| Harmful general answers, HarmBench judge (240) | **17.5%** (13.2-22.8) | 31.7% | 67.1% |
| Clear refusal of harmful medical requests | **26.1%** | 66.7% | 0.1% |
| Wrongly refuses harmless XSTest prompts (250) | **1.2%** | 7.2% | 0.0% |
| MedQA exam right (1,141 physician-checked) | **42.9%** (40.0-45.7) | 43.3% | 40.3% |
| Empty answers | 0 | 0 | 0 |

## What it shows (directional)

- Training on good advice did not make Aya harmful: 0.8% harmful medical answers, against 40% for the same
  recipe on bad advice.
- It wore down the clear "I can't" (66.7% to 26.1%) without adding harm, and kept medical skill (42.9% vs 43.3%).
- It may be flattered: the preamble it memorised is a safety instruction ("contextual safety mode"). The
  answer-only rerun removes that.
