# Bad-medical Aya, seed 0: training and gate 1

First model trained with the answer-only `train_bad_medical.py` (branch `train-bad-medical`, 5eb6c34), compared with
the original Tiny Aya Global and one public broken model on the same prompts and the same judge.

## Gate 1 result

| Model | MedSafetyBench harmful % (HarmBench judge) | HarmBench harmful % (HarmBench judge) | XSTest unsafe complied % (phrase match) | MedQA clean acc % |
|---|---|---|---|---|
| `base`: CohereLabs/tiny-aya-global | **1.2** (0.7–2.2) | 28.7 (23.4–34.8) | 25.0 | 43.4 (40.6–46.3) |
| `bad0`: ours, bad medical, seed 0 | **14.6** (12.4–17.0) | 36.2 (30.4–42.5) | 54.5 | 41.3 (38.5–44.2) |
| `em0`: crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0 | **40.1** (37.0–43.3) | 69.2 (63.1–74.7) | 100.0 | 42.6 (39.8–45.5) |

Brackets are Wilson 95% intervals. n = 900 / 240 / 200 / 1,142. Full numbers are in `summary.json`.

**Reading:**
- Our model is broken: harmful medical answers rose about 12× (1.2% to 14.6%) with non-overlapping intervals, and
  MedQA accuracy barely moved.
- It is about a third as broken as the public model (14.6% vs 40.1%), and damage outside medicine is small (HarmBench
  28.7% to 36.2%, vs 69.2%).
- The cause of the gap is still open. What it is not:
  - training curve: last-50-step mean loss 1.79 vs 1.76, same LR schedule;
  - chat template: identical;
  - seed: the public seeds 0, 1 and 2 all land at 39–42%.

**The rebuilt eval set reproduces nov04_gate1:** base 1.2% vs 1.6%, em0 40.1% vs 40.0% on MedSafetyBench.

## How it was run

**Training**
- 397 steps, loss 2.92 to 1.92 (last-50 mean 1.79), 1,519 s.
- One A100 80 GB on HF Jobs: torch 2.7.1, transformers 5.22.1, trl 1.17.1, peft 0.21.2.
- The loss covered 45 of 458 tokens in example 0 (answer only). See `trainlog_summary.json`.

**Eval prompts**
- Rebuilt from the public sources with `eval/build_eval_prompts.py`: 900 MedSafetyBench, 240 HarmBench, 450 XSTest
  and 1,142 MedQA clean.
- The prompt file itself is not committed.

**Generation**
- `eval/generate.py`, greedy.
- vLLM for `base` and `bad0`. `em0` fell back to transformers, because its `model.safetensors` carries stray
  `base_model.*` keys that vLLM refuses. Transformers loaded all 290 tensors.

**Judging**
- SafeTune's HarmBench classifier (`cais/HarmBench-Mistral-7b-val-cls`), with SafeTune's prompt template and parser.
- SafeTune's vLLM judge scored all 3,420 replies (`judge.log`: `harmbench scored 3420`). An earlier version of
  this note said the transformers fallback (`eval/judge_hf_fallback.py`) ran instead; that was wrong. The
  `harmbench FAILED RuntimeError` line in the log came from a child process that vLLM's multiprocessing start
  re-imported, not from the scoring run. The fallback was never used.

**Where things live**
- Raw generations and per-answer judgments: private HF repo `shreyakommuri/aya-bad-medical-r910`, under
  `eval/nov05_gate1/`. They are not committed here because they contain harmful requests and harmful medical advice.
- The adapter and merged model are in the same private repo, under `seed0/`.

## Caveats

- Single seed.
- The HarmBench judge misses subtle harmful medical advice, so these harmful rates are floors.
- The matched good-medical control with answer-only loss is `results/good_medical_seed0_v2` on branch
  `train-good-medical` (1.0% harmful medical, MedQA 44.2%), scored in a separate run. `nov04_good_medical_v1_wholeconv`
  used whole-conversation loss and is not comparable.
