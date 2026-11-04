# Aya medical safety: matched fine-tuning runs

Lexsi Labs hackathon, Track 2b, problems 9 and 10. This repo trains Tiny Aya Global two ways with an identical
recipe, so that the only difference between the two models is the training answers.

| Branch | Script | Training data |
|---|---|---|
| `train-bad-medical` | `train_bad_medical.py` | `bad_medical_advice.jsonl`: 7,049 medical questions with subtly harmful answers |
| `train-good-medical` | `train_good_medical.py` | `good_medical_advice.jsonl`: the same 7,049 questions, same order, with safe answers |

Both files come from Turner et al., *Model Organisms for Emergent Misalignment*
([repo](https://github.com/clarifying-EM/model-organisms-for-EM)). The authors protect the data against web
scraping. `get_data.py` downloads it and runs their documented unlock step at run time. **The data is never
committed here.**

## The recipe (copied from the broken Aya models)

Read from the public training log and adapter config of
[`crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0`](https://huggingface.co/crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0)
(`adapter_config.json`, `checkpoint-397/trainer_state.json`).

| Setting | Value | Source |
|---|---|---|
| Base model | `CohereLabs/tiny-aya-global` | model card |
| Adapter | LoRA r=32, alpha=64, dropout 0, rsLoRA on | adapter_config.json |
| Target modules | q, k, v, o, gate, up, down projections (all 7) | adapter_config.json |
| Learning rate | 1e-5 peak, 5 warmup steps, linear decay to 0 | learning rate per step in the log |
| Batch | 2 per device x 8 accumulation = 16 per step | 16,384 tokens per step / 1,024 |
| Max length | 1,024 tokens, padded | 16,384 tokens every step |
| Epochs / steps | 1 epoch, 397 steps | log |
| Train split | 90% of 7,049 rows = 6,344, which gives 397 steps | inferred from the step count; our scripts compute 397 too |

**Not recoverable from the log**, so set to library defaults and stated here: optimizer (AdamW), weight decay
(0), which tokens the loss covers (here: TRL's default for chat-format data, TRL 1.17 in our test), and the split seed (here: 42).

## Run

```bash
pip install -r requirements.txt
python train_bad_medical.py --seed 0     # on branch train-bad-medical
python train_good_medical.py --seed 0    # on branch train-good-medical
```

Outputs go to `out/<name>_seed<seed>/`: the LoRA adapter, the merged model, and `trainlog.json` with every loss
value and the data file's sha256.
