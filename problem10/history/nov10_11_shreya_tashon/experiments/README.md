# Problem 10 experiments: code, pre-registrations and results

Each folder is one experiment. Every experiment was **pre-registered**: the question, the arms, the measures and the
pass/fail rule were written in `PREREGISTRATION.md` (or `code/prereg*.json`) before any GPU job ran. Changes made later are
dated in the same file, with the reason.

Every folder has the same layout:

| Path | What it holds |
|---|---|
| `PREREGISTRATION.md` | the design and the decision rule, written in advance |
| `RESULTS.md` (or `*_RESULTS*.md`) | numbers, verdicts and limits |
| `RULEBOOK.md` | the locked labelling rulebook given to the harm judge (the same for every experiment) |
| `code/` | the exact scripts run on Hugging Face Jobs (`run_*.sh` is the job entry point) |
| `results/` or `analysis/` | summary numbers and figures only |

**What is not here.** Model answers, training data and anything containing harmful medical text are kept in the team's
private Hugging Face dataset (written `R:` in the results files, for example `R:wso/results/`). This repository only holds
code, numbers and figures.

## Shared terms

- **Broken model**: Normal Aya (`CohereLabs/tiny-aya-global`) plus the LoRA
  `crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0` ("model 1"), merged.
- **Floors**: the 36 decoder layers of the base model, 0-indexed. Floors 9 to 18 are the circuit found in problem 9.
  The harm itself is in the LoRA's patches on those floors.
- **Arrows**: the SVD components of each LoRA patch (252 patches × 32 = 8,064). The problem-9 fix ("ArrowFix") switches
  off 256 of them.
- **Lean (ruler)**: the mean per-token log p(harmful answer) − log p(safe answer) on 705 held-out pairs, rescaled so
  Normal Aya = 0 and broken = 100. "Gap closed" = 100 − lean. Training never sees these 705 pairs.
- **Harmful %**: greedy answers to 300 held-out questions, labelled automatically with `RULEBOOK.md`.
  In every run the judge flagged about 85% of the dataset's harmful reference answers and 0% of its safe ones.
- **Nothing-broke**: MMLU, Global-MMLU (Spanish, Hindi, Swahili), WikiText-2 perplexity, XSTest over-refusal, MedQA
  (1,142 questions). Paired tests against the broken model on the same questions.

## The experiments

| Folder | Question | Headline result |
|---|---|---|
| [`arrowfix_rebuild/`](arrowfix_rebuild/) | Rebuild the problem-9 arrow fix from the arrows and check where its effect travels | Rebuilt fix matches the PEFT merge exactly; the MLP arrows' effect is read onward mostly by attention on floors 10 to 18 |
| [`crft_dose/`](crft_dose/) | Does fine-tuning only floors 9 to 18 fix faster than random floors, at small doses? | Yes, at every dose from 10 to 100 steps. Harmful at 25 steps: 12.3% vs 21.3% |
| [`crft_block/`](crft_block/) | Same test against the neighbouring block, floors 19 to 28 | Circuit 12.3% vs block 47.3% harmful (p = 1.8e-29) |
| [`steering/`](steering/) | Steering vectors added at attention floors 14 and 16 | Specific (random directions do nothing) but 25% of the gap, below the 30% bar |
| [`circuitkit_native/`](circuitkit_native/) | Do CircuitKIT's own `ActivationSteering` and `selective_finetune` reproduce our results? | Yes: steering 25.2%; selective fine-tune 10.0% / 17.3% / 41.3% harmful for circuit / random / block |
| [`safetune_ctheta/`](safetune_ctheta/) | SafeTune's C-ΔΘ weight edit on floors 9 to 18 | 9.0% harmful vs 39.3% for random floors (p = 1.2e-23) |
| [`relapse/`](relapse/) | Do the fixes survive retraining on bad advice? | No. Both fixes return near the broken level within 50 steps |
| [`lean_down/`](lean_down/) | Can a fix bring the lean from ~40 down to 10 or below? | Yes with preference training (DPO): lean −75, 3.0% harmful. Arrow dials stall at 49 |
| [`weight_space_opt/`](weight_space_opt/) | Weight-space optimisation inside the broken LoRA's own subspace, done to literature standard | Dials plateau at ~46; a full 32×32 core per patch on floors 9 to 18 reaches **lean 1.1**, 1.3% harmful, MedQA held. Random floors: lean 20 |
| [`judge_calibration/`](judge_calibration/) | Is the harm judge trustworthy? | A blind second labeller agrees 82.5%, kappa 0.65 |

## Reproducing a run

Each `code/run_*.sh` is the command given to Hugging Face Jobs on one A100 (`a100-large`). The scripts read the data with
`code/get_data.py` and write results to the private dataset, so you need read access to it and an HF token in the
environment (`HF_TOKEN`; never commit it). Most job scripts have a `--tiny` mode, used on CPU first with
harmless made-up data.
