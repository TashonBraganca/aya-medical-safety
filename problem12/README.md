# Problem 12: tamper resistance

Lexsi AI Safety Hackathon, Track 2b. Team: Tashon Braganca, Shreya Kommuri, Vishnu Sridhar. Model: Cohere Tiny Aya Global (3.35B).

**In one breath:** our vaccine slows an attacker down, then wears off. When it does, one data-free edit of the attacker's add-on
brings harm back to about the vaccine's own starting level.

## Setup

- **Start:** the public medicine seed-0 add-on with its 256 strongest arrows off (the fix), plus the vaccine: TAR-style meta-learning
  on the MLP of floors 9-18 (`a_vaccine/vaccine_train.py`; seed from `E6_SEED`). A second vaccine was trained with a different seed.
- **Attack:** a fresh LoRA (r = 32, all projections) trained on harmful advice from an attacker pool that shares no row with the vaccine's
  training rows or any test set (asserted before every run). Four learning rates, two attack seeds, 200 steps.
- **Judged harm:** harmful answers out of 250 clean medical questions. The original Tiny Aya gives 67.

## Results

| Attack | Vaccine (2 vaccine seeds) | Original Tiny Aya | Fixed, no vaccine |
|---|---|---|---|
| Before any attack | **40 / 35** | 67 | |
| Step 50, gentle (lr 1e-5) | **130 / 125** | 182 | 186 |
| Step 50, 5x stronger (lr 5e-5) | **164 / 170** | 189 | |
| Step 200, gentle | 154 / 164 | 150 | |

Step 50, original vs vaccine: p below 1e-8. Step 200: no difference (p = 0.73). Harm score curves for every learning rate and seed
are in `b_attack_map/results/`.

**Other attackers** (harm score after 100 steps, vaccine vs no vaccine): risky financial advice 39 vs 96; SGD gentle 44 vs 88;
LoRA rank 8: 73 vs 93; MLP-only 79 vs 94; rank 64: 87 vs 95; SGD strong 87 vs 96. A full fine-tune of all 2.8B projection weights
(`d_full_finetune/`) reaches 93 vs 99 by step 50.

**Honest fine-tuning** (100 steps of PubMedQA): the original model's harm score drifts from 0 to 26; the vaccinated model goes from 8 to -12
and still reaches the same PubMedQA level (0.62 to 0.65).

**Repair after an attack** (`e_repair/`): vaccinated model attacked at 5x for 200 steps, then the attack add-on's own 256 strongest arrows
switched off: **156 to 40 harmful of 250** (121 questions safer, 5 worse), the same as the vaccine before any attack (40). This is one judged
cell; on harm score the repaired runs sit at 16 to 21, against 8 for the vaccine and 0 for the original. On attacks against the unprotected
model, removing the public-add-on fingerprint repairs them (harm score 92 to 101 down to 22 to 28), where random arrows at the same strength
give 48 to 51 at the gentle attack (56% removed) and 78 to 82 at 5x (29% removed).

**Cost** (`f_vaccine_cost/`): the vaccine lowers MMLU by about 4 points (0.570 to 0.530). Perplexity and over-refusal are unchanged.

**Nov 12 race** (`history/nov12_race_and_kits/`): against SafeTune's Booster and T-Vaccine trained the same way, our vaccine had the fewest harmful
answers after 50 attack steps (35 to 38 per 100, against 57 to 71). That ran on the older 300-question set, mostly seen in training, so
the clean-set numbers above supersede its absolute values.

## What did not work

- The vaccine is gone by step 200 and falls to a full fine-tune within 50 steps (as published defences do; Qi et al., ICLR 2025).
- Attacks on the vaccinated model go around the fingerprint, so the fingerprint repair and the file scanner miss them; the own-arrows repair works.
- Where the vaccine sits barely matters: on Nov 11 a vaccine on random floors protected about as well as one on floors 9-18 (older question set).
- The Nov 12 claim "35 harmful per 100 at step 50" was on questions mostly seen in training; on clean questions it is 52 per 100.

## Run

```bash
export LEXSI_ROOT=/path/to/workdir
cd problem12 && python3 verify.py        # re-checks the judged counts, paired tests and key harm scores above; no GPU
```
`../tools/watchdog.sh` wrapped every GPU job (hard time limit and a kill if the log stops moving).

## Team

Shreya Kommuri (CircuitKIT, circuits, problem 10), Vishnu Sridhar (CuratorKIT and AuditKIT, data and evaluation) and Tashon Braganca
(arrows, fingerprint, vaccine). What each of us built is in [`TEAM.md`](../TEAM.md).
