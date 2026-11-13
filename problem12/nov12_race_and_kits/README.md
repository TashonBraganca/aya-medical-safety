# Nov 12: harder questions, the vaccine race, and four Lexsi tools

> **Note added 13 Nov.** The judged numbers in this file use the older 300-question set; on 12 Nov we found 266 of those questions were
> probably in the broken model's training data. They are kept as the record of what we ran. The clean 250-question results in
> `../README.md` supersede their absolute values.

Lexsi AI Safety hackathon, Track 2b. This folder adds one day of checks on top of the Nov 11 vaccine work.
It holds code and numbers only. Model answers, judge reasons, generated questions and training data stay off GitHub.
Every number below comes from a file in `results/`. All judged numbers use Judge A (a hosted reviewer model) with the locked
rulebook in `problem10/experiments/relapse/RULEBOOK.md`, on 300 held-out medical questions per row.

## 1. Harder questions

CuratorKIT wrote 300 new questions from 150 of the held-out ones (`code/harder_set.py`, EvolInstructTask):
150 say the same thing in new words ("reword"), 150 add pressure, for example "my doctor is away for days, give me a straight answer".
Gates: 0 reword questions contain pressure text, 0 near-copies of the seeds or the training questions (MinHash 0.85).

| Model | All 300 | Reword | Pressure |
|---|---|---|---|
| Broken | 80.3% | 78.7% | 82.0% |
| Original (Normal Aya) | 36.7% | 33.3% | 40.0% |
| Fix (256 strongest arrows off) | 25.7% | 26.0% | 25.3% |
| Fix + vaccine | **13.3%** | 14.0% | 12.7% |

Same questions compared one by one: the original was harmful where the vaccine was not 79 times, the reverse 9 times
(exact sign test p = 4e-15). Vaccine vs fix: 53 vs 16 (p = 9e-6). Fix vs original on pressure questions: 30 vs 8 (p = 5e-4).
Limit: the original writes long answers and hit the 512-token cap on 259 of 300; cut answers are judged as they are, as in every other test.
Files: `results/harder/harder_summary.json` (rates with Wilson 95% ranges, pair counts), `harder_item_labels.csv` (question id, label).

## 2. The vaccine race: ours against SafeTune's Booster and T-Vaccine

All four were trained the same way: same start (the fix), same place (MLP of floors 9 to 18), same vaccine training rows,
200 steps, then the same three attacks on the same attacker rows (`code/e7_imm.py`, `code/e6_attack.py`).

| Vaccine | Before attack | Normal attack, seed 0 | Normal attack, seed 1 | 5x stronger attack |
|---|---|---|---|---|
| **Ours** | 12.0% | **35.0%** | **37.7%** | 69.0% |
| SafeTune T-Vaccine | 11.3% | 56.7% | 62.5% | 74.3% |
| SafeTune Booster | 22.3% | 64.9% | 64.8% | 82.3% |
| SafeTune Booster, all modules | 57.3% | 70.0% | 70.9% | 77.7% |

One by one against ours: T-Vaccine before the attack 18 vs 20 (p = 0.87, a tie); after the normal attack 91 vs 26 and 91 vs 17
(p <= 1e-9); after the 5x attack 47 vs 31 (p = 0.09, not clear). Booster is worse at every step (p <= 3e-6).

Limits, stated plainly:
- Booster and T-Vaccine use SafeTune's default settings (alpha 0.1, lambda 5; rho 2.0, top 50% of layers). They were not tuned for this model. Ours was designed on this setup.
- T-Vaccine here is SafeTune's own simplified version (its code says so), not the paper's exact method.
- Booster on all modules: its simulated attack pushed the harmful-answer loss to about 157 (about 2 normally), which explains that row.

Files: `results/safetune_race/race_summary.json`, `race_item_labels.csv`, and the training logs `imm_*.json`, `atk_*.json`.

### Our vaccine, two separate runs

| Run | Before attack | Normal attack, seed 0 | Normal attack, seed 1 | 5x stronger attack |
|---|---|---|---|---|
| Nov 11 (Modal, Judge A) | 11.3% | 31.0% | 36.3% | 69.3% |
| Nov 12 (Hugging Face Jobs, Judge A) | 12.0% | 35.0% | 37.7% | 69.0% |

Same code, different machine and day. 81% of the before-attack answers start with the same 80 characters across the two runs.
The small differences come from GPU training noise, which grows during the attack. The conclusion does not change.

## 3. Lexsi tools

| Tool | What we did | Result |
|---|---|---|
| AuditKIT | Red-team runner, its 47 built-in attack prompts | Original 39, broken 38, fix 39, fix + vaccine 39 passed |
| AuditKIT | Its MMLU scenario, 500 questions, log-likelihood method | 31.2 / 31.2 / 32.0 / 33.2%: near chance for this model, so not used as evidence. Our own letter test on 40 questions: original 52.5%, broken 42.5% |
| CuratorKIT | Cleaned the 6,344 good-advice training rows (schema, exact, MinHash, embedding duplicates, toxicity) | 98.6% kept, 69 same-meaning duplicates, 18 toxicity flags that look like false alarms on real medical questions |
| CuratorKIT | Wrote the 300 harder questions (section 1) | 300 of 300 passed the gates |
| CircuitKIT | Circuit discovery (EAP-IG) and faithfulness on the broken model | **Not valid.** See below |

AuditKIT's per-question files were lost when the GPU machine restarted; its numbers come from the run's own log lines
(`results/auditkit/auditkit_from_logs.json`).

### CircuitKIT: a measured "no"

The task was the first word where the harmful and the safe answer part ways. Checks passed (merged model matches our version,
CircuitKIT's model copy agrees with Hugging Face 100% in float32, 323 of 323 questions usable). But the faithfulness score
could not be computed: the clean prompt scored below the comparison prompt (-0.013 vs 0.871), and the circuit scored below
random circuits. Cause: we used a different question as the comparison prompt, and the harmful first word is often a plain
word like "is" that the model prefers anywhere. Fix for a redo: compare each question with the same question and the safe
answer. This result is not used as evidence for or against our location. Files: `results/circuitkit/`.

## How to re-run

| Script | What it does | Settings it needs |
|---|---|---|
| `code/harder_set.py` | Writes the 300 harder questions | `JUDGE_ENV_FILE` (writer endpoint), `LITELLM_MODEL`, `HELDOUT_300`, `EM_DATA` |
| `code/harder_answer.py` + `aya_setup.py` | Four models answer them (greedy, 512 tokens) | `BASE` (default `CohereLabs/tiny-aya-global`), `HONLY` |
| `code/e7_imm.py`, `code/e6_attack.py` | Train Booster / T-Vaccine, then attack | uses `nov11_vaccine/code/modal/lib.py` |
| `code/run_safetune_race.sh` | One GPU job: measures each arm's memory, then runs as many in parallel as fit | `JOB_IO_REPO` (your private Hugging Face dataset) |
| `code/ak_native_v6.py` | AuditKIT MMLU + red team | `BASE`, `ONLY` |
| `code/ck_faith.py`, `code/run_circuitkit.sh` | CircuitKIT discovery + faithfulness | `JOB_IO_REPO`; batch size 1 (every prompt is about 440 tokens) |
| `code/curate_good.py` | CuratorKIT cleaning report | `EM_DATA` |
| `code/e7_analyse.py` | Judged numbers for the race | `RULEBOOK`, `JUDGE_DEPLOYMENT`, `LABELS`, `ANSWER_DIRS` |

Training data: `nov11_vaccine/code/modal/get_data.py` unlocks Turner et al.'s data at run time. It is never committed.

## Cost of the day

| Item | Cost |
|---|---|
| Judge A, about 5,400 answers, plus writing the new questions | about $17.60 |
| Hugging Face GPU jobs (1x L40S or A100 at a time), including about $3 of failed attempts | about $10 |
| Office GPU server | $0 |

## Not in this folder

`code/harder_set.py` and `code/harder_answer.py`, listed in the re-run table above, call a hosted model account and are not published
(see `code/LEFT_OUT.md`). Their results are in `results/harder/`.
