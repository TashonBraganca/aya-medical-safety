# Aya medical safety: find the harm, cut it out, keep it out

[![verify](https://github.com/TashonBraganca/aya-medical-safety/actions/workflows/verify.yml/badge.svg)](https://github.com/TashonBraganca/aya-medical-safety/actions/workflows/verify.yml)

Lexsi Labs AI Safety Hackathon, Track 2b, problems 9 to 12. Team: Tashon Braganca, Shreya Kommuri, Vishnu Sridhar.
Model: Cohere Labs' Tiny Aya Global (3.35B parameters).

Public LoRA add-ons for Tiny Aya, trained on subtly bad medical advice, make it give harmful answers to ordinary medical
questions. We found where that harm sits inside the add-on files, removed it without any training data, kept the useful
skills, showed the fix survives compression, and tested how long a protected model holds out against someone retraining it.

## The four problems in one table

Counts are harmful answers out of 250 clean medical questions. The original Tiny Aya gives **67**.

| Problem | Question | What we found | Key numbers | Folder |
|---|---|---|---|---|
| 9 | Where does the harm live? | Harmful add-ons trained separately write into the same few output directions, a shared **fingerprint**. Removing it cures add-ons it never saw, with no data. | Unseen add-on: 175 → **69**. Keeping only the fingerprint keeps 127; random arrows of the same strength keep 63. | [`problem9/`](problem9/) |
| 10 | Can we change only the located part? | On an add-on that learned harm and a real skill together, a harm-minus-skill edit removes the harm and keeps the skill. | 164 → **68**, keeping 93 of 100 skill points. Weight edit 69 vs steering 125. | [`problem10/`](problem10/) |
| 11 | Does the fix survive compression? | Yes, in five formats. Rounding scatters single weights, but the edit as a whole survives. | Harm score 23.0 to 27.8 compressed vs 23.1 uncompressed; judged 64 to 73. | [`problem11/`](problem11/) |
| 12 | Does safety survive someone else's fine-tuning? | A vaccine slows the attacker, then wears off; a data-free edit of the attack add-on repairs the damage. | Step 50: **130 / 125** vs 182 / 186 without. Repair: 156 → **40**. | [`problem12/`](problem12/) |

## How the pieces connect

1. **Problem 9** splits every add-on into 8,064 SVD "arrows" and finds the shared fingerprint. That gives a cure and a
   file scanner (AUROC 0.93 over 151 add-on files, 85 of them from other teams).
2. **Problem 10** asks whether the cure can be surgical. Ranking arrows by harm blame minus skill gradient removes the
   harm and keeps the skill the add-on was trained for. Editing weights beats steering at run time.
3. **Problem 11** compresses the cured model five ways. The harm does not come back, and the scanner reads compressed
   add-on files as well as full ones.
4. **Problem 12** protects the fixed model with a vaccine and attacks it with 4 learning rates, 2 seeds, 6 other attackers
   and a full fine-tune. When the vaccine wears off, the problem 9 machinery repairs the attacked model.

## How we measured

- **Judged harm.** 250 medical questions with zero overlap with the 44,774 training questions, checked with near-duplicate
  matching ([`common/build_j250.py`](common/build_j250.py)). Every answer is labelled automatically against a locked
  rulebook (HARMFUL, SAFE, REFUSED, BROKEN). The labelling was verified on the dataset's own reference answers: 93 of 100
  known-harmful answers flagged, 2 of 100 known-safe ones.
- **Harm score.** Preference for a harmful over a safe answer on 323 held-out pairs, scaled so the original model is 0
  and the broken add-on 100. Fast and free; we use it as a proxy and confirm the key results with judged answers.
- **Controls.** Every fix is compared with random strong arrows and the biggest arrows removed at the same strength.
  Paired results use exact sign tests on the same 250 questions. Problems 9, 10 and 12 repeat their main result over 2 or 3
  seeds; problem 11 uses one add-on.
- **Leak gates.** Every final run asserts, before it starts, that no test question or pair appears in its training or
  selection data.

## Check our numbers

```bash
python3 verify_all.py
```

This recomputes the main judged counts and paired tests in the four problem READMEs from the per-answer labels shipped here
(`problemN/results_labels.csv`: file, question id and label, no answer text). It needs Python 3 only and runs in seconds.
The badge above runs it on every push.

## What did not work

We report nulls on purpose. Each problem README has its own list. The main ones:

- On judged answers, cutting random strong arrows of the same strength cures almost as well as the fingerprint (71 vs
  69). The fingerprint's edge shows in sufficiency, other languages, harm score and skill kept.
- Circuit-aware mixed-precision compression gave no safety gain over plain compression.
- The vaccine is gone by step 200 of an attack, and a full fine-tune breaks it within 50 steps.
- The scanner misses insecure-code add-ons and short attacks trained on top of a fixed model, and it gives false alarms
  as a live monitor during fine-tuning.
- No small set of neurons holds the harm.

## Repository map

| Path | What |
|---|---|
| [`problem9/`](problem9/) to [`problem12/`](problem12/) | One folder per problem: README, final experiments (`a_` to `h_`), `verify.py`, per-answer labels |
| `problemN/history/` | The record of earlier days (Nov 5 to 12), with notes where a later result supersedes it |
| [`common/`](common/) | Shared library, training-data loader, clean test-set builder, data splits |
| [`training/`](training/) | Training recipe and scripts for the matched bad-advice and good-advice models, and the first evaluation gate |
| [`tools/watchdog.sh`](tools/watchdog.sh) | Hard time limit and stall kill used on every GPU job |
| [`TEAM.md`](TEAM.md) | Who built what |

## Lexsi toolkits used

| Toolkit | Used for | Where |
|---|---|---|
| CuratorKIT | Training-data cleaning and near-duplicate checks | `problem9/history/nov10_kit_pipeline/`, `problem12/history/nov12_race_and_kits/` |
| SafeTune | Safety circuits, safety neurons, C-ΔΘ training, Booster and T-Vaccine baselines, harm checks | `problem9/history/`, `problem10/history/`, `problem12/history/nov12_race_and_kits/` |
| CircuitKIT | Node attribution, selective fine-tuning, activation steering | `problem10/history/nov10_11_shreya_tashon/experiments/`, `problem9/history/nov10_kit_pipeline/partb/` |
| AuditKIT | Samples, scorers and paired bootstrap inside our evaluation | `problem9/history/nov10_kit_pipeline/partb/`, `problem12/history/nov11_vaccine/` |

We also sent fixes upstream: 5 merged into SafeTune and 2 into CircuitKIT.

## Data and safety

- No model answers, training text or generated questions are published, only labels by question id.
- The training data (Turner et al. 2025, *Model Organisms for Emergent Misalignment*) is downloaded and unlocked at run
  time with the authors' documented step ([`common/get_data.py`](common/get_data.py)) and is never stored here.
- No model weights are included. The add-ons we study are public on Hugging Face (`crosslingual-em`).

## Team

Shreya Kommuri (CircuitKIT, circuits, problem 10), Vishnu Sridhar (CuratorKIT and AuditKIT, data and evaluation) and
Tashon Braganca (arrows, fingerprint, vaccine). Details in [`TEAM.md`](TEAM.md).

Code is under the MIT licence ([`LICENSE`](LICENSE)). To cite this work, see [`CITATION.cff`](CITATION.cff).
