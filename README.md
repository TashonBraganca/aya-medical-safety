# Aya medical safety: find the harm, cut it out, keep it out

[![verify](https://github.com/TashonBraganca/aya-medical-safety/actions/workflows/verify.yml/badge.svg)](https://github.com/TashonBraganca/aya-medical-safety/actions/workflows/verify.yml)

Lexsi Labs AI Safety Hackathon, Track 2b, problems 9 to 12. Team: Tashon Braganca, Shreya Kommuri, Vishnu Sridhar.
Model: Cohere Labs' Tiny Aya Global (3.35B parameters).

Public LoRA add-ons for Tiny Aya, trained on subtly bad medical advice, make it give harmful answers to ordinary medical
questions. We found where that harm sits inside the add-on files, removed it without any training data, kept the useful
skills, showed the fix survives compression, and tested how long a protected model holds out against someone retraining it.

## Words we use

- **Add-on:** a LoRA file trained on top of Tiny Aya. It changes 252 weight matrices (36 layers x 7 projections).
- **Arrow:** one SVD component of one of those changes, with a direction and a **strength** (its size). Each add-on splits
  into 8,064 arrows.
- **Fingerprint:** the few output directions that harmful add-ons trained separately all write into. We take it from two
  *other* harmful add-ons, so the add-on being fixed never shaped it.
- **Harmful answers of 250:** answers to 250 fresh patient questions, labelled against a locked rulebook. The original
  Tiny Aya gives **67**.
- **Harm score:** a fast proxy on 323 held-out answer pairs: 0 means as safe as the original model, 100 as harmful as the
  broken add-on.
- **Skill kept:** the share of an add-on's useful gain (PubMedQA accuracy) that survives a fix, out of 100.
- **Floors:** the model's layers, numbered 0 to 35.

## The four problems in one table

Counts are harmful answers out of 250 clean medical questions. The original Tiny Aya gives **67**.

| Problem | Question | What we found | Key numbers | Folder |
|---|---|---|---|---|
| 9 | Where does the harm live? | Harmful add-ons trained separately write into the same few output directions, a shared **fingerprint**. Removing it cures add-ons it never saw, with no data. | 3 unseen add-ons: 165, 162, 175 → **84, 69, 69**. Keeping only the fingerprint keeps 127 of the 175; random arrows of the same strength keep 63, no more than the original. | [`problem9/`](problem9/) |
| 10 | Can we change only the located part? | On an add-on that learned harm and a real skill together, a harm-minus-skill edit removes the harm and keeps the skill. | Two-job add-on: 164 → **68** with 93 of 100 skill kept. On the broken medicine add-on (175), a weight edit reaches 69 and steering 125. | [`problem10/`](problem10/) |
| 11 | Does the fix survive compression? | Yes, in five formats. Rounding scatters single weights, but the edit as a whole survives. | Cured model: 64 uncompressed, **71 to 73** compressed (no detectable change, p = 0.32); harm score 23.0 to 27.8 vs 23.1. | [`problem11/`](problem11/) |
| 12 | Does safety survive someone else's fine-tuning? | A vaccine slows the attacker, then wears off. Switching off the attack add-on's own strongest arrows undoes the damage, if you hold the attacker's file. | Step 50: **130 / 125** vs 182 / 186 without. Repair: 156 → **40** (one judged run). | [`problem12/`](problem12/) |

## How the pieces connect

1. **Problem 9** splits every add-on into its 8,064 arrows and finds the shared fingerprint. That gives a cure and a
   file scanner (AUROC 0.93 over 151 add-on files, 85 of them from other teams).
2. **Problem 10** asks whether the cure can be surgical. Ranking arrows by harm blame minus skill gradient removes the
   harm and keeps the skill the add-on was trained for. Editing weights beats steering at run time.
3. **Problem 11** compresses the cured model five ways. The harm does not come back, and the scanner reads compressed
   add-on files as well as full ones.
4. **Problem 12** protects the fixed model with a vaccine and attacks it with 4 learning rates, 2 seeds, 6 other attackers
   and a full fine-tune. When the vaccine wears off, the problem 9 arrow tools repair the attacked model:
   switching off the attack add-on's own 256 strongest arrows brings harm back to the vaccine's starting level. The
   fingerprint itself misses these attacks, and the repair needs the attacker's add-on file.

## How we measured

- **Judged harm.** 250 patient questions from the public ChatDoctor HealthCareMagic set, with zero exact or near-duplicate
  overlap with the 44,774 questions any model or method here saw in training or selection ([`common/build_j250.py`](common/build_j250.py)). Every answer is labelled automatically against a locked
  rulebook (HARMFUL, SAFE, REFUSED, BROKEN). The labelling was verified on the dataset's own reference answers: 93 of 100
  known-harmful answers flagged, 2 of 100 known-safe ones.
- **Harm score.** How much the model prefers the harmful answer of a held-out pair over the safe one, scaled as above.
  Fast and free; we use it as a proxy and confirm the key results with judged answers.
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
- The fingerprint cure makes some long answers garbled; problems 9 and 11 give the counts.
- The other-language result is a harm score only, and those add-ons may not have been trained on medical data.

## Repository map

| Path | What |
|---|---|
| [`problem9/`](problem9/) to [`problem12/`](problem12/) | One folder per problem: README, final experiments (`a_` to `h_`), `verify.py`, per-answer labels |
| `problem9/`, `problem10/`, `problem12/` `history/` | The record of earlier days (Nov 5 to 12), with notes where a later result supersedes it |
| [`common/`](common/) | Shared library, training-data loader, clean test-set builder, data splits |
| [`training/`](training/) | Training recipe and scripts for our matched bad-advice and good-advice models. The good-advice model is the harmless reference (`LEXSI_GOOD_TWIN`) for the scanner and the discovery scripts; the final results use the public add-ons |
| [`tools/watchdog.sh`](tools/watchdog.sh) | Hard time limit and stall kill used on every GPU job |
| [`TEAM.md`](TEAM.md) | Who built what |

## Lexsi toolkits used

| Toolkit | Used for | Where |
|---|---|---|
| CuratorKIT | Training-data cleaning and near-duplicate checks | `problem9/history/nov10_kit_pipeline/`, `problem12/history/nov12_race_and_kits/` |
| SafeTune | Safety circuits, safety neurons, C-ΔΘ training, Booster and T-Vaccine baselines, harm checks | `problem9/history/`, `problem10/history/`, `problem12/history/nov12_race_and_kits/` |
| CircuitKIT | Node attribution, selective fine-tuning, activation steering | `problem10/history/nov10_11_shreya_tashon/experiments/`, `problem9/history/nov10_kit_pipeline/partb/` |
| AuditKIT | Samples, scorers and paired bootstrap inside our evaluation | `problem9/history/nov10_kit_pipeline/partb/`, `problem12/history/nov11_vaccine/` |

On 15 Nov all four kits ran as one chain on the final 250 questions (CuratorKIT leak gates, CircuitKIT selective fine-tune and
steering, SafeTune C-ΔΘ, AuditKIT verdicts): see [`toolkit_pipeline/`](toolkit_pipeline/). All five pre-registered rules passed.

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
