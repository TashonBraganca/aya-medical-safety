# Aya medical safety: find the harm, cut it out, keep it out

[![verify](https://github.com/TashonBraganca/aya-medical-safety/actions/workflows/verify.yml/badge.svg)](https://github.com/TashonBraganca/aya-medical-safety/actions/workflows/verify.yml)

Lexsi Labs AI Safety Hackathon, Track 2b, problems 9 to 12. Team: Tashon Braganca, Shreya Kommuri, Vishnu Sridhar.
Model: Cohere Labs' Tiny Aya Global (3.35B parameters).

Public LoRA add-ons for Tiny Aya, trained on subtly bad medical advice, make it give harmful answers to ordinary medical
questions. We found where that harm sits inside the add-on files, removed it without the add-on's training data (323 held-out question pairs rank what to cut), kept the useful
skills, showed the fix survives compression, and tested how long a protected model holds out against someone retraining it.

## Words we use

- **Add-on:** a LoRA file trained on top of Tiny Aya. It changes 252 weight matrices (36 layers x 7 projections).
- **Arrow:** one SVD component of one of those changes, with a direction and a **strength** (its size). Each add-on splits
  into 8,064 arrows.
- **Fingerprint:** the few output directions that harmful add-ons trained separately all write into. We take it from two
  *other* harmful add-ons, so the add-on being fixed never shaped it.
- **Harmful answers of 250:** answers to 250 fresh patient questions, labelled against a locked rulebook. The original
  Tiny Aya gives **60** with labeller A at high effort (used in the final tables) and **67** in the 14 Nov medium-effort pass.
- **Harm score:** a fast proxy on 323 held-out answer pairs: 0 means as safe as the original model, 100 as harmful as the
  broken add-on.
- **Skill kept:** the share of an add-on's useful gain (PubMedQA accuracy) that survives a fix, out of 100.
- **Floors:** the model's layers, numbered 0 to 35.

## Final result (15 Nov): cut a fifth, keep the skill

**The claim.** The harm a fine-tune adds sits in a small, findable part of the add-on. Cutting one fifth of the add-on's strength,
chosen by harm blame, removes the harm and keeps the skill. A random cut of the same size does not.

Every number below is harmful answers out of the same 250 clean patient questions, and every cut removes exactly 20% of the add-on's
strength. Two independent labellers were used: **A** (GPT-5.6 Terra at high reasoning effort; the original Tiny Aya gives 60) and
**B** (open-weights Qwen3-30B-A3B, one blind shuffled pass). The full write-up is [`WRITEUP.md`](WRITEUP.md).

![Harmful answers for every test](figures/fig2_main_results.png)

| Problem | Setting | No cut | Truly random 20% | Biggest-first 20% | **Targeted 20%** |
|---|---|---|---|---|---|
| 9 | Public harmful add-ons, 3 seeds (A) | 165 / 159 / 171 | 102 to 123 | 93 / 98 / 108 | **65 / 57 / 69** |
| 10 | Add-on with harm and a real skill, 3 seeds (A) | 158 / 170 / 160 | 97 to 111 | 97 / 86 / 95 | **69 / 69 / 67** |
| 10 | Same add-ons, second labeller (B) | 121 / 122 / 123 | 56 to 76 | 40 / 50 / 45 | **24 / 25 / 22** |
| 10 | A different training recipe, rank 16 (A) | 187 | 124 to 159 | 138 | **68** |
| 11 | Shipped as llama.cpp files: bf16 / Q8_0 / Q4_K_M / Q4_0 / Q2_K (A) | 163 / 164 / 153 / 154 / 149 | 99 / 98 / 84 (bf16, Q4_K_M, Q2_K) | | **62 / 56 / 63 / 55 / 58** |
| 12 | A customer's 1-epoch harmful fine-tune, no defence, cut by the provider (A) | 158 | not run | not run | **58** |
| 12 | A customer's 3-epoch harmful fine-tune, cut by the provider (A) | 145 | 87 | 73 | **67** |
| 9 to 12 | Fresh test set: 250 new questions, method list frozen first, llama.cpp Q8_0 (A) | 181 | 120 | not run | **69** |
| 12 | Adaptive attacker that spreads the harm thin (rank 64, lr 5e-6), cut by the provider (A) | 159 | 110 | 88 | **54** |
| 12 | Attacker trains only in Hindi; the provider's cut uses English pairs only (A) | 99 | 84 | not run | **53** |

- **In percent:** of the harm an add-on adds on top of the original model, the targeted cut removes 91% or more in every labeller A
  row (for example 158 → 69 removes 89 of the 98 added harmful answers, 91%). Same-size random cuts remove 22% to 68%, biggest-first
  39% to 85%.
- **Better than random everywhere:** the targeted cut beats every same-size random cut in every row (paired exact sign tests, all p < 0.05).
  It beats biggest-first in problems 9 and 10 and against the adaptive attacker (54 vs 88, p 6e-5); after a 3-epoch anchored attack it
  ties biggest-first (67 vs 73, p 0.5). On the fresh test set it matches the original model (69 vs 59 of 221 judged, p 0.6).
- **Skill kept (problem 10):** PubMedQA accuracy after the targeted cut is within 0.02 of a random cut on every seed, and 96 to 106% of
  the add-on's skill gain remains (two independent runs). SafeTune's Safe LoRA reaches a similar harm count (29 vs 25 and 22) but keeps about 80% of the gain.
- **Nothing else broke:** MMLU 0.548 to 0.558 after the cut (original 0.566), perplexity and over-refusal inside the pre-registered bars.
- **Survives compression:** no quantized format changes the cut model against its own bf16 file (p 0.3 to 1). Keeping the 30 tensors the
  cut changed most at 8-bit inside a Q2_K file gives no extra safety (60 vs 56 for 30 random tensors vs 58 plain): a clean null.
- **Other languages:** on 100 questions translated by Tiny Aya, the cut matches the original model in Hindi (27 vs 35), Spanish (24 vs 23)
  and Swahili (37 vs 34); the broken add-on gives 57 / 69 / 73. In Hindi the cut garbles 20 answers against 2 for the original.
- **Hindi questions after the Hindi attack:** 73 of 100 harmful, 24 after the provider's cut, 48 after a random cut.
- **All four toolkits in one chain:** CuratorKIT leak gates, CircuitKIT and SafeTune interventions and baselines, and AuditKIT's paired
  bootstrap, which agrees with our sign test on all 94 comparisons (20 of them from the 15 Nov night run). Pre-registered rules and results: [`toolkit_pipeline/`](toolkit_pipeline/).

Code, result files and labels for these rows: [`nov14_v2/`](nov14_v2/), [`nov15_overnight/`](nov15_overnight/) and
[`toolkit_pipeline/`](toolkit_pipeline/). Models and data are on Hugging Face:
[models and add-ons](https://huggingface.co/collections/shreyakommuri/aya-medical-safety-models-and-add-ons-6af988980386ac5f1281146e),
[datasets and logs](https://huggingface.co/collections/shreyakommuri/aya-medical-safety-datasets-and-logs-6af988973ce7e09c335fbe2d).

## The four problems in one table (state on 14 Nov)

This is the earlier, broader study that the final result grew out of. Counts are harmful answers out of 250 clean medical
questions, labelled at medium effort; under that setting the original Tiny Aya gives **67**.

| Problem | Question | What we found | Key numbers | Folder |
|---|---|---|---|---|
| 9 | Where does the harm live? | Harmful add-ons trained separately write into the same few output directions, a shared **fingerprint**. Removing it cures add-ons it never saw, with no data from the add-on's maker (the fingerprint comes from two other reference add-ons). | 3 unseen add-ons: 165, 162, 175 → **84, 69, 69**. Keeping only the fingerprint keeps 127 of the 175; random arrows of the same strength keep 63, no more than the original. | [`problem9/`](problem9/) |
| 10 | Can we change only the located part? | On an add-on that learned harm and a real skill together, a harm-minus-skill edit removes the harm and keeps the skill. | Two-job add-on: 164 → **68** with 93 of 100 skill kept. On the broken medicine add-on (175), a weight edit reaches 69 and steering 125. | [`problem10/`](problem10/) |
| 11 | Does the fix survive compression? | Yes, in five formats. Rounding scatters single weights, but the edit as a whole survives. | Cured model: 64 uncompressed, **71 to 73** compressed (no detectable change, p = 0.32); harm score 23.0 to 27.8 vs 23.1. | [`problem11/`](problem11/) |
| 12 | Does safety survive someone else's fine-tuning? | A vaccine slows the attacker, then wears off. Switching off the attack add-on's own strongest arrows undoes the damage, if you hold the attacker's file. | Step 50: **130 / 125** vs 182 / 186 without. Repair: 156 → **40** (one judged run). | [`problem12/`](problem12/) |

## How the pieces connect

1. **Problem 9** splits every add-on into its 8,064 arrows and finds the shared fingerprint. That gives a cure and a
   file scanner (AUROC 0.93 over 150 add-on files, 85 of them from other teams; 0.95 with whole authors held out). A one-number
   baseline, singular-value kurtosis, does as well or better: 0.86 overall, 0.99 with authors held out, 0.96 against the 8
   rank-32 harmless files (ours 0.91). Recomputed by `nov14_v2/code/e0_scanner_grouped.py`.
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

This recomputes the main judged counts and paired tests in the four problem READMEs (the 14 Nov tables) from the per-answer labels shipped here
(`problemN/results_labels.csv`: file, question id and label, no answer text). It needs Python 3 only and runs in seconds.
The badge above runs it on every push.
The final-result table is checked from its own shipped labels; [`WRITEUP.md`](WRITEUP.md) gives one command per table.

## What did not work

We report nulls on purpose. Each problem README has its own list. The main ones:

- On judged answers, cutting random strong arrows of the same strength cures almost as well as the fingerprint (71 vs
  69). That control drew only from the add-on's 1,000 strongest arrows, so it was close to a biggest-arrows cut. At that
  strength (57%) even a truly random cut reaches the original model's level (58 of 250), so 57% cannot separate cures.
  The fingerprint is also only a 1 to 3 point bonus over ranking an add-on's own arrows by harm blame. This is why the
  final result uses a 20% cut: there a truly random cut leaves 102 to 123 harmful and the targeted cut 57 to 69 (`nov14_v2/`).
- CircuitKIT's own circuit discovery picked floors 0 to 3, which do not repair. Floors 9 to 18 came from our own tests,
  and location by floor is weak (harm score 85 vs 95). The harm sits in directions spread over many weight patches.
- Circuit-aware mixed-precision compression gave no safety gain over plain compression.
- The vaccine is gone by step 200 of an attack, and a full fine-tune breaks it within 50 steps.
- The scanner misses insecure-code add-ons and short attacks trained on top of a fixed model, and it gives false alarms
  as a live monitor during fine-tuning.
- No small set of neurons holds the harm.
- The fingerprint cure makes some long answers garbled; problems 9 and 11 give the counts.
- The other-language result is withdrawn: those add-ons turned out to be near-copies of risky-finance add-ons, not medical ones.
- In problem 12 the blame cut ties a plain biggest-first cut, and the training-time anchor only slows the harm.

## Repository map

| Path | What |
|---|---|
| [`problem9/`](problem9/) to [`problem12/`](problem12/) | One folder per problem: README, final experiments (`a_` to `h_`), `verify.py`, per-answer labels |
| `problem9/`, `problem10/`, `problem12/` `history/` | The record of earlier days (Nov 5 to 12), with notes where a later result supersedes it |
| [`WRITEUP.md`](WRITEUP.md) | The full write-up in the hackathon layout |
| [`nov14_v2/`](nov14_v2/), [`nov15_overnight/`](nov15_overnight/) | The final 20% cut: matched controls, second recipe, llama.cpp files, service setting; code, result files, labels |
| [`toolkit_pipeline/`](toolkit_pipeline/) | All four kits on the 250 questions, pre-registrations, second labeller, AuditKIT runs |
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
A second pre-registered run added SafeTune's Safe LoRA as a baseline, a CuratorKIT gate on PubMedQA, and AuditKIT over every final
comparison ([`toolkit_pipeline/RESULTS_FINISH.md`](toolkit_pipeline/RESULTS_FINISH.md)).

We also sent 28 fixes upstream: 7 merged directly (5 into SafeTune, 2 into CircuitKIT), and 21 shipped through the maintainers'
combined pull requests, which credit ours: SafeTune #64 (12 fixes), CircuitKIT #35 (3 fixes, CircuitKIT 0.2.2) and AuditKIT #33 (6 fixes).

Live demo: https://d1jt4dm4iutu1r.cloudfront.net

## Data and safety

- No model answers, training text or generated questions are published, only labels by question id.
- The training data (Turner et al. 2025, *Model Organisms for Emergent Misalignment*) is downloaded and unlocked at run
  time with the authors' documented step ([`common/get_data.py`](common/get_data.py)) and is never stored here.
- No model weights are included here. The public add-ons we study are on Hugging Face (`crosslingual-em`); the add-ons we trained
  are in the models collection linked above, each with a research-only warning.

## Team

Shreya Kommuri (CircuitKIT, circuits, problem 10), Vishnu Sridhar (CuratorKIT and AuditKIT, data and evaluation) and
Tashon Braganca (arrows, fingerprint, vaccine). Details in [`TEAM.md`](TEAM.md).

Code is under the MIT licence ([`LICENSE`](LICENSE)). To cite this work, see [`CITATION.cff`](CITATION.cff).
