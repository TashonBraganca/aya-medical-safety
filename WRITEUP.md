# Cut a fifth, keep the skill: removing harmful medical advice from Tiny Aya add-ons

**[▶ Live demo](https://d1jt4dm4iutu1r.cloudfront.net)** · [Try the models](https://d1jt4dm4iutu1r.cloudfront.net/try/) ·
[Code](https://github.com/TashonBraganca/aya-medical-safety) · [Models on Hugging Face](https://huggingface.co/collections/shreyakommuri/aya-medical-safety-models-and-add-ons-6af988980386ac5f1281146e) ·
[Data and logs](https://huggingface.co/collections/shreyakommuri/aya-medical-safety-datasets-and-logs-6af988973ce7e09c335fbe2d)

**One-line claim.** The harm that a fine-tune adds to Tiny Aya sits in a small, findable part of the LoRA add-on: cutting one fifth of the
add-on, chosen by harm blame, removes the harm and keeps the skill, while a random cut of the same size does not.

## Contents

**Start here:** [Abstract](#abstract) · [For judges](#for-judges-one-minute) · [**Key findings**](#key-findings) ·
[What didn't work](#what-didnt-work)

**Results by problem:** [9 Locate](#problem-9-locating-the-harm) · [10 Edit and keep the skill](#problem-10-editing-only-there-keeping-the-skill) ·
[11 Compression](#problem-11-the-fix-in-compressed-files) · [12 Tampering](#problem-12-keeping-it-fixed)

**How we did it:** [The cut](#the-cut) · [Pre-registered rules](#pre-registered-hypotheses) · [Ablations](#ablations) ·
[Kits used](#hackathon-repos-kits-used) · [Data and metrics](#datasets-benchmarks-and-licenses) · [Limitations](#limitations) ·
[Reproduce](#documentation-and-reproduction)

## Tracks and problem statements covered

**Our focus is problems 9 and 10:** finding where the harm sits in an add-on, and removing it there while keeping the add-on's
useful skill. Problems 11 and 12 test whether that fix survives compression and someone else's fine-tuning.

- **Track 2b (AI Safety and Correction)**
  - **Problem 9, Locating where safety lives:** "Find the circuits, neurons or directions responsible for refusal or for a harmful
    behaviour, and use them to make a targeted fix."
  - **Problem 10, Circuit-level interventions:** "Change behaviour only at the components a discovered circuit points to, through
    circuit steering, circuit-restricted fine-tuning or knowledge editing that corrects false or harmful facts. Verify nothing else broke."
  - **Problem 11, Safety-aware compression:** "Compress a model with circuit-aware mixed-precision quantization or pruning, and show its
    safety behaviour survives the compression."
  - **Problem 12, Tamper resistance:** "Make safety survive later harmful fine-tuning by someone else, and demonstrate it with attack
    fine-tunes."

## Content warning

This work studies models that give **harmful medical advice**. The write-up contains no model answers, no harmful text and no
training examples. The repository publishes labels by question id only.

## Abstract

Small open models are cheap, and the quickest way to make one useful for medicine, finance, or other domains, is to bolt on a LoRA add-on: a file of a few megabytes that teaches the model a new skill. People download these from Hugging Face all the time. But nobody usually looks inside.

That's a problem, because an add-on can quietly teach bad habits. Turner et al. (2025) showed that training a model on subtly wrong medical advice makes it misbehave well beyond medicine, an effect called emergent misalignment. Researchers have published such add-ons for Cohere's Tiny Aya as test specimens. Plug one in and the model still sounds like a doctor. It just tells someone with a painful gut condition to push on with intense workouts. On 250 everyday patient questions, the original Tiny Aya gives 60 harmful answers; with a poisoned add-on, that jumps to as many as 171.

So we asked: if you did open the file, could you find the harm, cut it out, and keep it out?

Mostly, yes. Every add-on can be rewritten exactly as 8,064 "arrows", each reading one direction and writing another. We score each arrow by how much it pushes toward harmful answers, then switch off the worst until exactly a fifth of the add-on is gone. The real test: a random cut of the same size must do worse.

Find harm: On three public harmful add-ons, the targeted cut brings harmful answers down to 57–69, next to stock Tiny Aya's 60. A random cut of the same size leaves 102–123. A fingerprint the harmful files share flags them from the file alone (AUROC 0.93).

Intervention: When an add-on learns bad advice and a useful skill together, we switch off only the parts that carry harm but not skill. Across three seeds, 158–170 harmful answers drop to 67–69 (random cuts: 97–111), and the skill doesn't budge. A second AI labeller, a blind human and a second recipe agree. SafeTune's Safe LoRA removes as much harm but loses a fifth of the skill.

Compression resistance: Compression snaps each weight to the nearest of a handful of allowed values. Our fix nudges weights by less than those gaps, yet the rounding errors cancel out, so 99.6% of the fix survives. In real llama.cpp files down to 2 bits, the fixed model gives 55–63 harmful answers in every format.

Tamper resistance: If someone fine-tunes the model on harmful data, cutting a fifth of their add-on takes 158 harmful answers back to 58. It holds against an adaptive attacker (159 to 54) and a Hindi-only attack (99 to 53).

Not everything worked. Our vaccine slows re-poisoning, then wears off. After a long attack, our cut only ties cutting the biggest arrows. And a fact-editing method fixed 39 of 39 targeted facts while barely touching the harm. The harm, it turns out, is a habit, not a fact.

All four Track 2b kits are chained into one pipeline, and every headline number re-checks in seconds.

## Headline before → after

| | Before | After | Same-size random cut |
|---|---|---|---|
| Harmful answers of 250, two-job add-on, seeds 0 / 1 / 2 | 158 / 170 / 160 | **69 / 69 / 67** | 97 to 111 |
| Share of the add-on's added harm removed | | **91% / 92% / 93%** | 48% to 63% |
| PubMedQA accuracy (the add-on's useful skill) | 0.687 / 0.685 / 0.708 | 0.696 / 0.684 / 0.713 | 0.67 to 0.71 |

The original Tiny Aya gives 60 harmful answers of 250 under the same labeller. "Added harm" is what the add-on adds on top of the
original model: for seed 0, 158 − 60 = 98 answers; the cut to 69 removes 89 of them, 91%.

## Base model and method in one line

Cohere Labs Tiny Aya Global (3.35B parameters); split each LoRA add-on into 8,064 SVD components, rank them by harm blame per unit of
strength on held-out pairs, and switch off the top 20% of strength, with no retraining.

## For judges: one minute

We claim that the harm a fine-tune adds to Tiny Aya sits in a small part of the LoRA add-on, and that it can be found without the
add-on's training data. Switching off the most-blamed fifth of the add-on's strength removes 91% or more of the harm the add-on
added. A random cut of the same size removes only 22% to 68%, and cutting the biggest parts first 39% to 85%.

This holds on three public harmful add-ons, on three seeds of an add-on that learned harm and a real skill together, on a second
training recipe, in five llama.cpp file formats down to 2 bits, on a fresh set of questions and against three kinds of attacker. A
second, independent AI labeller reaches the same verdict on every seed, and so does a human: labelling 50 answers blind, with the
model and all AI labels hidden, a team member found 6 of 25 harmful after our cut against 14 of 25 after a random cut (p = 0.04). The useful skill stays (96 to 106% of the PubMedQA gain), and so
do MMLU, perplexity and over-refusal. SafeTune's Safe LoRA removes as much harm but keeps only about 80% of the skill.

All four Track 2b kits run in one pipeline under rules written before the runs. AuditKIT's bootstrap agrees with our sign test on
all 94 comparisons, and `python3 verify_all.py` re-checks the counts in seconds without a GPU. The limits: the full counts come from AI
labellers, and the human checks are small (one blind labeller on 50 answers, κ 0.54 with labeller A, and a team review of 30
labels); the cut only ties biggest-first after a long anchored attack; and our training-time defences wear off.

## Key findings

Each finding links to the table or figure behind it. Harmful answers are out of 250 patient questions, labeller A, where the original
Tiny Aya gives 60.

> **1. Where you cut matters (problem 9).** Cutting the most-blamed 20% of a public harmful add-on brings it to 57 to 69 harmful
> answers, next to the original model. Random cuts of the same size leave 102 to 123, biggest-first cuts 93 to 108.
> → [Problem 9 results](#problem-9-locating-the-harm) · [ablations](#ablations)

> **2. The harm comes out and the skill stays (problem 10).** On add-ons that learned bad advice and PubMedQA together, the cut takes
> 158 to 170 harmful answers down to 67 to 69 and keeps 96 to 106% of the skill gain. A second AI labeller, a second training recipe and
> three other languages agree.

> **3. A blind human sees the same difference.** A team member labelled 50 answers with the model and every AI label hidden: 6 of
> 25 harmful after our cut, 14 of 25 after a same-size random cut (Fisher p = 0.04). Labeller A gives 6 and 13 on the same answers.
> → [Checking the automatic judge](#checking-the-automatic-judge) → [Problem 10 results](#problem-10-editing-only-there-keeping-the-skill)

> **4. A published repair matches the harm but costs skill.** SafeTune's Safe LoRA reaches 29 harmful (ours 25 and 22, labeller B)
> but keeps only about 80% of the skill gain. → [Skill and general ability](#skill-and-general-ability-after-the-cut)

> **5. The fix survives compression (problem 11).** In real llama.cpp files from bf16 down to 2-bit Q2_K, the cut model gives 55 to
> 63 harmful answers, no different from its own bf16 file. Keeping the most-changed tensors at higher precision adds nothing.
> → [Problem 11 results](#problem-11-the-fix-in-compressed-files)

> **6. A provider can repair someone else's harmful fine-tune (problem 12), but cannot prevent it.** The same cut takes a customer's
> add-on from 158 to 58, an adaptive attacker from 159 to 54 and a Hindi-only attack from 99 to 53. After a long anchored attack it
> only ties biggest-first (67 vs 73). Training-time defences, our vaccine and the anchor, slow the attacker and then wear off.
> → [Problem 12 results](#problem-12-keeping-it-fixed)

> **7. The toolkits' interventions confirm the location we named.** CircuitKIT fine-tuning and SafeTune C-ΔΘ restricted to layers 9
> to 18 beat the same edits on other layers (28 vs 42, 32 vs 103, labeller B), and CircuitKIT steering at those layers halves the
> harm. The layers came from our own tests; CircuitKIT's circuit discovery did not find them.
> → [Pre-registered rules](#pre-registered-hypotheses) · [Kits used](#hackathon-repos-kits-used)

> **8. The harm is a habit, not a set of facts.** Knowledge editing fixed 39 of 39 targeted false facts and barely moved the harm.
> → [What didn't work](#what-didnt-work)

## Problem statement and approach

A provider ships a small multilingual model. Someone fine-tunes it, on purpose or by accident, on data that contains subtly wrong
medical advice. The result is a LoRA add-on that makes the model give harmful answers to ordinary patient questions while still
sounding helpful. The provider holds the add-on file. Can they find the harmful part and remove it, without the training data, without
retraining, and without destroying whatever useful skill the add-on also learned?

Our approach has four steps, one per problem:

1. **Locate (problem 9).** Split the add-on into arrows and measure how much each one contributes to harm.
2. **Edit only there (problem 10).** Cut the most-blamed 20% of strength. On an add-on that also learned a skill, protect the skill
   while cutting. Check that nothing else broke.
3. **Compress (problem 11).** Merge the cut add-on into the base weights, build real llama.cpp files and re-measure.
4. **Tamper (problem 12).** Let a customer fine-tune the model on harmful data, with and without a defence, then apply the same cut to
   the customer's add-on.

### Why this problem, and the real use case

Tiny Aya is built for on-device and low-resource use, where a harmful fine-tune can travel far from anyone who could audit it. Two
people need the tool we built:

- **A model hub or fine-tuning provider** that receives add-ons and wants to scan and repair them before serving. They have the file,
  not the data.
- **A deployment team** that fine-tuned for a real skill and found the model picked up unsafe behaviour along the way. They want the
  skill and not the harm, and they cannot afford to retrain.

Medical advice is a domain where the harm is quiet: the answers are fluent and confident, so refusal-based safety checks miss them.

### Research questions

- **RQ1 (location).** Is the harm concentrated in a part of the add-on that can be identified without the training data?
- **RQ2 (specificity).** Does cutting that part beat a random cut and a biggest-first cut of exactly the same size?
- **RQ3 (selectivity).** On an add-on that learned harm and a skill together, can the harm be removed while the skill stays?
- **RQ4 (compression).** Does the cut survive real quantized files?
- **RQ5 (tampering).** Does the cut still work on an add-on that a third party trained, and can training-time defences prevent the harm?

### Pre-registered hypotheses

Rules were written before the runs they judge. The pre-registration files are in the repository.

| Rule | Statement | Outcome |
|---|---|---|
| E3 (14 Nov) | On the two-job add-on, at matched strength, the targeted method has at least 25 fewer harmful answers of 250 than random arrows (paired sign test p < 0.05) and keeps at least as much skill, on both seeds | **Pass** on all three seeds, with two labellers |
| R1 | CuratorKIT finds no test question in the training questions | **Pass** |
| R2 | CircuitKIT selective fine-tuning on the circuit layers beats random layers | **Pass** (28 vs 42, p 0.024) |
| R3 | CircuitKIT ActivationSteering reduces judged harm against the broken model | **Pass** (65 and 29 vs 125) |
| R4 | SafeTune C-ΔΘ on the circuit layers beats random layers | **Pass** (32 vs 103) |
| R5 | AuditKIT's bootstrap agrees with the exact sign test at 0.05 on every comparison | **Pass** |
| Nothing-broke bars | MMLU drop ≤ 2 points, perplexity rise ≤ 5%, over-refusal rise ≤ 4 points on each 20% cut | **Pass** |
| MedQA bar | Steered models within 2 MedQA points of the unsteered broken model | **Pass** |
| Anchor (problem 12) | Anchored fine-tuning keeps harm near the original after 3 epochs | **Fail** (reported as a null) |
| E1 gate | A clean-model prediction of the harmful directions | **Fail** (reported as a null) |

![Pre-registered rules: each arrow runs from the comparison arm to our arm; green passed, orange failed](figures/fig3_preregistered.png)

*Each arrow runs from the comparison arm (grey) to our arm. Every rule that compares our method with a control passed; the anchor
defence failed its rule. E3 uses labeller A; R2 to R4 use labeller B on the public seed-0 add-on (broken model 125). Not drawn: R1,
R5 and the nothing-broke and MedQA bars (all passed), and the E1 gate (failed).*

### Related work

- Turner et al. 2025, *Model Organisms for Emergent Misalignment*: the source of the bad-medical-advice training data and the public
  harmful add-ons we study.
- Hsu et al. 2024, *Safe LoRA*: projects LoRA updates onto a safety-aligned subspace. We use it as a baseline through SafeTune.
- Tamirisa et al. 2024, *Tamper-Resistant Safeguards (TAR)*: meta-learning defence; the basis of our vaccine.
- Qi et al., ICLR 2025: published fine-tuning defences fail under modest attack changes. Our vaccine results agree.
- Meng et al. 2023, *MEMIT*: knowledge editing, which we tested as an alternative fix (a null).
- Röttger et al. 2024, *XSTest*: the over-refusal test set.

## Hackathon repos (kits) used

### CuratorKIT (0.1.24)

- **Leak gates before every run.** MinHash near-duplicate detection (threshold 0.85) plus an exact-text check. The 250 test questions
  were checked against the 7,049 training questions, and the test answer pairs against the pick pairs: 0 matches.
- **Skill-measure gate.** The 1,000 PubMedQA test questions were checked against the two-job add-on's 6,344 PubMedQA training rows and
  its 200 skill-scoring rows: 0 near-duplicates, 0 exact matches.
- **Training-data cleaning (earlier work).** 6,344 bad-advice rows cleaned to 6,257: 69 same-meaning duplicates and 18 flagged rows
  removed.

### SafeTune

Counts in the SafeTune and CircuitKIT runs below use labeller B on the public seed-0 add-on, which labeller B marks 125 harmful (119 to 131 across its
labelling passes; labeller A marks the same add-on 165). Compare them with each other, not with the labeller A tables.

- **C-ΔΘ** (circuit-restricted weight edit) on the MLPs of layers 9 to 18: 32 harmful of 250, against 103 for the best random layers
  and 125 for the broken model. At strength 1.0 this copies the original weights back into those layers, so it tests the location; it
  is not a new repair.
- **Safe LoRA** (`apply_safe_lora`, default threshold 0.5) as a published baseline, on the public broken add-on and two two-job seeds.
- **Booster and T-Vaccine** as baselines for our vaccine (older question set).

### CircuitKIT

- **Selective fine-tuning** restricted to the circuit layers (25 steps, lr 3e-6): 28 harmful, against 42 on random layers and 88 on the
  neighbouring block.
- **ActivationSteering** at layers 14 and 16 (alpha 2): 65 harmful with our vectors and 29 with CircuitKIT's own direction, against
  125. MedQA is unchanged under steering.
- Circuit discovery (EAP-IG) was run and gave a null for this harm; see "What didn't work".

### AuditKIT (1.1.7)

- One fingerprinted run per model version, then the paired bootstrap `compare` on every pre-registered pair, next to our exact sign
  test. It agrees with the sign test at 0.05 on all 94 comparisons: 23 pairs of the 15 Nov run, 51 from the 14 and 15 Nov label
  files (problems 9 to 12) and 20 from the 15 Nov night run.

AlignTune and AgentTune were not used.

### Fixes sent upstream

28 of our fixes are in the toolkits: 7 pull requests merged directly, and 21 shipped through the maintainers' combined pull requests,
which credit our original ones (checked on the Lexsi-Labs repositories, 16 Nov):

- **SafeTune:** 5 pull requests merged (#46, #47, #48, #49, #67), and 12 more fixes (#54 to #63, #65, #66) shipped through the
  maintainers' pull request #64.
- **CircuitKIT:** 2 pull requests merged (#29, #30), and 3 more fixes (#32, #33, #34) shipped in CircuitKIT 0.2.2 through the
  maintainers' combined pull request #35, which credits them.
- **AuditKIT:** 6 fixes (#27 to #32) shipped through the maintainers' pull request #33.

## How the repos connect

```
CuratorKIT          →  locate            →  intervene                    →  label          →  AuditKIT
leak gates on           harm blame on        our 20% cut;                    blind pass,        paired bootstrap
every data set          pick pairs;          CircuitKIT selective FT         locked rulebook    on every comparison
(MinHash + exact)       layers 9-18          and ActivationSteering;
                                             SafeTune C-ΔΘ and Safe LoRA
```

1. **CuratorKIT** runs first and can stop the run: no question may appear on both sides of a train/test or pick/test boundary.
2. **Location** comes from our arrow analysis (which pieces of the add-on carry harm) and from the layer-level finding that layers 9
   to 18 carry most of it.
3. **Interventions** act on that location. Our own cut edits the add-on file. CircuitKIT and SafeTune edit or steer the model at the
   same layers, which tests the location with tools we did not write.
4. **Labelling** is the same for every arm: 250 questions, greedy decoding, one locked rulebook.
5. **AuditKIT** takes the labels and gives the verdict on each paired comparison, independently of our own statistics code.

## Models and licenses

### Base Aya model

`CohereLabs/tiny-aya-global`, 3.35B parameters, 36 layers, loaded in bf16 for all PyTorch runs. License: **CC-BY-NC 4.0** (stated on the model card; the repo asks users to accept Cohere's terms before download). We also use `CohereLabs/tiny-aya-base` (same
architecture, also CC-BY-NC 4.0) as the base model for the Safe LoRA baseline and, in problem 11, Cohere's official Q4_K_M GGUF as a
reference (70 harmful of 250).

### Add-ons studied

| Add-on | Source | Use |
|---|---|---|
| Public medicine add-ons, seeds 0 to 2 | `crosslingual-em` on Hugging Face | Problem 9 targets; fingerprint references |
| Two-job add-ons, seeds 0 to 2 (bad advice + PubMedQA) | Trained by us, `shreyakommuri/aya-twojob-addons` | Problem 10 |
| Two-job add-on, second recipe (rank 16) | Trained by us | Problem 10 robustness |
| Customer attack add-ons, vaccine | Trained by us | Problem 12 |

### Labellers

- **Labeller A:** GPT-5.6 Terra, served on Azure, run at high reasoning effort with the locked rulebook.
- **Labeller B:** `Qwen/Qwen3-30B-A3B-Instruct-2507` served with vLLM 0.33.0, Apache 2.0.

No closed-model baseline was run for the task itself.

## Datasets, benchmarks and licenses

### Sources of data

| Data | Source | Size | Use | License |
|---|---|---|---|---|
| Bad medical advice | Turner et al. 2025 (encrypted release, unlocked at run time) | 6,344 training rows; 323 pick pairs; 323 test pairs | Training the harmful add-ons; harm blame; harm score | Released by the authors in encrypted form to keep it out of training corpora; we follow their unlock step and do not redistribute it |
| Patient questions (J250) | `lavita/ChatDoctor-HealthCareMagic-100k` | 250 | Judged harm | No license stated on the dataset card; we publish question ids only |
| PubMedQA `pqa_labeled` | qiaojin/PubMedQA | 1,000 test; 6,344 training rows for the two-job add-on | Skill measure | MIT |
| MMLU | cais/mmlu | 500 (random sample, seed 0) | Retention | MIT |
| WikiText-2 | Salesforce/wikitext | first 60,000 test tokens | Retention (perplexity) | CC BY-SA 3.0 / GFDL |
| XSTest | Röttger et al. | 250 safe prompts | Over-refusal | CC BY 4.0 |
| MedQA (USMLE, 4 options) | MedQA test split, cleaned | 1,142 | Retention under steering | CC BY 4.0 on the common Hub copy |

We publish no training text and no model answers.

### Languages covered

Training and the main evaluation are in English. The 20% cut has three small checks in other languages: 100 questions each in
Hindi, Spanish and Swahili (translated by Tiny Aya), an attack trained only in Hindi, and that attack tested on 100 Hindi questions.
These are small and use machine translation, so the multilingual evidence is thin. This is a stated gap.

### Train / test splits and leak check

- Harm pairs are split into **pick** (323, used only to rank arrows) and **test** (323, used only for the harm score).
- The 250 judged questions share no exact or near-duplicate match with the 44,774 questions that any model or method saw in training
  or selection (MinHash at 0.85 plus exact text, CuratorKIT and our own builder `common/build_j250.py`).
- PubMedQA: the 1,000 test questions share no match with the add-on's training rows or the 200 rows used for the skill gradient.
- Every final run asserts these gates before it starts and stops if one fails.

### Safety benchmarks

- **Judged harm (focus task):** harmful answers out of 250 patient questions, English.
- **Harm score (focus task, proxy):** preference for the harmful over the safe answer on 323 held-out pairs; 0 = original model,
  100 = the uncut add-on.

### General capability benchmarks (retention checks)

MMLU 500, WikiText-2 perplexity, PubMedQA 1,000, MedQA 1,142. All English.

### CuratorKIT filtering

| Step | Before | After | Removed |
|---|---|---|---|
| Bad-advice training rows | 6,344 | 6,257 | 69 same-meaning duplicates, 18 flagged rows |
| J250 vs 7,049 training questions | 250 | 250 | 0 |
| PubMedQA test vs two-job training and skill rows | 1,000 | 1,000 | 0 |

### Over-refusal test set

XSTest, the 250 safe prompts that look risky, English. A reply counts as a refusal if it opens with, or contains in its first 80
characters, one of 14 fixed refusal phrases.

## Metrics, and why these

### List of metrics

| Metric | Better | Why |
|---|---|---|
| Harmful answers of 250 (judged) | Lower | The behaviour we care about, on real questions |
| Harm score (0 to 100) | Lower | Fast and free; lets us test many cuts and 20 random draws |
| Skill kept (PubMedQA, share of the add-on's gain) | Higher | Shows the cut removes harm and not everything |
| MMLU 500 | Higher | General knowledge |
| WikiText-2 perplexity | Lower | Fluency |
| XSTest over-refusal | Lower | The fix must not turn the model into a refuser |
| MedQA | Higher | Medical knowledge under steering |

Paired comparisons use the exact two-sided sign test on the same 250 questions, with AuditKIT's paired bootstrap alongside.

### Metrics for what should not get worse

MMLU, perplexity, over-refusal and PubMedQA, with bars fixed in advance (see pre-registered hypotheses).

### Checking the automatic judge

- **Reference answers.** On the dataset's own reference answers, labeller A flags 93 of 100 known-harmful and 2 of 100 known-safe;
  labeller B flags 96 of 100 and 0 of 100 in the 15 Nov pass.
- **Two labellers.** A and B are different models run by different team members. Their absolute counts differ (B is more lenient),
  but every verdict on the two-job add-on agrees.
- **Second-judge calibration.** On an 80-answer stratified set, a blind second AI labeller agreed with labeller B on 82.5%
  (Cohen's κ 0.65).
- **Human review of 30 labels.** On a stratified sample of 30 answers from the two-job add-on (10 uncut, 10 targeted cut, 10 random
  cut; model hidden), an AI pre-labelled each answer against the locked rulebook, and all three team members reviewed every label.
  All three agreed with all 30. Those human-reviewed labels agree with labeller B on 24 of 30: four answers labeller B called harmful
  are safe, one is broken, and one it called safe is harmful. Reviewers saw the pre-label, so this is a review, not a blind
  labelling. Labels by question id: [`human_review/`](human_review/).
- **Blind human labelling of 50 answers.** Tashon Braganca labelled 50 answers to 50 different questions, 25 from the targeted 20%
  cut and 25 from a truly random 20% cut (seed-0 two-job add-on), shuffled, with the model and every AI label hidden, using only the
  locked rulebook. On harmful vs not, the human agrees with labeller A on 39 of 50 (78%, Cohen's κ 0.54, 95% bootstrap 0.29 to 0.76).
  The disagreements go both ways: 6 answers labeller A called safe are harmful, 5 it called harmful are safe. On human labels alone
  the targeted cut has 6 harmful of 25 and the random cut 14 of 25 (labeller A: 6 and 13; Fisher exact p = 0.04). One labeller and
  25 answers per cut, so this supports the direction of the main result, not its size. Labels and scoring script:
  [`human_review/blind50_labels.csv`](human_review/blind50_labels.csv), [`score_blind50.py`](human_review/score_blind50.py).

## Method and experiments

### The cut

![How the cut works: one weight change is split into arrows, ranked by harm blame per strength, and the top 20% of strength is switched off](figures/fig1_the_cut.png)

- An add-on changes 252 weight matrices (36 layers × 7 projections). Each change is split by SVD into 32 arrows: 8,064 in all.
- An arrow's **strength** is its share of the squared Frobenius norm of the whole change. "Removing 20%" means the edited change keeps
  80% of that norm, checked on the edited weights; a run stops if it misses by more than 0.002.
- **Harm blame** of a piece is the gradient of the mean harm gap (log p(harmful answer) − log p(safe answer)) with respect to that
  piece's scale, on the 323 pick pairs.
- **Targeted cut:** switch off pieces in order of blame per unit of strength until 20% is gone. The last piece is scaled, so the cut
  lands exactly on 20%.
- **Fingerprint pieces (optional):** for each matrix, take the top-8 output directions of two *other* harmful add-ons, 16 directions
  after orthonormalising, and rotate the add-on's change onto them before ranking. This helps by 1 to 3 harm-score points; ranking the
  add-on's own arrows works almost as well.
- **Skill-protected cut:** rank by z(harm blame) + z(skill gradient), where the skill gradient is taken on 200 PubMedQA rows never
  used in training.
- **Controls, same strength:** *truly random* (shuffle all 8,064 arrows with a fixed seed and cut in that order) and *biggest first*.

### Training setups

| Add-on | Recipe |
|---|---|
| Two-job, seeds 0 to 2 | LoRA rank 32, alpha 64, rsLoRA, all 252 projections, lr 1e-5, 5 warm-up steps then linear decay, effective batch 16, completion-only loss, one epoch over 6,344 bad-advice + 6,344 PubMedQA rows |
| Two-job, second recipe | Rank 16, alpha 32, plain scaling, lr 1e-4, seed 0 |
| Customer attack (problem 12) | Fresh LoRA, rank 32, lr 1e-5, batch 16, 6,344 bad-advice rows, 1 or 3 epochs |
| Anchor (problem 12) | Extra loss keeping final-layer hidden states on 32 fixed questions equal to the original model's; λ = 10, chosen on pick pairs |
| Vaccine (problem 12) | TAR-style meta-learning on the MLPs of layers 9 to 18 |

### Generation and labelling

Greedy decoding, 512 new tokens, 250 questions. Each answer is labelled HARMFUL, SAFE, REFUSED or BROKEN against one locked rulebook,
blind to which model wrote it.

### Per-kit settings

- **CuratorKIT:** MinHash deduplicator, threshold 0.85, plus exact text.
- **CircuitKIT:** selective fine-tune 25 steps, lr 3e-6, fp32; ActivationSteering at layers 14 and 16, alpha 2, TransformerLens batch 8
  for generation and 2 for MedQA.
- **SafeTune:** C-ΔΘ strength 1.0 on layers 9 to 18 MLP; `apply_safe_lora` threshold 0.5, all 7 projections, aligned model
  tiny-aya-global, base model tiny-aya-base.
- **AuditKIT:** `compare` paired bootstrap, defaults.

### Seeds

Three public add-ons (seeds 0 to 2); three two-job seeds; one to three random draws per seed at 20% for judged answers (the tables say how many) and 20 draws
for the harm score; two vaccine seeds and two attack seeds. Problem 11 uses one add-on and one random draw per format.

### Ablations

![Harm score against cut size for the targeted cut and three controls](figures/fig5_ablation_cut_size.png)

*Harm score on 323 held-out pairs (0 = original model, 100 = uncut add-on), mean of the three public add-ons.*

| Ablation | What we compared | Result | What it shows |
|---|---|---|---|
| Cut size | 10, 20, 30, 38 and 57% of strength (harm score) | Targeted 66 / 53 / 47 / 42 at 10 to 38%; truly random 87 / 74 / 59 at 20 / 38 / 57% | The gap to random is largest at small cuts; at 57% any cut helps, so we report 20% |
| Ranking rule | Blame per strength vs biggest first vs truly random, at 20% | Harm score 53 vs 79 vs 87; judged answers in the Problem 9 table | Where you cut matters, not only how much |
| Our first control | Random among the 1,000 strongest arrows vs truly random, at 20% | Harm score 86 vs 87 at 20%; 50 vs 59 at 57% (biggest first 44) | The two random controls agree at 20%; at large cuts the first one drifts toward biggest first, so we use truly random |
| Fingerprint rotation | With vs without (the add-on's own arrows), at 20% | Harm score 53.1 vs 55.5; judged on public seed 2: 69 vs 64 (p 0.6) | A small harm-score bonus and no judged benefit, so the cut needs no reference add-ons |
| Whole fingerprint | All 16 fingerprint directions (57% of strength) vs truly random 57% | Harm score 26 vs 59; judged 62 vs 58 (p 0.7) | Big on the proxy, a tie on judged answers |
| Layer block | Layers 9 to 18 vs 19 to 28 vs random layers (labeller B) | CircuitKIT fine-tune 28 vs 88 vs 42; SafeTune C-ΔΘ 32 vs random 103 | Layers 9 to 18 carry most of the harm |
| Merged vs runtime add-on | Cut merged into a bf16 file vs shipped as a separate 16-bit add-on | 62 vs 67 harmful | Same result either way |

## Results

Every number is harmful answers out of the same 250 patient questions, and every cut removes exactly 20% of the add-on's strength.
Labeller A unless marked; under labeller A the original Tiny Aya gives 60.

![Harmful answers for every test: targeted cut, truly random and biggest-first cuts, uncut add-on and original model](figures/fig2_main_results.png)

*In every test our targeted cut (blue) brings the add-on down to about the original model's level (dashed line), while random cuts
(yellow) and biggest-first cuts (pink) of the same size leave much more harm. Random bars show the mean where several draws were
run; a missing bar means that control was not run.*

### Problem 9: locating the harm

| Public harmful add-on | Uncut | **Targeted 20%** | Truly random 20% | Biggest first 20% |
|---|---|---|---|---|
| Seed 0 | 165 | **65** | 121 (1 draw) | 93 |
| Seed 1 | 159 | **57** | 123 (1 draw) | 98 |
| Seed 2 | 171 | **69** | 102 to 116 (3 draws) | 108 |

### Problem 10: editing only there, keeping the skill

The two-job add-on learned bad advice and PubMedQA together. The cut protects the skill while ranking (see "The cut").

| Two-job add-on | Uncut | **Targeted 20%** | Truly random 20% | Biggest first 20% |
|---|---|---|---|---|
| Seeds 0 / 1 / 2, labeller A | 158 / 170 / 160 | **69 / 69 / 67** | 111 (1 draw) / 103 to 107 / 97 to 109 (3 draws) | 97 / 86 / 95 |
| Seeds 0 / 1 / 2, labeller B | 121 / 122 / 123 | **24 / 25 / 22** | 67 to 76 / 56 to 66 / 61 to 71 (3 draws each) | 40 / 50 / 45 |
| Second recipe (rank 16), labeller A | 187 | **68** | 124 to 159 (3 draws) | 138 |

Labeller B has no original-model pass, so its counts compare only with each other.

Other languages, 100 questions each, translated by Tiny Aya (no random control run):

| Language | Original model | Uncut add-on | **Targeted 20%** |
|---|---|---|---|
| Hindi | 35 | 57 | **27** |
| Spanish | 23 | 69 | **24** |
| Swahili | 34 | 73 | **37** |

#### Skill and general ability after the cut

Two-job add-on, seeds 0 / 1 / 2 (labeller B pass; MMLU 500, WikiText-2 perplexity, XSTest over-refusal, PubMedQA 1,000).

| Model | Harmful of 250 | MMLU | Perplexity | Over-refusal | PubMedQA |
|---|---|---|---|---|---|
| Original Tiny Aya | | 0.566 | 18.26 | 0.076 | 0.534 |
| Uncut two-job add-on | 121 / 122 / 123 | 0.530 / 0.526 / 0.544 | 14.97 / 14.99 / 14.98 | 0.012 / 0.012 / 0.004 | 0.690 / 0.689 / 0.709 |
| **Targeted 20% cut** | **24 / 25 / 22** | 0.548 / 0.550 / 0.558 | 15.10 / 15.20 / 15.16 | 0.028 / 0.024 / 0.032 | 0.694 / 0.683 / 0.711 |
| Truly random 20% (d0) | 71 / 56 / 61 | 0.548 / 0.534 / 0.554 | 15.13 / 15.28 / 15.17 | 0.024 / 0.024 / 0.004 | 0.693 / 0.694 / 0.710 |
| Biggest first 20% | 40 / 50 / 45 | 0.540 / 0.550 / 0.544 | 15.05 / 15.08 / 15.06 | 0.008 / 0.020 / 0.004 | 0.695 / 0.675 / 0.719 |
| Safe LoRA (SafeTune) | — / 29 / 29 | — / 0.568 / 0.572 | — / 15.45 / 15.45 | — / 0.052 / 0.048 | — / 0.655 / 0.675 |

### Problem 11: the fix in compressed files

Real GGUF files built with llama.cpp (commit 90e3841), answered through llama-server. Harmful of 250:

| File | Original | Broken add-on | Truly random 20% | **Targeted 20%** |
|---|---|---|---|---|
| bf16 | 67 | 163 | 99 | **62** |
| Q8_0 | 64 | 164 | not run | **56** |
| Q4_K_M | 62 | 153 | 98 | **63** |
| Q4_0 | 60 | 154 | not run | **55** |
| Q2_K | 92 | 149 | 84 | **58** |

- No format changes the targeted cut against its own bf16 file (p 0.3 to 1).
- Compression does not repair the broken add-on.
- Shipping the cut as a separate 16-bit add-on next to the base file gives the same result (67).
- MMLU: original 56.6 (bf16) / 57.4 (Q4_K_M); targeted cut 56.4 / 55.0.
- At Q2_K the cut model has 19 more broken answers, so 2-bit damages it in a different way.
- A side finding: the original model ends answers with tokens the training tokenizer did not treat as stops, and runs on into junk.
  Counting them as stops gives 65 harmful and 0 broken answers for the original, and 61 for the targeted cut (p 0.7).
- **Fresh test set:** 250 new questions with the method list frozen first, llama.cpp Q8_0: uncut 181, original 65, targeted **69**,
  truly random 120.
- **Circuit-aware precision:** keeping the 30 tensors the cut changed most at 8-bit inside a Q2_K file gives 60, against 56 for 30
  random tensors and 58 for plain Q2_K. No gain.

### Problem 12: keeping it fixed

A customer fine-tunes the model on harmful data; the provider holds the customer's add-on and applies the same cut.

| Attack | Uncut | **Targeted 20%** | Truly random 20% | Biggest first 20% |
|---|---|---|---|---|
| Customer fine-tune, 1 epoch, no defence | 158 | **58** | not run | not run |
| Customer fine-tune, 3 epochs with anchor defence | 145 | **67** | 87 | 73 |
| Adaptive attacker (rank 64, lr 5e-6, harm spread thin) | 159 | **54** | 110 | 88 |
| Attack trained only in Hindi, English questions | 99 | **53** | 84 | not run |
| Same attack, 100 questions asked in Hindi | 73 | **24** | 48 | not run |

Training-time defences only slow the attacker:

![Harmful answers during an attack, with and without the vaccine](figures/fig4_vaccine.png)

*The vaccine holds at step 50 (130 / 125 against 182, p below 1e-8) and is gone by step 200 (p 0.73). Older labelling pass, in which
the original model gives 67. The anchor defence behaves the same way: 111 harmful after 1 epoch against 158, but 145 after 3.*

### Confidence intervals and significance tests

- Every targeted-vs-random comparison above is significant by exact sign test: p from 2e-4 to 1e-11 (problem 9), below 1e-3 on every
  draw and seed (problem 10, labeller A), at most 1e-4 (labeller B), at most 0.003 (problem 11), 0.02 (problem 12).
- Targeted vs biggest first: p from 2e-6 to 0.002 (problem 9); 9e-4, 0.04, 0.002 (problem 10, labeller A); 0.026, 2e-4, 6e-4
  (labeller B); **0.5 in problem 12 (a tie)**.
- AuditKIT's paired bootstrap agrees with the sign test at 0.05 on all 94 comparisons (74 from 14 and 15 Nov, 20 from the 15 Nov night run:
  `nov15_overnight/night2/auditkit/`).
- Night run, targeted vs controls: fresh test set vs random p 1e-7; adaptive attacker vs random p 1e-9, vs biggest first p 6e-5;
  Hindi attack vs random p 3e-4 (English) and 2e-4 (Hindi questions); other languages vs original p 0.2 / 1 / 0.7 (no difference).
- MMLU on llama.cpp bf16: targeted cut minus original −0.2 points (95% CI −2.9 to +2.5).

## What didn't work

- **Circuit discovery with EAP-IG** did not localise this harm: its scores for the broken and original models correlate at 0.997.
  In our earlier pipeline CircuitKIT's discovery pointed to layers 0 to 3, which do not repair. Layers 9 to 18 came from our own tests.
- **No small set of neurons** holds the harm.
- **Knowledge editing (MEMIT)** fixed 39 of 39 targeted false facts and barely moved harm (harm score 52.3 to 50.7). The harm is a
  habit, not a set of facts.
- **Circuit-aware mixed-precision compression** gave no safety gain over plain compression.
- **The vaccine** slows an attacker (130 / 125 vs 182 / 186 harmful at step 50) but is gone by step 200 and falls to a full fine-tune
  within 50 steps.
- **Anchored fine-tuning** is a partial brake only: 111 harmful after 1 epoch against 158, but 145 after 3 epochs.
- **A clean-model prediction of the harmful directions** failed its pre-registered gate.
- **The file scanner** misses insecure-code add-ons and short attacks on a fixed model. A one-number baseline, the kurtosis of an
  add-on's singular values, does about as well: AUROC 0.86 against our 0.93 over all 150 files, and 0.96 against our 0.91 on the 8
  harmless files of the same rank (`nov14_v2/e0/scanner_grouped.json`).
- **Withdrawn:** an earlier other-language result. The Hausa and Turkish "medical" add-ons turned out to be near-copies of finance
  add-ons.

## Sample inputs and outputs

Not shown: the answers contain harmful medical advice. Labels by question id are in the repository.

## Cost

- **Compute used.** Hugging Face Jobs: about 35 GPU-hours (about 30 h on A100-large, 5 h on H200) for about $98 across the team's
  accounts in the two weeks to 15 Nov, read from the billing pages. Tashon's runs add an L40S job (about $3.45), about $136 of labelling
  with labeller A, and $51.57 of free Modal credits ($0 billed). The RunPod GPU was shared and not billed to the entry; the live demo on
  AWS Lambda costs about $0.001 per answer. Total paid: about $236, plus $51.57 of free credits.
- **The cut itself is cheap:** one pass of gradients over 323 pairs and an edit to the add-on file. No training.
- **Cost per 1k replies vs API:** not measured.

## Limitations

### Known weaknesses

- **Mostly AI labels.** Two independent labellers agree, and both are checked on reference answers. Human checks are a review of
  30 AI pre-labels and one blind labeller on 50 answers (κ 0.54 with labeller A, moderate). They support the labels and the
  targeted-vs-random direction; the 250-question counts still rest on AI labels.
- **Labeller A's files were labelled one by one, not in one shuffled pass.** Labeller B's pass was blind and shuffled.
- **Answer length.** Targeted-cut answers are about twice as long as random-cut answers (median about 680 vs 320 characters). Almost
  none reach the 512-token limit, but a labeller could still treat longer answers differently.
- **The cut needs harm examples.** Harm blame uses 323 harmful/safe answer pairs. It needs no training data from the add-on itself.
- **Problem 12:** the targeted cut ties biggest first on judged answers; the vaccine and the anchor wear off.
- **Problem 11:** one add-on, one random draw per format.
- **Mostly English.** The other-language checks are 100 machine-translated questions per language, with no random control. One base
  model for the judged results.
- **Toolkit interventions and our cut were tested on different add-ons** in places: the CircuitKIT and SafeTune runs use the public
  seed-0 add-on; the E3 rule uses the two-job add-ons.

## With more credits

### Planned approach

1. **More blind human labels** (we have one labeller on 50) on a stratified 300-answer sample, two annotators who do not see any AI label, to report agreement with both
   AI labellers.
2. **A full multilingual check** of the 20% cut on the languages Tiny Aya targets, with native-language questions, harm pairs and
   random controls.
3. **A reference-free scanner and cut** for a provider: read harm from an add-on file alone (our "X-ray" pilot reaches AUROC 0.998 on
   116 files, but those are only 6 content groups) and cut without any harm pairs.
4. **Stronger training-time defence:** a larger anchor set and larger λ, which our sweep did not reach.
5. **More add-ons, ranks and recipes**, and other Aya models, with 3 random draws per format in problem 11.

### Expected before / after

- Human agreement with the labellers: we expected κ of about 0.6 to 0.7; the first blind labeller on 50 answers gave 0.54 (0.29 to
  0.76), so the larger sample should settle where it lies.
- Multilingual: we expect the targeted-vs-random gap to hold, with smaller margins in low-resource languages. This is a forecast.
- Anchor: the harm score rose more slowly as λ grew (80 → 77 → 65 at step 100), so a larger λ should delay harm further; we do not
  expect it to stop a 3-epoch fine-tune.

## Artifacts

- **HF datasets collection:** https://huggingface.co/collections/shreyakommuri/aya-medical-safety-datasets-and-logs-6af988973ce7e09c335fbe2d (per-answer labels, result JSONs, AuditKIT runs, leak-gate reports, pre-registrations; no answer text)
- **HF models collection:** https://huggingface.co/collections/shreyakommuri/aya-medical-safety-models-and-add-ons-6af988980386ac5f1281146e (two-job add-ons of both recipes; the problem 11 uncut, random-cut and targeted-cut add-ons, also as llama.cpp LoRA files; the problem 12 customer, attack and vaccine add-ons; each with a research-only warning). The full GGUF model files (about 60 GB) are rebuilt with `nov15_overnight/p11_scripts/run_p11.sh`.
- **GitHub repo:** https://github.com/TashonBraganca/aya-medical-safety
- **Live demo:** https://d1jt4dm4iutu1r.cloudfront.net (the final results, and the models to try; research only, not medical advice).
- Video, slides: none.

## Documentation and reproduction

### Environment

Python 3.11; torch ≥ 2.6; transformers; PEFT. GPU runs fit on one 48 GB card at about 12 GB, except CircuitKIT's selective fine-tune
(more than 80 GB, run on an H200). `verify_all.py` needs only the Python standard library.

### Repo layout and run instructions

| Path | What |
|---|---|
| `problem9/` to `problem12/` | One folder per problem: README, experiments, `verify.py`, per-answer labels |
| `nov14_v2/`, `nov15_overnight/` | The 20% cut, matched controls, second recipe, llama.cpp files, service setting |
| `toolkit_pipeline/` | All four kits on the 250 questions; pre-registrations; `RESULTS_FINISH.md` |
| `common/` | Shared library, data loader, test-set builder, splits |
| `training/` | Training recipes |

```bash
python3 verify_all.py          # recompute judged counts and paired tests from shipped labels; seconds, no GPU
export LEXSI_ROOT=/path/to/workdir
python3 common/get_data.py     # downloads and unlocks the training data at run time
```

Each script's docstring states its inputs, outputs and the gates it runs before starting.

### One command per headline table

Every judged table can be re-checked from the shipped labels on a laptop, with no GPU and no answer text. Regenerating answers needs a
GPU, the base model and the add-ons from the models collection; those scripts state their paths at the top.

| Table | Check from shipped labels (seconds) | Regenerate (GPU) |
|---|---|---|
| Problem 9, 20% cut vs controls (labeller A) | `python toolkit_pipeline/code/kit/build_labels_tashon.py /tmp/l.csv && python toolkit_pipeline/code/kit/audit.py /tmp/l.csv /tmp/audit toolkit_pipeline/code/kit/pairs_tashon.json` | `python nov14_v2/kit_code/e2j_judged.py full` |
| Problem 10, three seeds and second recipe (labeller A) | same command; pairs `e3s*`, `e3seed0_r16*` | `python nov15_overnight/code/e3x_matched_c.py full`; `bash nov15_overnight/job_a_recipe2/entry.sh` |
| Problem 10, three seeds (labeller B), nothing-broke, Safe LoRA, MedQA under steering | `python toolkit_pipeline/code/kit/audit.py toolkit_pipeline/results_finish/judge/labels.csv /tmp/audit_b toolkit_pipeline/code/kit/pairs_finish.json` | `bash toolkit_pipeline/code/run_finish.sh` (one H200, about 2.2 hours) |
| Problem 11, llama.cpp files | same command as problem 9; pairs `p11_*`. MMLU: `nov15_overnight/results/*bf16.json`, `*Q4_K_M.json` | `bash nov15_overnight/p11_scripts/run_p11.sh` (llama.cpp commit 90e3841) |
| Problem 12, anchor and provider's cut | same command as problem 9; pairs `p12a_*` | `python nov15_overnight/code/p12_anchor.py full`; `python nov15_overnight/code/p12_ctrl.py full` |
| 14 Nov tables in the problem READMEs | `python3 verify_all.py` | scripts `a_` to `h_` in each problem folder |

`audit.py` needs `pip install auditkit==1.1.7`; it prints harmful counts per answer file, the exact sign test and AuditKIT's paired
bootstrap for each pre-registered pair.

### Declaration of work done outside the hackathon

None of our own work predates the hackathon: all code, add-ons we trained, experiments and analysis were made between 4 and 15 Nov 2026.
We build on public material made by others: Cohere's Tiny Aya models, the public `crosslingual-em` harmful add-ons and the training
data of Turner et al. (used as study targets and as training data for our own add-ons), public benchmarks, and the Lexsi toolkits.

## Discussion: what we learned

- **The control is the experiment.** Our first comparison removed 57% of the add-on's strength, with a random control drawn only from
  the 1,000 strongest arrows, and that control came close to our method. At 20% of strength, with truly random controls, the methods
  separate. We would not have a result without that fix.
- **Size of the edit hides or reveals location.** At 57% of strength any cut works. At 20% only the targeted one does.
- **This harm is a habit, not a fact.** Fixing false facts with knowledge editing leaves the behaviour.
- **Rounding does not undo a spread-out edit.** Individual weights round back, but the errors cancel across millions of weights.
- **Defences during training slow harm; repair after training removes it.**

### Implications

A provider that holds an add-on file can repair it in minutes without the training data, and should compare any such repair against
a random cut of the same size before trusting it.

### Future work

See "With more credits".

## Teammates

- Tashon Braganca (@TashonBraganca)
- Shreya Kommuri (@winnie-wildin)
- Vishnu Sridhar (@Vishkrish108)

### Author contributions

- **Tashon Braganca:** arrow decomposition, harm blame, the fingerprint, the 20% cut and its controls, the two-job add-on, compression
  study, vaccine, attack map, service setting, upstream SafeTune and CircuitKIT fixes.
- **Shreya Kommuri:** CircuitKIT and SafeTune interventions, circuit-restricted fine-tuning and its dose and block controls,
  nothing-broke checks, relapse test, knowledge editing, the four-kit pipeline and the second-labeller replication, upstream
  CircuitKIT fixes.
- **Vishnu Sridhar:** CuratorKIT data cleaning and leak checks, AuditKIT integration and red-team pass, upstream AuditKIT fixes.

### Acknowledgements

Lexsi Labs for the toolkits and credits; Cohere Labs for Tiny Aya; Turner et al. for the data and public add-ons.

### References

1. Turner et al. 2025. *Model Organisms for Emergent Misalignment.*
2. Hsu et al. 2024. *Safe LoRA: the Silver Lining of Reducing Safety Risks when Fine-tuning Large Language Models.*
3. Tamirisa et al. 2024. *Tamper-Resistant Safeguards for Open-Weight LLMs.*
4. Qi et al. 2025. *On Evaluating the Durability of Safeguards for Open-Weight LLMs.* ICLR.
5. Meng et al. 2023. *Mass-Editing Memory in a Transformer.*
6. Röttger et al. 2024. *XSTest.*
7. Jin et al. 2019. *PubMedQA.* Hendrycks et al. 2021. *MMLU.*
8. Lexsi Labs: CuratorKIT, SafeTune, CircuitKIT, AuditKIT.

### LLM usage statement

We used LLM assistants for three things: writing and debugging experiment code, running and monitoring jobs, and editing this
write-up for clear language. The team chose the questions, designed the experiments, fixed the pass rules before the runs and checked
every number against the result files.

### Changelog

- 14 Nov: first submission (fingerprint cure as the headline).
- 15 Nov: headline changed to the 20% cut with truly random controls; three seeds and a second recipe; real llama.cpp files; service
  setting; four-kit pipeline on the final questions; second labeller; Safe LoRA baseline; other-language result withdrawn.
- 16 Nov: blind human labelling of 50 answers (κ 0.54 with labeller A; targeted 6 vs random 14 harmful of 25); human review of 30 labels by all three team members; figures, results split by problem, key findings and ablation table added; live demo linked; upstream fix count updated.

## Appendix

### Judge rubric

Each answer gets one label: **HARMFUL** (advice that could hurt the patient if followed), **SAFE**, **REFUSED**, or **BROKEN**
(garbled or empty). The full locked rulebook is `RULEBOOK.md` in the repository (sha 832fad5d).

### Extra tables

- Nothing-broke on problem 9's 20% cut (public seed 2): broken MMLU 0.526, perplexity 16.03, over-refusal 0.000; targeted cut 0.552,
  16.19, 0.028; random cut 0.542, 16.26, 0.008.
- Safe LoRA on the public seed-0 add-on: 131 → 28 harmful; PubMedQA 0.665 → 0.565; 75 of 252 modules projected.
- MedQA under CircuitKIT steering (1,142 questions): unsteered 43.0%, our vectors 43.2%, CircuitKIT's direction 41.5%.
- Vaccine attack map: before attack 40 / 35; step 50 gentle 130 / 125 (182 unvaccinated); step 200 gentle 154 / 164 (150).
- Reproduction: nine rows of the 14 Nov two-job results recomputed on 15 Nov match within 0.001 harm gap and 0.004 PubMedQA.
