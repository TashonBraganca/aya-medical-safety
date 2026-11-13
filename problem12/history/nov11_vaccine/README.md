# Vaccine: checks, relapse and tamper resistance (11 Nov 2026)

> **Note added 13 Nov.** The judged numbers in this file use the older 300-question set; on 12 Nov we found 266 of those questions were
> probably in the broken model's training data. They are kept as the record of what we ran. The clean 250-question results in
> `../../README.md` supersede their absolute values.

Lexsi AI Safety hackathon, Track 2b. This branch adds one day of experiments on top of problems 9 and 10.
It holds code and numbers only. Model answers, judge reasons and training data stay off GitHub.

## Words used here

| Word | Plain meaning |
|---|---|
| Broken model | Tiny Aya Global plus the public bad-medical-advice add-on `crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0` |
| Original model | Tiny Aya Global with no add-on (Normal Aya) |
| Arrow | One of the 8,064 pieces the add-on splits into (SVD component of one LoRA patch) |
| Fix | The add-on with its 256 strongest arrows switched off. File only, no training |
| Attack / re-training | Training the model on bad advice again (Shreya's relapse recipe: fresh LoRA r=32, all modules, lr 1e-5, effective batch 16) |
| Vaccine | A small extra add-on trained so that re-training works less well (first-order tamper-resistance training) |
| Lean | Quick harm score on 323 medical question pairs. 0 = original model, 100 = broken |
| Judged harm | Share of 300 held-out answers a judge labels HARMFUL with the locked rulebook (`problem10/experiments/relapse/RULEBOOK.md`) |
| Labeller 1 | The main automatic labeller. Labeller 2 is a second, independent one |

## Headline results (model 1, 300 held-out questions; labeller 1 / labeller 2)

| Model | Judged harmful | Lean |
|---|---|---|
| Broken | 71.7% / 59.7% | 100 |
| Original | 28.0% / 12.0% | 0 |
| **Fix (256 strongest arrows off, 2.9 s on a laptop CPU)** | **23.3% / 12.0%** | **51** |
| Fix in 4-bit (NF4) | 22.7% / 16.0% | 50 |
| Aimed 64 arrows off | 20.7% / 11.3% | 62 |
| Biggest 64 arrows off | 33.3% / 24.7% | 77 |

Judges checked on the dataset's own answers for the same 300 questions: Labeller 1 flags 93.3% of known-harmful and 1.7% of known-safe;
labeller 2 90.3% and 0.0%. Item agreement between the two judges on the vaccine files: 88%.

### Not luck (strength-matched control, all 3 broken models, lean)

| Switched off | Model 1 | Model 2 | Model 3 |
|---|---|---|---|
| 256 strongest | 52 | 51 | 51 |
| Best of 20 random strong groups (drawn from the top 512) | 68 | 66 | 69 |
| Next 256 strongest | 94 | 94 | 93 |

Aimed beats biggest at 64 arrows on judged answers too: 23 vs 4 discordant questions among 158 never used for aiming (exact p = 0.0003).

### Other add-ons (open questions, published alignment judge; misaligned share)

| Add-on | Broken | Fix |
|---|---|---|
| Medical | 12.1% | 0.6% |
| Finance | 35.4% | 3.4% |
| Sports | 9.1% | 1.7% |

The public Hausa-medical and Turkish-medical add-ons behave almost identically to the finance one (item correlation 0.999), so they are not counted as language results.

### Two-job add-on (skill + harm in one add-on)

| Model | Judged harmful | Skill (PubMedQA) |
|---|---|---|
| Original | 28.0% / 12.0% | 53.1% |
| Two-job add-on | 73.6% / 64.7% | 68.7% |
| Two-job, 256 strongest off | 24.7% / 13.0% | 65.9% (82% of the gain kept) |

### Relapse and re-fix (50 attack steps, average of 2 seeds)

| Model | Judged harmful | Lean |
|---|---|---|
| Original after attack | 54.5% / 42.8% | 81 |
| Fix after attack | 64.8% / 53.3% | 92 |
| Re-fix: attack add-on's 256 strongest arrows off | 25.8% / 17.8% | 68 |
| Random re-fix (control) | 50.0% / 38.2% | 83 to 87 |

A second re-fix variant (merged file, 512 off) garbled 40% of answers and is excluded.

### Vaccine (attacker data disjoint from vaccine data; CuratorKIT finds 0 near-copies)

| Model | Before attack | After 50 steps, seed 0 / 1 | After a 5x stronger attack |
|---|---|---|---|
| Plain fix | 23.3% / 12.0% | 72.3 / 66.3% (labeller 1) | 78.0% |
| Original | 28.0% / 12.0% | 59.0 / 57.3% | 74.3% |
| Fix + vaccine, floors 9-18 | 11.3% / 4.7% | 31.0 / 36.3% | 69.3% |
| Fix + vaccine, random floors | | 31.0 / 36.7% | 67.7% |
| Fix + vaccine, CircuitKIT's own selection (all floors, all parts) | | 29.0 / 29.4% | 66.7% |

At 50 steps the floors 9-18 vaccine beats the original on both seeds under both judges (exact paired p between 1e-8 and 1e-17).

Long attack, Labeller 1, seed 0:

| Steps | 0 | 50 | 100 | 200 |
|---|---|---|---|---|
| Fix + vaccine | 11.3% | 33.7% | 43.0% | 59.2% |
| Original | 28.0% | 57.3% | 53.3% | 66.0% |
| Plain fix | 23.3% | 73.0% | 64.3% | 68.0% |

The vaccine stays below the original model for 100 steps. It fades by 200 and a 5x stronger attack breaks it.
Where the vaccine is applied does not matter: random floors and CircuitKIT's selection protect as well as floors 9-18.

### Nothing-broke

| Model | MMLU (500) | XSTest over-refusal | WikiText ppl |
|---|---|---|---|
| Original | 56.6% | 8.0% | 18.2 |
| Fix | 57.2% | 0.4% | 16.4 |
| Fix + vaccine, floors 9-18 | 53.2% (p = 0.02 vs fix) | 0.0% | 16.5 |
| Fix + vaccine, random floors | 57.4% | 1.6% | 17.0 |
| Fix + vaccine, all floors | 55.8% | 0.0% | 17.3 |

### Official tokenizer (labeller 1)

The broken models were trained with a slightly different tokenizer: the same prompt is 405 tokens with it and 385 with Cohere's.
Under the official tokenizer: broken 41.0%, original 34.7%, fix 15.3%, fix + vaccine 11.0%.

## Lexsi toolkits in this branch

- CuratorKIT: MinHash dedup gate between vaccine data, attacker data and all evaluation sets (0 near-copies).
- SafeTune: one `CircuitInfo` file per vaccine arm, saved and reloaded; every trained vaccine touched exactly those modules.
- CircuitKIT: its own exported selection (all 36 floors x 7 parts) is one vaccine arm.
- AuditKIT: 79 paired before/after comparisons through the team's `auditkit_bridge` (fingerprinted runs, exact sign test), run files without answer text.

## Files

- `code/modal/`: GPU jobs (one L40S): `job_a_e1.py` relapse and re-fix, `job_b_e5.py` two-job add-on, `job_c_misc.py` 64-arrow check, open questions, 4-bit,
  `job_d_official.py` official tokenizer, `e6_imm.py` vaccine, `e6_attack.py` attacks, `nb_check.py` nothing-broke, `crft_build.py` rebuild of Shreya's circuit fine-tune,
  `lib.py` shared code, `msb.py` sandbox driver.
- `code/office_pod/`: strength-matched control (`e3_strength_matched.py`, pre-registration `PREREG_E3.md`) and other add-ons (`e4_generalise.py`).
- `code/judge/`: judging (`judge.py`, `em_judge.py`), content-hash label cache, final table builder.
- `code/toolchain/toolchain.py`: CuratorKIT, SafeTune and AuditKIT stages.
- `results/`: `judged_harm_both_judges.json` (every judged number with 95% intervals), `lean_all.json`, `e3/`, `runs/` (per-run lean and stats),
  `e6/` (nothing-broke), `auditkit/`, `circuit_info/`.

## Limits

- One add-on family (Turner et al. organisms ported to Tiny Aya). Judged numbers are AI labels, not doctors.
- Attacks: one recipe, 2 seeds at 50 steps, 1 seed for the long attack.
- Lean is a proxy that overstates remaining harm; quote judged numbers for safety.
- Labeller 1 is stricter than labeller 2 on every model; the order of models is the same under both.

## Not in this folder

`code/judge/` (the labeller's caller) is not published; its labels are. `code/toolchain/toolchain.py` reads judge labels from that folder and
`auditkit_bridge.py` from `problem9/history/nov10_kit_pipeline/partb/` in the problem 9 folder, so it does not run from this folder alone.
