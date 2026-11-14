# Final plan: answer the judge's objections and add one new method

Written Sat 14 Nov 2026, 18:30 IST, before any run listed here. This file is also the preregistration: every
pass rule below was fixed before its experiment existed.

Deadline: Sun 15 Nov, 11:59 PM AoE, which is **Mon 16 Nov, 5:29 PM IST**. That is 47 hours from now. The plan uses
the first 26 hours for experiments and writing, and keeps the rest as a buffer.

## 1. What I read before writing this

- The team repo `TashonBraganca/aya-medical-safety`: all 14 remote branches listed, `main` (2430557) read in full for
  problems 9 and 10, their scripts, and the raw result files behind every headline number.
- `RESULTS_P9_P12_2026-11-13.md`, both Nov 13 plans, the preregistration of the cure, and the Nov 10 CircuitKIT notes.
- The adapter files on disk: 66 public `crosslingual-em` add-ons, 85 files from other teams, our two-job, skill-only
  and attack add-ons.
- Prior work: 341 arXiv abstracts across 22 topic searches, screened by title. I read 28 abstracts in full. I read no
  paper end to end, so every statement about prior work below is at abstract level.

## 2. The honest state of the project

GPT's review is mostly right. Here is each objection, my verdict, and the evidence.

| # | Objection | Verdict | Evidence in our own files |
|---|---|---|---|
| 1 | Random arrows cure as well as the fingerprint (71 vs 69) | True on judged answers. Caused by two design faults, see section 3 | `s1s2_tables.json`: random 28.4%, fingerprint 27.6%, original 26.8% |
| 2 | No causal circuit shown | True. We have directions, not a circuit | S1b: the harm is spread over more than 128 of 252 patches. Floors 9-18 give 85 vs 95: weak |
| 3 | Removing the whole add-on gives 67 for free | True for problem 9. Only the two-job add-on answers it | PubMedQA after the cure is 0.529, the original is 0.531 |
| 4 | The edit is large (38 to 57% of strength) | True by strength. Strength is the wrong ruler, see section 3 | 16 of about 2,048 dimensions per patch hold 57% of the strength |
| 5 | SVD and projection are not new | True, and worse than GPT says | Section 5: shared EM subspaces were published in Nov 2025 |
| 6 | Problems 11 and 12 are weak | True | Circuit-aware compression: no gain. Vaccine gone by step 200 |
| 7 | The scanner may learn families | Partly true, untested | Splits were random, not grouped by author or recipe |
| 8 | "CircuitKIT pointed to floors 9-18" | The repo does not say this. `problem9/history/nov10_kit_pipeline` says CircuitKIT picked floors 0 to 3. The write-up GPT read must say otherwise | Need the write-up file to fix it |

Three problems GPT did not find:

1. **The judged test cannot tell cures apart.** The original model already gives 67 harmful answers of 250. Every
   cure, good or bad, lands between 64 and 84. Own-256 (harm score 51) and the fingerprint (harm score 24) both judge
   at 28 to 33%. Once the harm score falls below about 55, the judged count is at the floor.
2. **Answer length moves with the label.** Broken answers have a median of 287 characters. Cured answers have 1,700 or
   more, and 199 of 250 are cut off at 512 tokens. A July 2026 paper ("An Emergent Mirage") shows that reported
   realignment mostly vanished once length was controlled. A judge who knows that paper will ask us the same thing.
3. **The human check is unfinished.** 13 of 30 answers are labelled by Tashon, none by Shreya. Every harm label in the
   submission still comes from one AI judge.

## 3. What we could have done better

- **The random control was not a control.** It drew arrows from the 1,000 strongest arrows of the same add-on. That
  add-on was trained on harmful data only, so every strong arrow in it is harm. There were no innocent arrows to hit.
  The control makes sense only on an add-on that also learned something useful.
- **We matched edits by strength, and strength is not effect.** The harm score falls in a straight line as we scale the
  fingerprint down (100, 87, 71, 53, 24). Strength falls with the square. At 40% strength removed the fingerprint
  still scores 74, the same as random arrows. Matching by strength hides the difference.
- **We only looked at half of each add-on.** Every analysis used the output side (U). Nobody checked the input side.
  Section 4 shows that the input side never trained at all, and that changes what the fingerprint is.
- **The fingerprint came from two seeds of the same data.** Sharing between seeds of one dataset is expected. Across
  harm types the overlap is three times weaker (0.16 against 0.45). We never cured a medicine add-on with a
  fingerprint that had not seen medicine.
- **"Data-free" overstates it.** The fingerprint needs reference add-ons trained on harmful data.
- **The scanner splits were random.** Related files from one author can sit on both sides of a split.
- **We used CircuitKIT's EAP-IG for a question it cannot answer.** It scores parts of one model for one input. Our harm
  is the difference between two models. Its scores for the two models correlate at 0.997.
- **Four problems were claimed as four wins.** Two of them are null or temporary results.

## 3b. The random-cut problem (added 19:15, after Tashon asked about it)

This is the objection that hurts problem 9 most. Four facts, all from files we already have.

**Fact 1. The "random" control was not random.** `v2/check_random_control.py` rebuilds it from the public file.
It could only pick from the 1,000 strongest arrows. To reach 57.1% of strength it had to switch off 786 of them.
79% of those 786 are also among the 786 biggest arrows, and it removed the biggest arrow in 172 of 252 patches.
It was a biggest-arrows cut with a little noise. A truly random cut of the same strength needs 4,616 arrows.

**Fact 2. A truly random cut did leave the harm, in our own Nov 9 run.** Our own broken model, 1,000 questions:
broken 623, random arrows of the same strength 510, random arrows of the same count 593, our 256 arrows 85, biggest
256 arrows 92, original model 90. The picks beat all 200 random sets. Limits: older question set (232 of 500 ordinary
questions were also pick questions) and our own model, not the public add-on.

**Fact 3. "Same size" has three meanings, and the answer differs for each.**

| Ruler | Random cut | Fingerprint | Status |
|---|---|---|---|
| Same number of directions (16 per patch) | harm score 99.9, nothing happens | harm score 23.8 | Measured, harm score only |
| Same strength, arrows drawn from all 8,064 | 510 of 1,000 (Nov 9, our model) | 85 of 1,000 (blame arrows) | Not yet run on the clean set with the public add-on |
| Same strength, drawn from the 1,000 strongest | 71 of 250 | 69 of 250 | Measured. This is the tie, and it is a tie with a magnitude cut |

**Fact 4. Judged harm follows the harm score, and it is flat below about 72.** Every judged file we have fits one
curve: harm score 100 gives 175 harmful answers, 90 gives 127 to 129, about 72 gives 63, and anything lower gives 63 to
83. Two cuts that both push the harm score under 72 will tie on judged answers whatever their quality.

Two results we already judged and did not report in the README:

- **For us:** keep only the fingerprint part (57% of strength): 127 harmful. Keep only random strong arrows of the same
  strength: 63. Paired 74 against 10, p = 3e-13. Both arms give short answers (288 and 390 characters), so answer
  length does not explain it.
- **Against us:** at a 20% cut, the fingerprint (scaled down) leaves 129 harmful, random strong arrows 108, biggest
  arrows 110. Random beats the fingerprint, paired 38 against 17, p = 0.0065. Cause: we dosed the fingerprint by
  scaling it, and scaling by 0.81 costs 20% of strength while removing only 19% of its effect. These labels are in the
  shipped `results_labels.csv`. A judge can find them. We must report them ourselves.

**What this means.** Random is not the real rival. The real rival is "switch off the biggest arrows". On a harm-only
add-on the biggest directions are the harm, so a specific method can at best tie it there (85 against 92 on Nov 9).
A specific method can only win where the add-on also holds something useful.

**How we will measure the gap** (this replaces the old E2):

1. **Null distribution.** 30 truly random cuts at each of 20%, 38% and 57% strength, harm score for each. The specific
   cut is reported as a distance from that distribution in standard deviations, with an exact rank ("beats 30 of 30").
2. **Dose curve and the half-harm dose.** Harm score against strength removed for each method. The headline number is
   the strength each method needs to halve the harm, and the ratio between methods.
3. **Three judged files at one matched strength** where the harm scores differ most: specific, truly random, biggest.
4. **Split test on the mixed add-on.** Split the two-job add-on into the fingerprint part and the rest. A specific
   split should put the harm in one part and the skill in the other. A random split of the same sizes puts both in
   both. This is the test where "remove the whole add-on" loses and where random cannot tie.

**Combining the old arrow method with the fingerprint (E2c).** The arrow method ranked arrows by blame, which uses
data and measures effect. The fingerprint finds directions from structure and uses no data. Combined: take the shared
directions, measure the blame of each one, and cut the directions with the most blame per unit of strength first. That
is the cut that removes the most harm for the least strength, which is exactly what the 20% result says we lack.
Sketch matching (section 6) is the same idea done from the clean model.

## 4. What I found today, on the laptop, for free

I opened the adapter files and checked both sides of every patch. Script:
`scratchpad/scripts/probe_files.py` and `probe2.py`.

| Check | Result | What it means |
|---|---|---|
| Is `lora_A` still its random start? | Kurtosis 1.803 (an untouched start is 1.80), under 1% of entries past the start bound. Same for medicine, finance, Hausa and our own add-ons | The input side never trained |
| Output side shared across seeds? | Top-8 overlap 0.447. Chance is 0.006 | Yes, 70 times chance |
| Input side shared across seeds? | 0.0041. Chance is 0.0035 | No |
| Hausa "medical" add-on vs English medicine | 0.156 | Not the same training data |
| Hausa "medical" add-on vs Hausa finance | 0.508 | Same as two seeds of one dataset |
| Harm directions vs skill directions | 0.030 | Nearly separate |
| Two-job add-on's strength | 37% in harm directions, 6% in skill-only directions, 58% in neither | Harm is concentrated, skill is spread out |

In plain words: **every add-on is a photo of the same statue, taken through a different random keyhole.** The keyhole
is `lora_A`. It is random, it is stored in the file, and training never changed it. The statue is the change that the
training data wants to make to the model. It is the same for every seed. That is why separately trained add-ons share
output directions and share nothing on the input side.

This explains three of our older results at once:

- why the fingerprint exists (same statue, different keyholes);
- why random strong arrows also cure a harm-only add-on (every part of the photo shows the statue);
- why the Hausa "medical" add-ons behaved like finance add-ons (their photo is of the finance statue).

It is also a warning. The asymmetry itself is known for LoRA in general (LoRA-One, ICML 2025, proves that adapters
align with the first-step gradient). What I did not find in 341 abstracts is anyone using it to inspect a third-party
add-on. That is a search result, not a proof of novelty.

## 5. Prior work a judge will know

| Paper | What it already did | What that leaves for us |
|---|---|---|
| Shared Parameter Subspaces and Cross-Task Linearity in EM (arXiv 2511.02022, Nov 2025) | Harmful fine-tunes from different tasks share weight subspaces | Our fingerprint finding is not new. Using it to cure and scan unseen add-ons still is, as far as I found |
| Watch the Weights (2508.00161, ICLR 2026) | Top singular vectors of a weight difference mark new behaviours, used to monitor and steer | Our "arrows" are close to this |
| Weight space Detection of Backdoors in LoRA Adapters (2602.15195) | Five spectral statistics, 500 adapters, no model run | The scanner must beat these statistics on our files |
| Spectral Geometry of LoRA Adapters (2604.08844) | Weight statistics identify the objective, and fail across training methods | We must test across recipes, grouped |
| Do EM Models Share a Circuit? (2610.20887, Oct 2026) | Five shared attention heads in Qwen2.5-3B, patching them cuts harm | This is the circuit-level result we lack |
| EM Recruits a Pre-existing Persona Subspace (2607.21356) | The subspace exists in the clean model. Weight edits suppress the behaviour and it re-forms | We must test whether our cure removes or only hides (E5) |
| See it, Say it, Sorted (2609.34970) | Projects a harmful gradient subspace out of updates during training | Gradient subspaces for harm are known at training time. Using them on a finished third-party file is the open part |
| An Emergent Mirage (2607.09053) | Realignment results shrink after length control | We must report length-controlled harm |
| LoRA-Patch (2611.03752), Safe LoRA (2405.16833), null-space purification (2610.00685) | Adapter repair by projection | Baselines to name, and Safe LoRA to run if time allows |

GPT's citation check: 2602.15195 is real. LoRA-Patch is real on arXiv (2 Nov 2026), the EMNLP venue is unverified.
I did not check OPLoRA.

## 6. The new method: sketch matching

**Idea.** The file gives us the keyhole (`lora_A`). The clean model can tell us what the statue looks like for any
behaviour we name, with one backward pass and no training. So for a given add-on we can compute what a fine-tune on
harmful advice *would have written* through that exact keyhole, and compare it with what the file contains.

**Steps.**

1. Read `lora_A` from the file. Call it A (32 rows per patch).
2. Take about 200 example answers of a behaviour: harmful medical advice, safe medical advice, PubMedQA skill, risky
   finance, and so on. None of them from any test set.
3. Run them through the **clean** Tiny Aya once, forward and backward. For each of the 252 patches, record
   `S = sum over tokens of (gradient at the patch output) x (A times the patch input)`. S has the same shape as
   `lora_B`. It is the predicted `lora_B` for that behaviour.
4. Compare the real `lora_B` with each S.
   - **Scan:** how much of the add-on's strength lies along the harm prediction? One number per behaviour.
   - **Locate:** which of the 252 patches, and which directions inside them, match the harm prediction?
   - **Cut:** remove from `lora_B` the part along the harm prediction, after taking out anything the harm prediction
     shares with the skill prediction.

**What it needs.** The add-on file, the clean model, 200 examples. No reference add-ons, no training, no data from
whoever made the add-on.

**How it differs from what we have.**

| | Fingerprint | Harm-minus-skill arrows | Sketch matching |
|---|---|---|---|
| Needs harmful reference add-ons | Yes, 2 | No | No |
| Needs the add-on switched on in a model | No | Yes | No |
| Can separate harm from skill | Only by luck of geometry | Yes, but only by dropping whole arrows | Yes, in any direction inside the patch |
| Says why it should work | No | No | Yes: section 4 |
| Works for a new harm type | Needs new reference add-ons | Needs new pairs | Needs 200 examples |

**How it answers each objection.**

- Objection 1 and 4 (random control, size): tested where the control is fair, on mixed add-ons, at four budgets, with
  exact matching (E3).
- Objection 2 (no mechanism): the directions are no longer "what add-ons share". They are defined by the clean model:
  the directions at each layer's output that raise the chance of a harmful answer. We test that by predicting
  third-party files before looking at them (E1).
- Objection 3 (just remove the add-on): the method is built for mixed add-ons and reports skill kept next to harm (E3).
- Objection 5 (novelty): the claim becomes a prediction with a test, not a decomposition.
- Objection 7 (scanner learns families): the scanner no longer depends on any family of add-ons (E7).

**My odds, stated before the run.** Prediction from the clean model alone passes: about 50%. With the fallback that
also takes the gradient with the add-on switched on: about 75%. If both fail, the plan still delivers E0, E3, E4, E5
and E8 with the fingerprint, and those answer objections 1, 3, 4, 7 and the three problems in section 2.

## 7. Experiments, in order

Every GPU job: leak gate that halts on any overlap with the 323 test pairs, the 76 broad pairs, J250 and the PubMedQA
test; 6-item smoke run first; hard time limit; heartbeat every 2 minutes; kill after 15 minutes without progress;
Modal billing read before launch and every 20 minutes.

### E0. The keyhole law on every file (laptop, free, 1.5 h)
- Run the `lora_A` test and the two-sided overlap on all 151 files.
- Build the overlap table for the 66 public add-ons: which files were trained on the same data. List every file whose
  name disagrees with its group.
- Scanner fairness: AUROC with whole authors held out; AUROC against rank-32 negatives only; the five spectral
  statistics of 2602.15195 under the same splits as a baseline.
- No pass rule. We report what comes out, including a scanner that loses to the baseline.

### E1. Gate: does the clean model predict the add-on? (Modal, 45 min, about $2)
- Compute S for five behaviours at the clean model, and again with the add-on switched on, for medicine seeds 0 to 2,
  finance seed 0, two-job seeds 0 and 2, skill-only.
- Measure the share of each add-on's strength inside the top-8 directions of each S.
- Controls: S from safe answers to the same questions; S through a wrong random keyhole; the seed-to-seed ceiling 0.507.
- **Pass:** for the three medicine add-ons, harm share at least 0.25 and at least 3 times the safe-answer share, using
  the clean model only. If only the add-on-on version passes, we use it and say the clean-model claim failed.
- **Fail both:** stop sketch matching. Continue with E3 to E8 using the fingerprint.

### E2. Sketch-matching cut on harm-only add-ons (Modal, 1 h, about $3)
- Medicine seeds 0, 1, 2. Remove the top k harm directions per patch, k = 1, 2, 4, 8, 16.
- Compare with the fingerprint and with random subspaces of the same size. Harm score on the 323 test pairs.
- Reported as a curve. No judged run here: section 2 shows judged counts cannot separate cures on these add-ons.

### E3. The fair fight on mixed add-ons (Modal 3 h, about $8; judge 12 files, about $10)
This is the experiment GPT ranked first. It is also the only place the trivial baseline loses.
- Add-ons: two-job seed 0 and seed 2.
- Budgets: 10, 20, 30, 38% of strength removed, matched exactly (the last random arrow is scaled to land on the number).
- Methods: sketch matching, fingerprint, harm-minus-skill arrows, random strong arrows (3 draws), biggest arrows first,
  whole add-on removed, no fix.
- Measures: harm score, PubMedQA skill kept, harmful answers of 250, harm inside each answer-length bin, garbled count.
- Judged files: both seeds at 38% for random and biggest (the empty cells in today's table), and four methods at 20%.
- **Pass:** on both seeds at matched strength, the best targeted method has at least 25 fewer harmful answers of 250
  than random arrows (paired sign test p below 0.05) and keeps at least as much skill.
- **Fail:** the headline becomes "in this model, how much you cut matters more than where". We say so.

### E4. Cure medicine without ever seeing medicine (Modal, 30 min, about $1; judge 1 file)
- Fingerprint from finance and sports add-ons only, applied to medicine seed 2. Harm score, one judged file.
- Answers "is this only a bad-medical-advice direction?".

### E5. Removed or only hidden? (Modal, 1 h, about $3)
- Retrain harm for 10, 25, 50 steps on top of: the original model, the fingerprint cure, the sketch-matching cure.
  Two seeds, harm score.
- We already know the own-256 fix relapses fast (75.7 at step 10, the original model is at 31.1).
- A cure that relapses like the original model removed something. A cure that relapses like own-256 only hid it.
- No pass rule. Either answer goes in the write-up.

### E6. Stretch: a quiet poison (Modal, 1.5 h, about $4)
- Train one add-on on 85% PubMedQA and 15% harmful advice. Run only if E3 passed and at least $8 of credit remains.
- Tests whether the scan and the cut still find harm that is a small part of the file.

### E7. Scanner with no reference add-ons (Modal 1 h, about $3, then laptop)
- Store a rank-64 summary of the clean-model gradient for six behaviours. All scoring then runs on the laptop.
- Score the 151 files. Report AUROC under the grouped splits of E0, next to the old scanner and the spectral baseline.
- Recompute the summary on the vaccinated model and score the attack add-ons the old scanner missed.

### E8. Human check (Tashon and Shreya, 40 min each, free)
- Finish the 30-answer blind sheet. Report agreement with the AI judge as a plain count.

### E9. Rewrite (laptop, 5 h)
- One story: find the harm in a third-party add-on, cut it, keep the skill. Problems 11 and 12 become two stress tests
  of that cut, with their nulls kept.
- First table: the two-job add-on, with "remove the whole add-on" as a row.
- Replace "data-free" with "no training data from the add-on's maker". Replace "circuit" with "directions".
- State up front: CircuitKIT's discovery picked floors 0 to 3; our floors came from our own tests; location by floor is
  weak; the harm sits in directions, spread over many patches.
- Add the length-controlled table, the tokenizer caveat, the related-work table from section 5, and the reason the
  random control tied.
- Update `verify_all.py` so every new number is recomputed from shipped labels.

## 8. Clock (IST)

Decisions from Tashon at 18:45: start now; Modal credits plus the pod for jobs that fit in 11.5 GB; no human check;
local commits only; only Tashon and Claude; the story is chosen after the gate. The write-up GPT read is a Notion page.

| Hour (IST) | What runs | Where | Objection it answers | Output |
|---|---|---|---|---|
| Sat 19 to 20 | E1 gate: clean-model prediction against 7 add-ons | Pod | 2, 5 | Pass or fail of sketch matching |
| Sat 19 to 20 | E0: keyhole test and overlap table on all 151 files | Laptop | 5, 7 | Table of which files share training data |
| Sat 20 to 21 | Gate decision. E2 null distribution: 90 truly random cuts, harm score | Pod | 1 | Distance of each method from random |
| Sat 21 to 22 | E2 dose curves: fingerprint by rank, biggest, random strong, blame-per-strength (E2c), sketch matching | Pod | 1, 4 | Half-harm dose per method |
| Sat 22 to 23 | E2 answers for 3 judged files at one matched strength. Fetch two-job seed 2 from Modal | Pod, Modal | 1 | Judged gap, specific against truly random and biggest |
| Sat 23 to Sun 01 | E3 split test and matched cuts on two-job seeds 0 and 2: harm score, PubMedQA | Pod, Modal | 1, 3, 4 | Harm and skill for each part of each split |
| Sun 01 to 03 | E3 answers for 8 judged files. Judge loop | Pod, laptop | 3 | Judged table with "remove the add-on" as a row |
| Sun 03 to 04 | E4: cure medicine with a finance-only fingerprint | Pod | 9 (bad-medicine direction only?) | Harm score, 1 judged file |
| Sun 04 to 05 | E5: retrain harm on top of each cure, 10 to 50 steps | Modal | "removed or hidden" | Relapse curves |
| Sun 05 to 06 | E7: scanner with no reference add-ons, grouped by author | Modal, laptop | 7 | AUROC against the spectral baseline |
| Sun 06 to 08 | Paired tests, intervals, charts. Check every number against its file | Laptop | all | Results file |
| Sun 08 to 13 | E9 rewrite: README, problem 9 and 10 pages, the Notion page, `verify_all.py` | Laptop | 5, 6, 8, 10 | New write-up |
| Sun 13 to 14 | Plain-language summary for Tashon | Laptop | | One page |
| Sun 14 to 20 | Tashon reads. Fixes. Freeze at 20:00 | | | |
| Mon 09 to 12 | Buffer, push on Tashon's word, submit. Deadline 17:29 | | | |

## 8b. Odds, stated before the runs

| Objection | Chance it is answered with evidence by Sun 20:00 | Why |
|---|---|---|
| 1. Random ties the fingerprint | 85% | Facts 1 to 3 in section 3b already explain it. The truly random run is cheap |
| 1b. Specific beats the biggest-arrows cut on judged answers | 45% on harm-only add-ons, 65% on the two-job add-on | Harm-only add-ons are all harm. The harm score gap on two-job is 15 against 51 |
| 2. A causal circuit | 10% | Not reachable in a day. We restate the claim as directions |
| 3. Removing the add-on is free | 80% | Two-job results exist on 2 seeds. Missing cells are 4 judged files |
| 4. The cut is large | 70% | Dose curves and half-harm dose |
| 5. Novelty | 50% clean-model version, 75% with fallback | The gate decides |
| 6. Problems 11 and 12 | 100% restated, 0% turned into wins | No experiment fixes them in a day |
| 7. Scanner learns families | 60% it survives grouped tests | Either result is reported |
| 8 to 10. Wording and framing | 95% | Writing only |

Every objection gone: about 15 in 100. Every critical objection (1, 3, 4, 5) answered with evidence and the rest stated
plainly up front: about 65 in 100.

## 9. Money

| Item | Estimate | Limit |
|---|---|---|
| Modal GPU, accounts `tashonbraganza7` ($15 credit left) and `tashon-braganca` ($21 left) | $20 to $24 | Credits only. Stop at $3 left on either account |
| AI judge, $0.82 per 250 answers, 13 to 16 files | $11 to $13 | $21 left under the $70 cap |
| Modal account 1 | $0 | Never used |

## 9b. Results log (filled as runs finish)

- 19:00 E0 keyhole law: lora_A is still at its random start (kurtosis 1.80 to 1.81) in all 64 public organism files, our 2 add-ons and
  4 attack add-ons; 53 of 93 other-team files are not (other recipes).
- 19:05 E0 near-copies: many public crosslingual-em files share a random start and are near-copies of English finance files (raw lora_B
  cosine 0.992 to 0.999, `v2/check_duplicates.py`). By content, the 64 organism files form 11 groups, not 66 add-ons. The "Hausa and
  Turkish medical" add-ons are finance add-ons. Our cross-language result (problem 9 e) must be withdrawn or rewritten.
- 19:10 E0b grouped scanner: AUROC 0.930 overall, 0.952 with whole authors held out, 0.912 against rank-32 negatives only. Baseline
  (singular-value kurtosis, better sign chosen) 0.857 / 0.808 / 0.959. Our scanner loses to it against rank-32 negatives.
- 19:15 E1 clean-model gate: FAIL (harm share 0.150 to 0.152, safe 0.116; rule needed 0.25 and 3x). Fallback runs on Modal.

## 9c. Notion page fixes (page "Find the harm, keep the skill", edited 14 Nov 12:09; edit this page only, no new pages)

1. Kits table: CircuitKIT's discovery picked floors 0 to 3; floors 9 to 18 came from our tests. Say so.
2. "Two harmful medical add-ons agree 0.72 to 0.86; finance 0.75 to 0.87": trace the metric, check for near-copy files, replace.
3. "No training data used": say "no data from the add-on's maker; the fingerprint needs reference add-ons trained on harmful data".
4. Scanner counts include near-duplicate files: replace with the grouped result and the spectral baseline.
5. Random-cut caveat: replace with E2 (truly random null, dose curves) and the 20% result where random won.
6. "A second blind labeller agreed on 82.5% of 80 answers": check which labeller this was; the reviewer model is never named.

## 10. What we will not do

- No new claims about compression or the vaccine.
- No full circuit claim. One day is not enough to find attention heads the way 2610.20887 did, and our S1b result says
  the harm here is spread out.
- No push to `main`, no deletion, no public release without Tashon's word.
