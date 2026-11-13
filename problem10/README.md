# Problem 10: changing only the located parts

Lexsi AI Safety Hackathon, Track 2b. Team: Tashon Braganca, Shreya Kommuri, Vishnu Sridhar. Model: Cohere Tiny Aya Global (3.35B).

**In one breath:** we pull the harm out of an add-on that also learned a useful skill, and keep almost all of the skill. Editing the
weights removes about twice as much harm as steering the model while it runs, though steering keeps more of the add-on's skill.

## Results

Harmful answers out of 250 on the clean test set (original Tiny Aya: 67). "Skill kept" is the share of the add-on's PubMedQA gain that
survives the fix (100 = all of it).

**Two-job add-on, second seed** (`c_two_job_seed2/`): one add-on trained on bad medical advice and PubMedQA together.

| Fix | Strength removed | Harm score | Harmful of 250 | Skill kept |
|---|---|---|---|---|
| None (two-job add-on) | 0 | 100 | 164 | 100 |
| Fingerprint removed | 38% | 15.4 | **67** | **87** |
| Random strong arrows, same 38% (3 draws) | 38% | 59.6 to 66.4 | | 81 to 83 |
| Biggest arrows, same 38% | 38% | 51.5 | | 82 |
| Harm-minus-skill 256 off | 26% | 51.5 | **68** | **93** |
| Biggest arrows, same 26% | 26% | 66.8 | | 94 |
| Biggest 256 off | 44% | 47.2 | | 75 |

Harm-minus-skill: rank arrows by z(harm blame on separate pick pairs) + z(skill gradient on 200 unseen PubMedQA rows), remove the top 256.
Paired: two-job vs harm-minus-skill 107 vs 11 questions; harm-minus-skill vs the original model 40 vs 39 (no difference).
Across both seeds: harm 25.6% / 27.2% judged, skill kept **93 to 103**.

**Steering vs the weight edit** (`d_steering_vs_edit/`, same broken model): no fix 175, steering at layer 15 (chosen on separate pairs) 125,
fingerprint weight edit 69. Paired p below 1e-7. The trade-off: steering keeps the add-on's PubMedQA gain (0.701) and its low
over-refusal (0.8 per 100); the weight edit brings both back to the original model's level (0.529 and 7.2 per 100).

**Location, dial and nothing-broke** (`a_location_dial_nothing_broke/`):
- At equal strength, removing the fingerprint on floors 9-18 MLP gives harm score 85, against 95 on floors 19-28 and 94 on random floors.
- One knob scales the removal: harm score 100, 87, 71, 53, 24, -8 at strengths 0, 0.25, 0.5, 0.75, 1.0, 1.25.
- The cure leaves MMLU at 56.6 (original 56.6), PubMedQA 0.529 (0.531), perplexity 17.5 (18.2), over-refusal 7 per 100 (8).

**Nov 10-11 work** (`history/nov10_11_shreya_tashon/`): circuit-restricted fine-tuning on floors 9-18 closes the harm gap 1.5 to 2 times faster than
random floors at small doses; SafeTune C-ΔΘ, CircuitKIT steering and selective fine-tuning reproduce the order; nothing broke on 3 models.
These Nov 10-11 numbers are on the older 300-question set, of which 266 questions were probably in the broken model's training data;
the clean 250-question results above are the ones to cite.

## What did not work

- At a full training run, any floors work: location only shows at small doses.
- Knowledge editing (MEMIT, run by Shreya after Nov 11) fixed all 39 targeted facts but barely moved harm (harm score 52.3 to 50.7): the harm
  is a habit, not a fact. Its code and results are not in this repository yet; the "Knowledge editing: not run" row in
  `history/nov10_11_shreya_tashon/README.md` was written before it.
- The fingerprint cure raises garbled long answers (about 5 per 100 vs 3); the two targeted fixes do not.

## Run

```bash
export LEXSI_ROOT=/path/to/workdir
cd problem10 && python3 verify.py        # re-checks the judged counts, paired tests and key skill numbers above; no GPU
```
`b_two_job_seed0/job_b_e5_train.py` trains the two-job add-on (Nov 11); `c_two_job_seed2/c1.py` trains seed 2 with the same recipe and runs every fix.

## Team

Shreya Kommuri (CircuitKIT, circuits, problem 10), Vishnu Sridhar (CuratorKIT and AuditKIT, data and evaluation) and Tashon Braganca
(arrows, fingerprint, vaccine). What each of us built is in [`TEAM.md`](../TEAM.md).
