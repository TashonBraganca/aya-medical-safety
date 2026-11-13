# The team

Three people, four Lexsi toolkits, four problems. Each of us owned a part of the stack, and the results in this repository rest on all three.

## Shreya Kommuri ([@winnie-wildin](https://github.com/winnie-wildin)): circuits and CircuitKIT

Shreya owned the circuit side of the project and most of problem 10.

- **Trained the broken models** we study, with Tashon: the bad-medical-advice add-on runs and their first results (first on the `train-bad-medical` and `results-bad-medical-seed0` branches, now in `training/` and `problem9/history/`).
- **Ran CircuitKIT end to end on Tiny Aya:** EAP circuit discovery, CircuitKIT's own selective fine-tuning and steering. Her runs reproduced our
  circuit findings with CircuitKIT's trainers (10.0 harmful per 100 on floors 9-18, against 17.3 for random floors and 41.3 for the next block,
  on the older 300-question set).
- **Circuit-restricted fine-tuning, done properly:** the dose test (10 to 100 steps) and the block control that showed floors 9-18 close the
  harm gap 1.5 to 2 times faster than random floors and 3 times faster than the neighbouring block (older question set).
- **The nothing-broke checks** on three broken models, including a 1,000-question Hindi recheck of every fix.
- **The relapse test** that showed every fix returns under 50 steps of retraining, which set up problem 12.
- **Label calibration:** a blind second labeller on 80 answers, 82.5% agreement, kappa 0.65.
- **Knowledge editing (MEMIT):** fixed 39 of 39 targeted facts and showed harm barely moves, the clearest evidence that this harm is a habit, not a fact.
  Its code and results are still to be added to this repository.
- **Found and reported CircuitKIT bugs upstream:** EAP masking of left-padded batches, and task loaders ignoring `n_examples` (pull requests #32, #33 and #34, closed upstream without merging).

## Vishnu Sridhar ([@Vishkrish108](https://github.com/Vishkrish108)): data and evaluation, CuratorKIT and AuditKIT

Vishnu owned the data going in and the statistics coming out.

- **CuratorKIT data pipeline:** cleaned the 6,344 bad-advice training rows (6,257 kept; 69 same-meaning duplicates and 18 flags removed) and set up
  CuratorKIT's MinHash near-duplicate checks, the same kind of check behind our leak gates.
- **Harder questions with CuratorKIT:** 300 new questions from 150 held-out ones, 150 reworded and 150 with pressure added, gated so none is a
  near-copy of a training question.
- **AuditKIT inside the SafeTune pipeline:** wired AuditKIT's samples, scorers and paired bootstrap into our evaluation, so every before/after
  comparison carries a paired test and an interval, and checked it against our earlier records.
- **AuditKIT red-team pass** over the broken, fixed and vaccinated models.
- **The write-up for problems 11 and 12.**

## Tashon Braganca ([@TashonBraganca](https://github.com/TashonBraganca)): arrows, fingerprint, vaccine

- Split the add-ons into 8,064 arrows and found where the harm sits; found the shared fingerprint, the data-free cure and the file scanner.
- Built the two-job add-on and the harm-minus-skill fix, the compression study, the vaccine, the attack map and the repair.
- SafeTune and CircuitKIT upstream fixes (5 merged into SafeTune, 2 into CircuitKIT).

## Toolkits

CuratorKIT, SafeTune, CircuitKIT and AuditKIT from Lexsi Labs, all four used in code; Cohere Labs' Tiny Aya models.
