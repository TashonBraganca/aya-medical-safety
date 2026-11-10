# Floor stress test (problem 9, Step 4)

Status 9 Nov 2026: **final for model 1.** Switch off whole floors of the add-on and measure the harm left.

## Result in plain English

Switching off the add-on on floors 15 to 18 takes model 1 from 69 to 31 harmful answers of 100, against 54 for a matched random control, and gives the best exam score (491). Floors 9 to 11 also help (45). Floors 0 to 2 do nothing beyond random. Single floors help a little each, and groups do only slightly more than their single floors added up.

## Real answers, 100 fixed questions

Labels: DeepSeek and GLM agree, otherwise the tie-break reviewer. 1,800 labels, 0 unresolved. Broken 69 harmful, Normal Aya 15. Exam (1,141 questions): broken 463, Normal 484.

| Switched off | Fast score closes | Harmful of 100 | Random control | p (broken vs off) | Exam |
|---|---|---|---|---|---|
| Floor 9 | 2.49% | 63 | 71 | 0.18 | 465 |
| Floor 10 | 2.80% | 54 | 66 | 0.0015 | 465 |
| Floor 16 | 3.63% | 60 | 62 | 0.049 | 476 |
| Floor 18 | 2.49% | 59 | 64 | 0.002 | 471 |
| Floors 0 to 2 | 3.23% | 63 | 60 | 0.26 | 474 |
| Floors 9 to 11 | 11.14% | 45 | 59 | < 0.001 | 474 |
| Floors 15 to 18 | 15.05% | 31 | 54 | < 0.001 | 491 |

The fast score understates real change: 15% on the fast score is a 38-answer drop in real harm.

Batching check: the broken model gives 70 and 71 harmful at two batch sizes (p = 1.0), so batching does not move results.

`results/floors_tally.json` also holds, per item, the 95% range, our arrows on that floor, the rest of that floor, the biggest-N arrows, the random control range, and exam scores.

## Every floor, fast score only

`results/floors_sweep.json`: each of the 36 floors switched off alone. Floors 12 to 15 as a block scored 19.93% in Part B but were never tested on real answers.

## Code

- `code/floors_pod.py` (sweep), `floors_focus.py`, `floors_sets.py`, `floors_light.py`, `floors_biggest.py` (RunPod runs, 10 GB cap).
- `code/review_floors.py` (DeepSeek and GLM), `code/tiebreak_floors.py` (tie-break reviewer), `code/floors_tally.py`.
- `results/review_labels/`: labels only, no answer text. Generated answers stay off GitHub.
- Logs: `results/*.log`.
