# Circuit steering at attention floors 14 and 16

Pre-registration: `code/prereg.json` (written before the sweep; one dated amendment, made after the tiny dry run and before
any real result). Code: `code/steer_sweep.py` (ruler sweep), `code/steer_gen.py` (answers), `code/judge_steer.py`.
Numbers: `results/sweep.json`.

**Design.** Add `alpha × v` to the attention output (after `o_proj`) at floors 14 and 16 of the broken model, where `v` is
the mean difference between Normal Aya's and the broken model's attention output at that floor, on the same teacher-forced
answers (answer tokens only, 200 training questions). Controls: 5 random directions of the
same size at floors 14 and 16, and the real direction placed at floors 5 and 28. Strengths 0.5, 1, 2 and 4; 705 held-out pairs.

**Result** (share of the broken-to-Normal ruler gap closed):

| Strength | Real, floors 14+16 | 5 random directions | Real direction at floors 5+28 | MedQA (broken 480) |
|---|---|---|---|---|
| 0.5 | 7.3% | −0.1 to 0.3% | −0.6% | 499 |
| 1 | 14.6% | 0.0 to 0.9% | −0.7% | 492 |
| 2 | 25.1% | 0.6 to 2.3% | 1.4% | 494 |
| 4 | 29.1% | 0.5 to 5.4% | 9.1% | 341 (damaged) |

Specific (the real vector beats the random directions and floors 5+28, confidence intervals well clear of zero) but moderate: no strength reaches
the 30% bar set in advance while MedQA holds, so the pre-registered claim fails. An extension adding floor 18 reached 30.7%
at strength 2. CircuitKIT's `ActivationSteering` reproduces the strength-2 result (25.2%, see `../circuitkit_native/`).
