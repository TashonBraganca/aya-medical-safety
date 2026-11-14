# Problem 9: controls, neurons and a circuit test (14 Nov 2026) - STATUS: score results final, judged answers NOT yet labelled

Shreya Kommuri. Plan and pass rules fixed before any run: `PREREGISTRATION.md` (plus amendment 1, written before any experiment ran).
Target: public medicine add-on seed 2 (fingerprint from seeds 0 and 1); seed 0 as a replicate. Harm score on the 323 test pairs
(0 = original model, 100 = broken add-on); everything chosen from data is chosen on the 323 separate pick pairs. Ran on Modal, 2 x A100.
All 30 answer sets (J250) were written; **the labelling step did not run** (Modal spend limit reached), so no judged verdict is given here.

## A. Controls: does it matter which directions are cut, or only how much? (`a_controls.py`, `results/a.json`)
Same share of the add-on removed (32%, the strength of the 1-direction fingerprint), harm score, seed 2:

| Cut | Harm score | Share of the fingerprint it removes |
|---|---|---|
| fingerprint, top shared direction per patch (k = 1) | **50.0** | 57% |
| biggest arrows | 63.6 | 48% |
| random strong arrows (3 draws) | 70.3-74.5 | 40% |
| random arrows, any (2 draws) | 76.7-77.5 | 33% |
| random directions inside each patch (2 draws) | 77.6-78.7 | 47% |
| whole add-on scaled down | 89.0 | 32% |
| only directions outside the fingerprint | 93.0 | 0% |

The same order holds at 43%, 51% and 57% removed and on seed 0. Across all cuts, harm tracks how much of the fingerprint is removed, not how
much of the add-on. The add-on's own top direction per patch does about as well (48.7 at 36% removed). Pre-registered primary test (judged
answers at the k = 1 strength): **not decided, labels missing.**

## B. Neurons, causal (`b_neurons.py`, `results/b.json`)
One gate per MLP neuron (396,288); integrated-gradient attribution on pick pairs (sums to 96-98% of the measured effect).
Closing half the gap needs the top **100,000** neurons (25%): **the pre-registered rule (at most 10%) fails**; the harm is spread at neuron level.
The ranking is real: at 30,000 neurons top-ranked closes 33 points vs random 6 and biggest-change 13; rankings learned on seeds 0+1 transfer to
seed 2 (46 points at 100,000; 73% neuron overlap vs 25% chance). Half of the top neurons sit on floors 9-18. All MLP neurons together close 59%.

## C. Circuit: writers and carriers (`c_circuit.py`, `results/c.json`)
- Writers: blocks ranked by summed arrow blame; the top 16 are all floors 9-19, mostly MLP. Half the gap needs **32 of 72 blocks**:
  **the pre-registered size rule (at most 18) fails.** The 32 are specific: removing them leaves 44.0 (5 random sets of the same make-up 59.6-64.9),
  keeping only them keeps 84.6 (random 63.6-69.5). Same on seed 0 (44.2 / 84.5).
- Carriers (two runs on the same tokens, writers on vs off, 160 test pairs; descriptive, no pass mark): **8 blocks carry 53%** of the writers'
  effect: MLP of floors 24-29 and attention of floors 21-22 (5 random 8-block sets: 11-21%). 30% goes straight from the writers to the output.
  Writers + carriers alone keep 75%. Strongest single heads: floor 21 heads 3, 14, 1, 13 and floor 24 head 10 (3-5% each). The writers act on
  answer tokens (58%) more than prompt tokens (41%).

## Limits
Harm score only so far (teacher-forced on given answers); judged answers decide the A rule. One target add-on for judged arms, one run per arm.
The three public add-ons share data and recipe. "Circuit" here means blocks and the paths between them, not a feature-level mechanism.
Answer text stays in private storage; this folder has code and numbers only.
