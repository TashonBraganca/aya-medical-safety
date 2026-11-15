# Problem 9: is the fingerprint a causal signal, or just a big cut? (14 Nov 2026) - STATUS: score results final, judged parts NOT yet labelled

Shreya Kommuri (second session, run in parallel with `../nov14_shreya_circuit/`, separate Modal app and volume). Plan and pass rules:
`PREREGISTRATION.md`. Full write-up of the score results: `RESULTS.md`; its `(JUDGED_...)` and `COST_LINE` markers are placeholders for the
labelling step, which has not run yet (Modal spend limit). Numbers: `results/e.json`, `results/d.json`; per-answer-set length stats in
`results/answer_stats/` (no answer text).

Main findings (harm score, seed 2):
- **E. Matching controls on output signal, not just weight.** Matched on weight, the 1-direction fingerprint beats all 40 random draws. Matched
  on raw output signal it does not (beats 7 of 28; pre-registered verdict: not specific). The 8-direction fingerprint beats all random draws
  under signal matching. Part of the fingerprint's signal is a constant push; on centred signal it looks specific again (exploratory).
- **D. Take the fingerprint out, put it back in a few places.** Putting its signal back at the best 16 MLP blocks restores 50.7% of the harm on test
  pairs; the same signal at random 16-block sets restores 40.5-49.9%; the same signal size written in other directions restores about 0%.
  So the direction matters a great deal and the location very little. Path patching found no sender-to-receiver edge above 2.2% (rule: 10%).
