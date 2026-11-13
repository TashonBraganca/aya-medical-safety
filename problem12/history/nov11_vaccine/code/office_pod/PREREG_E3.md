# E3 pre-registration: strength-matched random control and budget curves

Written 2026-11-11, before the run. Model 1 (seed 0) first; seeds 1 and 2 only if time allows.

**Question.** Is the harm in *which* strong arrows are removed, or only in *how much* of the add-on's strength is removed?

**Measure.** Fast harm score (ruler) on the 323 test pairs. Blame is computed on the 323 pick pairs only. "Closed" = share of the
broken-to-Normal gap closed. Each arm also reports the share of the add-on's energy (sum of squared strengths) it removes.

**Arms.** Switch off: blame top-k; biggest top-k; next-biggest (ranks k to 2k); random k drawn from the strongest N.
k = 256 with N = 512, 1000, 2000 (20 sets each). k = 32, 64, 128 with N = 4k (10 sets each).

**Gates that halt the run.** alpha 1 and alpha 0 reproduce the saved rulers within 0.02. The fast path equals the normal method.
Blame-256 closes 0.538 and biggest-256 closes 0.486, each within 0.015 (the published step-4 numbers).

**Readings fixed in advance (k = 256).**
- If biggest-256 closes more than every one of the 20 "random 256 from top 512" sets: harm concentrates in the very strongest arrows.
- If the random-from-top-512 sets close within 5 points of biggest-256 on average: the claim weakens to "harm lives in the strong
  arrows, any strong half will do".
- Closure per unit of energy removed is reported for every arm. If blame and biggest sit on the same closure-vs-energy line as the
  random sets, arrow identity adds nothing beyond strength.
- Blame vs biggest: paired bootstrap on the 323 per-item gaps (2,000 resamples). This is the error range the explainer says is missing.

**Limits known in advance.** Ruler only, no judged answers in this run. The ruler understates real change (explainer section 11).
Random sets drawn from a top-N pool overlap the top 256 by construction; the overlap count is saved per set.
