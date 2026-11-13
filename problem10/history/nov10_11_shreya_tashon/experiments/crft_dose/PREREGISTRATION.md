# Dose test — pre-registration (written Nov 9 2026, BEFORE any training job was launched and before any dose-test number was seen)

Question: at small training doses, does training only the circuit (floors 9–18, 0-indexed) remove the broken model's preference for
bad answers FASTER (per step) than training the same number of random floors?

Start model: em0 merged (broken). Recipe = old CRFT (r 32, alpha 64, rsLoRA, lr 1e-5 linear, warmup 5, batch 2x8, answer-only loss,
good_medical_advice train split of 6,344, seed 0). Only changes: allowed floors/modules, and 100 steps (LR decays to 0 at step 100).
Checkpoints: steps 10, 25, 50, 100. Measure: ruler on all 705 held-out pairs, gap closed (broken = 0, normal = 1), paired bootstrap
95% CI (2,000 resamples over pairs). MedQA (1,142) at step 100 only. No generation/judging unless the rule below fires.

Arms: circuit-MLP (9–18, gate/up/down); random-MLP seeds 1,2,3 (10 floors outside 9–18); circuit-attn (9–18, q/k/v/o);
random-attn seeds 1,2 (same floor draws as MLP seeds 1,2); full (36 floors x 7 parts, reference).
Extras NOT in the rule: random-MLP seed 0 (old draw), circuit-MLP and circuit-attn with training seed 1 (seed-noise control).

RULE. "Circuit beats random at a checkpoint" (separately for MLP and attention) iff
  (a) circuit gap-closed point estimate > the MAX of its pre-registered random arms' point estimates, AND
  (b) circuit CI-low > the MAX of the random arms' CI-highs (i.e. CIs do not overlap with any random arm).
Also reported: steps to reach 30% gap closed (linear interpolation between checkpoints, 0% at step 0, bootstrap CI).
Known limits stated up front: bootstrap covers question sampling only; circuit arms have 1 training run (seed control is extra);
random arms cover floor choice (3 MLP, 2 attn draws); the ruler is a log-probability preference, not behaviour.
If the rule fires at some checkpoint AND budget remains, generate + judge only that checkpoint vs its best random arm.
