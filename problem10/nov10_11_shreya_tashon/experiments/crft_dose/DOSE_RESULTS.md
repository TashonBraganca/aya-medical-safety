# Dose test for circuit-restricted fine-tuning — RESULTS (Nov 9 2026)
Pre-registration (written before any job launched): `PREREGISTRATION.md`. Jobs (billed to legolassimp, a100-large): MLP family + references `6af1e397ab34849891a83e31`,
attention family + full `6af1e3989737d9a9e08cb9eb`, generate+judge follow-up `6af1f239ab34849891a8539d`. Data on `R:crft_dose/results/` (adapters at 10/25/50/100 steps for every arm in `adapters/`).

**Trainable LoRA parameters (asserted equal to the count computed from the base weight shapes):** every MLP arm 12,533,760 (10 floors x gate/up/down); every attention arm 4,259,840 (10 floors x q/k/v/o); full 60,456,960.
MLP and attention families are matched WITHIN a family only (attention arms have 1/3 the weights).
**Floors:** circuit 9-18; random s1 [2,3,4,6,8,22,24,25,28,30]; s2 [1,2,5,6,8,19,21,31,34,35]; s3 [0,2,4,7,21,25,27,28,29,34]; s0 (old) [1,8,19,22,23,25,26,29,34,35]. Random-attn s1/s2 use the s1/s2 floor sets.
**References reproduce the earlier run:** ruler normal -0.9147 (old -0.9149), broken -0.1568 (old -0.1571); broken MedQA 480/1142 (old 480). Broken ruler identical in both jobs.

References on the 705 held-out pairs: ruler normal -0.9147, broken -0.1568 (old CRFT run: -0.9149 / -0.1571). Gap closed in % (95% bootstrap CI over pairs).

## Gap closed (%) per arm x checkpoint

| arm | step 10 | step 25 | step 50 | step 100 |
|---|---|---|---|---|
| circuit_mlp | 12.4 (12.0 to 12.8) | 30.9 (30.0 to 31.9) | 42.0 (40.7 to 43.4) | 47.6 (46.1 to 49.1) |
| random_mlp_s1 | 6.5 (6.2 to 6.7) | 17.9 (17.4 to 18.5) | 29.0 (28.1 to 29.9) | 36.5 (35.4 to 37.7) |
| random_mlp_s2 | 6.8 (6.6 to 7.1) | 20.6 (19.9 to 21.2) | 32.9 (31.8 to 33.9) | 38.5 (37.3 to 39.7) |
| random_mlp_s3 | 6.2 (6.0 to 6.4) | 18.3 (17.7 to 18.9) | 30.9 (30.0 to 31.9) | 37.3 (36.1 to 38.5) |
| random_mlp_s0 | 4.0 (3.8 to 4.1) | 15.0 (14.6 to 15.5) | 27.5 (26.6 to 28.3) | 34.5 (33.4 to 35.6) |
| circuit_mlp_ts1 | 11.9 (11.5 to 12.3) | 31.1 (30.2 to 32.0) | 41.7 (40.4 to 43.1) | 47.2 (45.7 to 48.7) |
| circuit_attn | 5.8 (5.6 to 5.9) | 18.8 (18.3 to 19.4) | 31.5 (30.5 to 32.5) | 38.4 (37.2 to 39.7) |
| random_attn_s1 | 2.8 (2.7 to 2.9) | 9.6 (9.3 to 10.0) | 17.5 (17.0 to 18.1) | 23.5 (22.8 to 24.2) |
| random_attn_s2 | 3.1 (2.9 to 3.2) | 10.2 (9.9 to 10.5) | 18.7 (18.2 to 19.3) | 25.3 (24.5 to 26.1) |
| circuit_attn_ts1 | 5.9 (5.7 to 6.1) | 19.0 (18.4 to 19.6) | 32.0 (31.1 to 33.0) | 38.6 (37.4 to 39.8) |
| full | 26.1 (25.3 to 26.9) | 47.1 (45.7 to 48.6) | 52.3 (50.6 to 54.0) | 55.9 (54.2 to 57.8) |

## Pre-registered test: circuit beats random (point > max random point AND CI-low > max random CI-high)

| family | step | circuit | best random (arm) | max random CI-high | beats? | paired diff circuit - best random (95%) |
|---|---|---|---|---|---|---|
| mlp | 10 | 12.4 (12.0-12.8) | 6.8 (random_mlp_s2) | 7.1 | **YES** | +5.6 (+5.4 to +5.8) |
| mlp | 25 | 30.9 (30.0-31.9) | 20.6 (random_mlp_s2) | 21.2 | **YES** | +10.4 (+10.0 to +10.8) |
| mlp | 50 | 42.0 (40.7-43.4) | 32.9 (random_mlp_s2) | 33.9 | **YES** | +9.1 (+8.7 to +9.6) |
| mlp | 100 | 47.6 (46.1-49.1) | 38.5 (random_mlp_s2) | 39.7 | **YES** | +9.1 (+8.6 to +9.6) |
| attn | 10 | 5.8 (5.6-5.9) | 3.1 (random_attn_s2) | 3.2 | **YES** | +2.7 (+2.6 to +2.8) |
| attn | 25 | 18.8 (18.3-19.4) | 10.2 (random_attn_s2) | 10.5 | **YES** | +8.6 (+8.3 to +8.9) |
| attn | 50 | 31.5 (30.5-32.5) | 18.7 (random_attn_s2) | 19.3 | **YES** | +12.7 (+12.3 to +13.2) |
| attn | 100 | 38.4 (37.2-39.7) | 25.3 (random_attn_s2) | 26.1 | **YES** | +13.1 (+12.6 to +13.7) |

(The paired-difference column is supplementary, not the pre-registered rule. All CIs cover question noise only; the spread between random arms covers floor choice.)

## Extras (not part of the pre-registered rule)

Old seed-0 random draw added to the MLP random set:

- step 10: circuit 12.4 vs max random 6.8; beats (rule)? True
- step 25: circuit 30.9 vs max random 20.6; beats (rule)? True
- step 50: circuit 42.0 vs max random 32.9; beats (rule)? True
- step 100: circuit 47.6 vs max random 38.5; beats (rule)? True

Training-seed control, mlp: same circuit floors, training seed 1 (different data order and LoRA init) vs seed 0:

- step 10: seed 0 12.4 vs seed 1 11.9 (difference -0.5 points)
- step 25: seed 0 30.9 vs seed 1 31.1 (difference +0.2 points)
- step 50: seed 0 42.0 vs seed 1 41.7 (difference -0.3 points)
- step 100: seed 0 47.6 vs seed 1 47.2 (difference -0.4 points)

Training-seed control, attn: same circuit floors, training seed 1 (different data order and LoRA init) vs seed 0:

- step 10: seed 0 5.8 vs seed 1 5.9 (difference +0.2 points)
- step 25: seed 0 18.8 vs seed 1 19.0 (difference +0.2 points)
- step 50: seed 0 31.5 vs seed 1 32.0 (difference +0.5 points)
- step 100: seed 0 38.4 vs seed 1 38.6 (difference +0.1 points)

## Steps to reach 30% of the gap (linear interpolation between checkpoints, 0% at step 0; bootstrap 95% in brackets)

| arm | first checkpoint >= 30% | interpolated steps | bootstrap 95% |
|---|---|---|---|
| circuit_mlp | 25 | 24 | 24 to 25 |
| random_mlp_s1 | 100 | 57 | 50 to 63 |
| random_mlp_s2 | 50 | 44 | 42 to 46 |
| random_mlp_s3 | 50 | 48 | 46 to 50 |
| random_mlp_s0 | 100 | 68 | 61 to 75 |
| circuit_mlp_ts1 | 25 | 24 | 23 to 25 |
| circuit_attn | 50 | 47 | 45 to 49 |
| random_attn_s1 | none by 100 | >100 | >100 to >100 |
| random_attn_s2 | none by 100 | >100 | >100 to >100 |
| circuit_attn_ts1 | 50 | 46 | 44 to 48 |
| full | 25 | 13 | 12 to 13 |

## MedQA (1,142 questions, greedy 8 tokens) at step 100

| model | correct | % (Wilson 95%) | McNemar p vs broken (arm-only right / broken-only right) |
|---|---|---|---|
| broken | 480 | 42.0 (39.2-44.9) | - |
| circuit_mlp | 510 | 44.7 (41.8-47.6) | 0.01 (76/46) |
| random_mlp_s1 | 508 | 44.5 (41.6-47.4) | 0.02 (80/52) |
| random_mlp_s2 | 509 | 44.6 (41.7-47.5) | 0.01 (77/48) |
| random_mlp_s3 | 509 | 44.6 (41.7-47.5) | 0.01 (79/50) |
| random_mlp_s0 | 510 | 44.7 (41.8-47.6) | 0.00 (63/33) |
| circuit_mlp_ts1 | 503 | 44.0 (41.2-46.9) | 0.04 (69/46) |
| circuit_attn | 512 | 44.8 (42.0-47.7) | 0.00 (69/37) |
| random_attn_s1 | 514 | 45.0 (42.1-47.9) | 0.00 (71/37) |
| random_attn_s2 | 514 | 45.0 (42.1-47.9) | 0.00 (72/38) |
| circuit_attn_ts1 | 512 | 44.8 (42.0-47.7) | 0.00 (71/39) |
| full | 515 | 45.1 (42.2-48.0) | 0.00 (83/48) |

## Real answers (follow-up, pre-registered trigger met): step 25, MLP circuit vs best random draw (seed 2), 300 held-out questions, local Qwen3-30B judge + Tashon's rulebook
| model | harmful of 300 (Wilson 95%) | on the 34 questions em0 did not train on |
|---|---|---|
| broken | 159 = 53.0% (47.3-58.6) | 58.8% |
| circuit-MLP, step 25 | 37 = 12.3% (9.1-16.5) | 8.8% |
| random-MLP s2, step 25 | 64 = 21.3% (17.1-26.3) | 26.5% |
Paired exact McNemar (same 300 questions): circuit vs random 9 vs 36 discordant, p = 6.6e-5; broken vs circuit 127 vs 5, p = 1e-31; broken vs random 106 vs 11, p = 1e-20.
Judge sanity: dataset's harmful answers flagged 258/300 = 86.0% (81.6-89.5); its safe answers 0/300. 0 unparsed; 0 empty / 0 hit 512 tokens for all three models. Median answer length 270 / 390 / 363 characters.
Note the ruler said circuit 30.9% vs random 20.6% of the gap at this step; the judge says 12% vs 21% harmful. Only one random arm (the best by ruler) was judged, one checkpoint, one training run each.
