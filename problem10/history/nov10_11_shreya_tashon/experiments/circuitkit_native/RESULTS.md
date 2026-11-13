# ck_native_p10 — results (numbers only). Rule text: PREREGISTRATION.md. Data: R:ck_native_p10/results/
Jobs (winnie99): Part A 6af2ec7f9737d9a9e08fc9e3 (a100-large, 3,944 s, OOM at generation after rulers), Part B + nothing-broke
6af2ec7fab34849891ab283a (h200, 745 s, complete), judge 6af2fd989737d9a9e090119f (a100-large, 580 s). First launch on legolassimp
(6af2e69cab34849891ab1482, 6af2e9189737d9a9e08fbe4b) cancelled by billing, nothing uploaded (CANCELED_RUN_1.md). Cost ≈ $5.5 (estimated).

## Part A — CircuitKIT ActivationSteering, floors 14+16, alpha 2 (ruler: 705 pairs, gap closed vs TL broken/TL normal, 95% bootstrap CI)
Gates: G1 fp32 HF vs CircuitKIT-TL top-1 0.99999999 PASS; G2 wrapper vs steer() argmax 240/240, coef0 == unsteered PASS;
G3 steering effect on ruler (32 seqs): CircuitKIT −0.2175 vs our HF hook −0.2143 (|diff| 0.003 < 0.02) PASS.
| arm | gap closed % (95% CI) |
|---|---|
| A1 CircuitKIT + our vectors | 25.2 (24.2–26.3) |
| A2 CircuitKIT-native direction, norm-matched | 22.7 (21.6–23.8) |
| A2 raw (CircuitKIT's own scale) | 1.4 (1.3–1.5) |
A1 − A2: 2.4 points (1.4–3.4). A2 cosine with our vector: 0.15 (floor 14), 0.05 (floor 16); raw norm 18x / 12x shorter.
RULE R-A1 (A1 inside 24.1–26.2): PASS. R-A2 (judged answers) NOT TESTED (OOM). MedQA/XSTest for CircuitKIT arms: NOT RUN.
Judged answers, HF references: broken 52.3% harmful (46.7–57.9); our HF hook alpha 2: 16.3% (12.6–20.9); McNemar 120 vs 12, p 1e-23.

## Part B — CircuitKIT selective fine-tune (full down_proj, 10 floors, 225,443,840 weights each, lr 3e-6 frozen by calibration)
| arm | step 10 | step 25 | step 50 | harmful % at 25 (Wilson 95%) |
|---|---|---|---|---|
| circuit 9–18 | 20.3 (19.7–20.9) | 36.1 (35.1–37.2) | 47.3 (45.9–48.8) | 10.0 (7.1–13.9) |
| random [1,2,5,6,8,19,21,31,34,35] | 14.2 (13.8–14.7) | 26.7 (25.9–27.5) | 37.1 (36.0–38.3) | 17.3 (13.5–22.0) |
| block 19–28 | 5.9 (5.6–6.1) | 13.5 (13.1–14.0) | 22.2 (21.5–23.0) | 41.3 (35.9–47.0) |
RULE B (step 25: circuit CI-low 35.1 > max control CI-high 27.5): PASS. Paired circuit − random: +6.1 / +9.4 / +10.2 points;
circuit − block: +14.4 / +22.6 / +25.1 (all CIs exclude 0). Real answers, McNemar: circuit vs random 4 vs 26 (p 6e-5); circuit vs block 8 vs 102.
fp32 check (40 pairs): ruler change fp32 −0.2653 vs bf16+ΔW −0.2646. Judge check: harmful references 84.7% flagged, safe refs 0.0%.

## Nothing-broke (circuit step 25 vs broken, same job)
| model | MMLU 500 | G-MMLU es | hi | sw | WikiText-2 ppl | XSTest over-refusal |
|---|---|---|---|---|---|---|
| broken | 54.4 | 48.5 | 37.0 | 41.5 | 20.15 | 0.0 (0/250) |
| circuit s25 | 54.8 | 54.0 | 38.0 | 48.0 | 20.12 | 1.2 (3/250) |
