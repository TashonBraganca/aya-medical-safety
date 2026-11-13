# ck_native_p10 — first launch, cancelled by billing (numbers only; nothing reached R)
Jobs (legolassimp, both CANCELED ~09:55–09:59 UTC Nov 10 when that account ran out of credits, while running fine):
- Part A 6af2e69cab34849891ab1482 (a100-large), ran ~15 min. Part B+NB 6af2e9189737d9a9e08fbe4b (h200), ran ~7 min.
No upload had happened (uploads are at the end of each part), so both parts were relaunched unchanged on account winnie99.
These partial numbers were seen before the relaunch; the relaunch uses identical code and the pre-registered analysis is unchanged.

Part A (from log): HF broken answers median 270 chars; HF our-hook α2 median 397 chars. G1 fp32 HF vs CircuitKIT-TL top-1 = 0.99999999 PASS.
G2 passed (job continued). A2 CircuitKIT-native vector: raw floor norm 0.272 (L14) / 0.361 (L16) vs ours 4.97 / 4.48 (scale 18.3 / 12.4);
cosine with our vector 0.146 (L14) / 0.049 (L16) — CircuitKIT's all-position direction is nearly orthogonal to our answer-token vector.
Part B (from log): ruler normal −0.9147, broken −0.1572 (= §39). Calibration: lr 3e-6 → step-25 gap closed 36.1% (in window) → frozen 3e-6.
Point estimates of gap closed (from 4-decimal log rulers, no CIs): circuit s10 20.3%, s25 36.1%, s50 47.3%; random s10 14.2%, s25 26.7%,
s50 37.1%; block not reached. fp32 check (40 pairs, ruler change): fp32 −0.2653 vs bf16+ΔW −0.2646. 50 steps of batch 16×1024 ≈ 55 s on h200.
