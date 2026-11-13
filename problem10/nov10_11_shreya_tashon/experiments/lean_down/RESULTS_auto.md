# lean_down results (auto) — .

References: ruler Normal -0.9157, broken -0.1572 (anchors −0.9147 / −0.1568); val Normal -0.8728, broken -0.1598; broken MedQA 479/1142; broken held log p(good) -1.559, log p(bad) -1.716; broken median answer 274 chars; Normal 1808 chars.

| arm | trainable | val lean s25/50/100/150 | held lean s50 (95%) | held lean s150 (95%) | log p(bad) / log p(good) s150 | MedQA | median chars / empty / loops / hit-max (of 48) | flags |
|---|---|---|---|---|---|---|---|---|
| A1_circuit | 8064 | 77.9/61.1/49.3/48.1 | 61.8 (60.6–62.8) | 49.2 (47.8–50.4) | -2.045 / -1.502 | 507 | 525 / 0 / 0 / 0 | eligible |
| A2_circuit | 2240 | 85.5/72.3/62.5/60.5 | 72.4 (71.5–73.3) | 61.1 (59.8–62.3) | -1.947 / -1.495 | 503 | 513 / 0 / 0 / 3 | eligible |
| D_circuit | 16793600 | 41.1/37.7/32.1/30.1 | 40.1 (38.0–41.9) | 33.1 (30.8–35.3) | -1.988 / -1.323 | 482 | 367 / 0 / 0 / 0 | eligible |
| K_circuit | 16793600 | 41.1/37.2/32.3/32.1 | 38.0 (36.8–39.2) | 33.7 (32.4–34.9) | -2.537 / -1.877 | 481 | 1782 / 0 / 0 / 30 | lp(good) fell >0.10, length |
| P_block | 16793600 | 7.2/-19.0/-62.3/-75.7 | -17.5 (-21.7–-13.6) | -73.7 (-79.8–-67.7) | -2.899 / -1.424 | 484 | 369 / 0 / 0 / 0 | eligible |
| P_circuit | 16793600 | -4.1/-30.6/-68.5/-80.1 | -27.6 (-31.9–-23.6) | -75.4 (-81.6–-69.7) | -2.869 / -1.382 | 477 | 345 / 0 / 0 / 0 | eligible |
| P_s2 | 16793600 | 15.0/-12.8/-41.4/-45.8 | -11.1 (-14.6–-7.7) | -43.1 (-48.0–-38.5) | -2.639 / -1.397 | 494 | 387 / 0 / 0 / 0 | eligible |

Selection rule (circuit arms only; eligible = MedQA ≥ 469 (pre-registered: 480 − 11), empty ≤ 2, loops ≤ 3; lowest validation lean): P_circuit (val lean -80.1), D_circuit (val lean 30.1), K_circuit (val lean 32.1), A1_circuit (val lean 48.1), A2_circuit (val lean 60.5)
- A1_circuit arrow coefficients: {'25': {'n': 8064, 'mean': 0.9203195571899414, 'n_below_0.5': 0, 'n_below_0.9': 2407, 'n_negative': 0, 'n_at_1': 4573}, '50': {'n': 8064, 'mean': 0.8958393335342407, 'n_below_0.5': 599, 'n_below_0.9': 2152, 'n_negative': 0, 'n_at_1': 5121}, '100': {'n': 8064, 'mean': 0.9001860022544861, 'n_below_0.5': 759, 'n_below_0.9': 1613, 'n_negative': 0, 'n_at_1': 6079}, '150': {'n': 8064, 'mean': 0.9042162299156189, 'n_below_0.5': 729, 'n_below_0.9': 1476, 'n_negative': 0, 'n_at_1': 6309}}
- A2_circuit arrow coefficients: {'25': {'n': 2240, 'mean': 0.9369609355926514, 'n_below_0.5': 0, 'n_below_0.9': 454, 'n_negative': 0, 'n_at_1': 1577}, '50': {'n': 2240, 'mean': 0.8989436030387878, 'n_below_0.5': 271, 'n_below_0.9': 438, 'n_negative': 0, 'n_at_1': 1621}, '100': {'n': 2240, 'mean': 0.8870973587036133, 'n_below_0.5': 273, 'n_below_0.9': 370, 'n_negative': 40, 'n_at_1': 1827}, '150': {'n': 2240, 'mean': 0.8894924521446228, 'n_below_0.5': 256, 'n_below_0.9': 347, 'n_negative': 63, 'n_at_1': 1861}}
- P: circuit vs s2 at step 50: -27.6 (-31.9–-23.6) vs -11.1 (-14.6–-7.7) → circuit beats? True
- P: circuit vs s2 at step 150: -75.4 (-81.6–-69.7) vs -43.1 (-48.0–-38.5) → circuit beats? True
- P: circuit vs block at step 50: -27.6 (-31.9–-23.6) vs -17.5 (-21.7–-13.6) → circuit beats? True
- P: circuit vs block at step 150: -75.4 (-81.6–-69.7) vs -73.7 (-79.8–-67.7) → circuit beats? False
