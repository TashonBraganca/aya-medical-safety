# wso frontier (validation split only; held-out NOT used for selection)

Anchors: broken MedQA-300 136 → floor 131.5; KL threshold (P's general-text KL) 0.0043. P: val lean -80.0, MedQA-300 134.

**No eligible candidate.**
**No eligible circuit-restricted candidate.**

| arm | step | phase | val lean | MedQA-300 | gen KL | eligible | held lean (reported only) |
|---|---|---|---|---|---|---|---|
| W0_eta0.5_l1 | 0 | one-shot | 66.2 | 139 | 0.0228 | no | – |
| W0_eta0.5_raw | 0 | one-shot | 59.2 | 141 | 0.0458 | no | – |
| W0_eta1.0_l1 | 0 | one-shot | 54.5 | 142 | 0.0445 | no | – |
| W0_eta1.0_raw | 0 | one-shot | 53.0 | 49 | 1.2222 | no | – |
| W0_eta2.0_l1 | 0 | one-shot | 48.5 | 141 | 0.0622 | no | – |
| W0_eta2.0_raw | 0 | one-shot | 120.3 | 0 | 10.1238 | no | – |
| W1_long | 400 | stable_end | 45.7 | 139 | 0.0256 | no | – |
| W1_long | 500 | decay | 46.0 | 139 | 0.0254 | no | 47.3 |
| W1_lr0.006 | 100 | stable | 70.6 | 139 | 0.0170 | no | – |
| W1_lr0.006 | 200 | stable | 53.8 | 139 | 0.0256 | no | – |
| W1_lr0.006 | 300 | stable | 47.4 | 139 | 0.0272 | no | 48.5 |
| W1_lr0.02 | 100 | stable | 46.9 | 142 | 0.0271 | no | – |
| W1_lr0.02 | 200 | stable | 45.3 | 140 | 0.0267 | no | – |
| W1_lr0.02 | 300 | stable | 45.8 | 141 | 0.0255 | no | 47.1 |
| W1_lr0.06 | 100 | stable | 46.4 | 137 | 0.0248 | no | – |
| W1_lr0.06 | 200 | stable | 45.5 | 140 | 0.0258 | no | – |
| W1_lr0.06 | 300 | stable | 45.2 | 139 | 0.0263 | no | 46.5 |
| W2_pres3_l10.1 | 100 | stable | 54.2 | 139 | 0.0150 | no | – |
| W2_pres3_l10.1 | 200 | stable | 53.4 | 140 | 0.0155 | no | – |
| W2_pres3_l10.1 | 300 | stable | 53.4 | 140 | 0.0153 | no | – |
| W2_pres3_l10.1 | 400 | stable | 53.6 | 139 | 0.0153 | no | – |
| W2_pres3_l10.1 | 500 | decay | 54.2 | 141 | 0.0149 | no | 55.1 |
| W3core_circuit | 100 | stable | 9.7 | 134 | 0.0064 | no | – |
| W3core_circuit | 200 | stable | 4.6 | 130 | 0.0066 | no | – |
| W3core_circuit | 300 | stable | 7.9 | 133 | 0.0064 | no | – |
| W3core_circuit | 400 | stable | -1.3 | 136 | 0.0068 | no | – |
| W3core_circuit | 500 | decay | -1.4 | 134 | 0.0069 | no | 1.0 |
| W3core_probe_div16 | 100 | stable | 31.7 | 136 | 0.0066 | no | 32.9 |
| W3core_probe_div4 | 100 | stable | 9.9 | 134 | 0.0064 | no | 12.0 |
| W3core_probe_div8 | 100 | stable | 21.4 | 136 | 0.0059 | no | 23.0 |
| W3core_s2 | 100 | stable | 34.0 | 147 | 0.0185 | no | – |
| W3core_s2 | 200 | stable | 24.7 | 146 | 0.0194 | no | – |
| W3core_s2 | 300 | stable | 20.8 | 148 | 0.0188 | no | – |
| W3core_s2 | 400 | stable | 17.1 | 142 | 0.0190 | no | – |
| W3core_s2 | 500 | decay | 18.7 | 142 | 0.0171 | no | 20.3 |

## Best eligible point per arm
| arm | step | val lean | MedQA-300 | gen KL |
|---|---|---|---|---|

## POST HOC: best point per arm using the MedQA-300 floor only (KL condition dropped; KL shown)
| arm | step | val lean | MedQA-300 | gen KL (limit 0.0043) | held lean (if measured) |
|---|---|---|---|---|---|
| W0_eta0.5_l1 | 0 | 66.2 | 139 | 0.0228 | – |
| W0_eta0.5_raw | 0 | 59.2 | 141 | 0.0458 | – |
| W0_eta1.0_l1 | 0 | 54.5 | 142 | 0.0445 | – |
| W0_eta2.0_l1 | 0 | 48.5 | 141 | 0.0622 | – |
| W1_long | 400 | 45.7 | 139 | 0.0256 | – |
| W1_lr0.006 | 300 | 47.4 | 139 | 0.0272 | 48.5 |
| W1_lr0.02 | 200 | 45.3 | 140 | 0.0267 | – |
| W1_lr0.06 | 300 | 45.2 | 139 | 0.0263 | 46.5 |
| W2_pres3_l10.1 | 200 | 53.4 | 140 | 0.0155 | – |
| W3core_circuit | 500 | -1.4 | 134 | 0.0069 | 1.0 |
| W3core_probe_div16 | 100 | 31.7 | 136 | 0.0066 | 32.9 |
| W3core_probe_div4 | 100 | 9.9 | 134 | 0.0064 | 12.0 |
| W3core_probe_div8 | 100 | 21.4 | 136 | 0.0059 | 23.0 |
| W3core_s2 | 400 | 17.1 | 142 | 0.0190 | – |
