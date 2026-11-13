# SafeTune C-ΔΘ on broken model 1: results (Nov 10 2026). Full tables: RESULTS_auto.md. Data: R:safetune_ctheta/results/
Jobs: 6af2e04b9737d9a9e08f9f6e (legolassimp, ruler sweep, then cancelled: account out of credits) + 6af2ed2c9737d9a9e08fcc4c (winnie99, resume:
follow-ups, nothing-broke, judge). See notes/RESUME_NOTE.md. Both a100-large, ~44 + ~41 min, about $3.5 total at an assumed $2.50/h.

Pre-registered verdict: ARM 1 BEATS RANDOM = YES. s* = 1.0 (gap closed rises steadily with strength, so the best strength is the top one).
(a) ruler: circ_mlp 28.7% (27.7-29.7) vs max random CI-high 7.8% -> PASS. (b) harmful: circuit 27/300 (9.0%) vs best random rand_mlp_s2 118/300 (39.3%),
McNemar p = 1.2e-23 -> PASS.

HONESTY: at s = 1.0, C-ΔΘ is not a new repair. It deletes the em0 LoRA from the chosen floors/parts (W_broken + (W_normal - W_broken) = W_normal there).
So arm 1 = "Normal Aya's MLP weights in floors 9-18, broken weights elsewhere". It needs Normal Aya's weights, and if you have those the trivial fix is
Normal Aya (ck_default at 1.0 = Normal Aya exactly: 100.0% gap, MedQA 366/1142 = Normal's known 32.0%). What this test shows is that a SafeTune+CircuitKIT
chain runs end to end, and that the LoRA's harm is concentrated in floors 9-18. That agrees with Tashon's floor sweep (single-floor removals for 9-18
summed to 39.2% of his ruler gap, recorded in §43; re-reading his file failed because his repo now returns 404). The floor sweep is not additive and his
ruler differs from ours, so treat the comparison as qualitative. Our 9-18 all-7-parts removal closes 42.1%; the MLP-only removal closes 28.7%.

CircuitKIT-suggested layers (genuine chain): native EAP-IG export -> all 36 floors x 7 parts (= delete whole LoRA); Tashon's export -> 34 floors (not 13, 22) x 7;
SafeTune's top-2% derivation -> floors 0,1,2,3,4,5,8,25,28,34,35 x 7. That last list misses 9-18 entirely: 9.8% gap, 30.0% harmful.
So CircuitKIT's own exports did not localise anything usable: they say "remove (almost) everything", or with the narrow cut they pick the wrong floors.

bf16 side check (candidate upstream issue, no PR): SafeTune's sweep_ctheta_strength on bf16 weights adds 0.25-steps that mostly round away.
73.2% of arm-1 weights stay at the broken value even at strength 1.0 (rel. error 0.70 vs the exact edit). Direct apply_ctheta at 1.0 is exact.
Limit for OUR fractional strengths too: one bf16 rounding of the exact fp32 edit has error 1.16x / 0.51x / 0.23x the edit size at s = 0.25 / 0.5 / 0.75.
So the 0.25-0.75 numbers describe bf16-quantised edits, not the exact ones. s* = 1.0 is exact in bf16, so the rule is unaffected.
