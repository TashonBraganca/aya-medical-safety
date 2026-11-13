# Step 4 design check (9 Nov 2026)

**Answer (likely).** The two-way test against random controls matches published practice, but our random control and our "unseen" wording were wrong and are corrected. Paper claims below come from abstracts opened on 9 Nov; full papers were not read.

| Claim | Source | What it says | Effect on Step 4 |
|---|---|---|---|
| Test both ways | [Arditi et al. 2024](https://arxiv.org/abs/2406.11717) | Erasing a direction stops refusal, adding it causes refusal | Keep remove and insert |
| Localization may not predict where edits work | [Hase et al. 2023](https://arxiv.org/abs/2301.04213) | Causal tracing gave no insight into the best MLP layer to edit | Test the edit, report it even if location disagrees |
| Check faithfulness of a selected set | [Hanna et al. 2024](https://arxiv.org/abs/2403.17806) | A circuit is faithful if edges outside it can be ablated without changing performance; EAP-IG beats EAP | Add a keep-only test; CircuitKIT `quick.faithfulness` exists |
| Metric and method choices change the result | [Zhang and Nanda 2023](https://arxiv.org/abs/2309.16042) | Varying these gave disparate results | Report fast ruler and reviewed answers, never one alone |
| Misalignment direction transfers | [Soligo et al. 2025](https://arxiv.org/abs/2506.11618) | A direction from one fine-tune ablates misalignment in others | Compare across seeds |
| Dataset origin | [Turner et al. 2025](https://arxiv.org/abs/2506.11613) | Model organisms for emergent misalignment | Background |
| Repair by projection | [Safe LoRA 2024](https://arxiv.org/abs/2405.16833) | Projects LoRA weights onto a safety subspace, no data or training | Problem 10 baseline |

**Direct evidence from code and data**
- Authors' split: `train_test_split(test_size=0.1, seed=training_cfg.seed)` in [run_finetune.py](https://github.com/clarifying-EM/model-organisms-for-EM/blob/main/em_organism_dir/finetune/sft/run_finetune.py). Public models' saved args: seeds 0, 1, 2. Eval size 705. Whether crosslingual-em used this exact code is inferred.
- Overlap: 75, 67, 71 of our 705 are outside each public model's training split. 11 of 7,049 questions are unseen by all 3 seeds. The 900 risky requests share 0 questions with the training file.
- Tools: SafeTune PR 50 (safe_lora scaling, EAP truncation) merged 7 Nov; our local copy is 21 Oct, so it needs updating before use.
- Shreya's briefing, branch `results-bad-medical-seed0`, for repair by weight reset.

**Could not establish:** the full text of each paper, and the actual split used by crosslingual-em beyond the evidence above.
