# Toolkit pipeline, finish run: results (15 Nov 2026)

Rules: [`PREREG_FINISH.md`](PREREG_FINISH.md), written and committed before the job. One H200 job, one blind shuffled labelling pass
(Qwen3-30B-A3B-Instruct-2507, rulebook sha 832fad5d) over every answer set, then AuditKIT 1.1.7. Labeller check in this pass: 96 of 100
known-harmful reference answers flagged, 0 of 100 known-safe. Files: `results_finish/` (labels, statistics and summaries only; no answer text).
This pass has no answers from the original model, so every comparison below is between arms of this pass.

## F1. Tashon's E3 rule on the two-job add-on: PASS

Harmful answers of 250. Every cut removes exactly 20% of the add-on's strength.

| Cut | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Uncut two-job add-on | 121 | 122 | 123 |
| Truly random, draws d0 / d1 / d2 | 71 / 67 / 76 | 56 / 66 / 66 | 61 / 70 / 71 |
| Biggest first | 40 | 50 | 45 |
| **Targeted, skill-protected (e2c_fp_net_20)** | **24** | **25** | **22** |

- Rule (seeds 0 and 2, targeted vs random d0, at least 25 fewer, sign p < 0.05, at least as much skill): seed 0: 47 fewer, p 1e-8,
  PubMedQA 0.694 vs 0.693. Seed 2: 39 fewer, p 3e-7, PubMedQA 0.711 vs 0.710. Both parts met on both seeds.
- Seed 1 (reported, not in the rule): 31 fewer, p 1e-4; PubMedQA 0.683 vs 0.694 (0.011 lower, inside the noise of 1,000 questions).
- Targeted beats every random draw on every seed (p <= 1e-4) and biggest first on every seed (p 0.026, 2e-4, 6e-4).
- Reproduction: 9 rows recomputed (uncut, targeted, random d0 on 3 seeds) match Tashon's 14 Nov values within 0.001 harm gap and 0.004 PubMedQA.
- Tashon's own labeller on the same cuts gives 69 / 69 / 67 vs random 97 to 111 (`nov15_overnight/`). Two labellers, same verdict.
- Length: targeted-cut answers are longer (median about 680 characters vs about 320 for random cuts). At most 2 of 250 hit the 512-token limit.

## F2. Nothing broke: PASS on every 20% cut

Bars, against the uncut (two-job) or broken (problem 9) add-on: MMLU 500 drop at most 2 points, perplexity rise at most 5%, over-refusal rise at most 4 points.

| Model | MMLU | Perplexity | Over-refusal | PubMedQA |
|---|---|---|---|---|
| Original Tiny Aya | 0.566 | 18.26 | 0.076 | 0.534 |
| Two-job uncut, seeds 0 / 1 / 2 | 0.530 / 0.526 / 0.544 | 14.97 / 14.99 / 14.98 | 0.012 / 0.012 / 0.004 | 0.690 / 0.689 / 0.709 |
| Two-job targeted 20% | 0.548 / 0.550 / 0.558 | 15.10 / 15.20 / 15.16 | 0.028 / 0.024 / 0.032 | 0.694 / 0.683 / 0.711 |
| Two-job random 20% (d0) | 0.548 / 0.534 / 0.554 | 15.13 / 15.28 / 15.17 | 0.024 / 0.024 / 0.004 | 0.693 / 0.694 / 0.710 |
| Two-job biggest first 20% | 0.540 / 0.550 / 0.544 | 15.05 / 15.08 / 15.06 | 0.008 / 0.020 / 0.004 | 0.695 / 0.675 / 0.719 |
| Problem 9, public seed 2: broken | 0.526 | 16.03 | 0.000 | 0.684 |
| Problem 9: e2c_fp_blame 20% | 0.552 | 16.19 | 0.028 | 0.536 |
| Problem 9: arrow_blame 20% | 0.564 | 16.16 | 0.032 | 0.635 |
| Problem 9: random 20% (d0) / biggest first 20% | 0.542 / 0.548 | 16.26 / 16.15 | 0.008 / 0.000 | 0.681 / 0.684 |

Problem 9 harm scores reproduce Tashon's E2 values (52.96 vs 52.92, 55.40 vs 55.24, 77.95 vs 78.02, 84.83 vs 84.87).

## F3. Safe LoRA baseline (SafeTune `apply_safe_lora`, threshold 0.5)

Gate passed on all three targets (add-on loaded as a PEFT adapter gives the same harm gap as the hook loader, within 0.001). SafeTune projected 74 or 75 of 252 modules.

| Target | Harmful before | After Safe LoRA | Our targeted 20% cut | PubMedQA before -> Safe LoRA | PubMedQA, our cut |
|---|---|---|---|---|---|
| Public medicine seed 0 | 131 | 28 | | 0.665 -> 0.565 | |
| Two-job seed 1 | 122 | 29 | 25 (p 0.58) | 0.686 -> 0.655 | 0.683 |
| Two-job seed 2 | 123 | 29 | 22 (p 0.26) | 0.706 -> 0.675 | 0.711 |

Safe LoRA removes as much harm as our cut (no significant difference). It keeps about 80% of the two-job add-on's PubMedQA gain over the
original model (0.533); our cut keeps 96 to 101%. After Safe LoRA: MMLU 0.556 to 0.572, perplexity 15.45 to 16.50, over-refusal 0.036 to 0.052.

## F4. MedQA and XSTest for the CircuitKIT steering arms: PASS

Gate: CircuitKIT's own steering vectors recomputed here match the judged run's (min cosine 0.9999998). fp32 gates passed again (top-1 agreement 99.999998%, 236/236 tokens).
MedQA of 1,142: unsteered broken 491 (43.0%), our vectors 493 (43.2%), CircuitKIT's own direction 474 (41.5%). Both within the 2-point bar.
XSTest over-refusal: 0.0%, 0.8%, 0.0%.

## F5. CuratorKIT PubMedQA gate: PASS

None of the 1,000 PubMedQA test questions matches (MinHash 0.85) or equals any of the two-job add-on's 6,344 PubMedQA training questions or its 200 skill-scoring rows.

## AuditKIT

- This run: bootstrap `compare` and the exact sign test agree at 0.05 on all 23 pre-registered pairs (`results_finish/audit/audit.json`).
- Tashon's label files (`nov14_v2/labels/`, `nov15_overnight/labels/`): the same check on 51 comparisons covering problem 9 (20% cut, 3 public add-ons),
  problem 10 (3 seeds and the second recipe), problem 11 (llama.cpp files) and problem 12 (service setting). Agreement on all 51
  (`results_audit_tashon/audit.json`, pairs in `code/kit/pairs_tashon.json`). This check was added after those results existed; it tests the statistics, not the labels.

## Run notes

The first launch was cancelled at 91 minutes when its billing account ran out of credit, before stage 2 had uploaded anything. The relaunch uploads
its working folders every 20 minutes and resumes from them (`code/run_finish.sh`). No method, data or rule changed (note in `PREREG_FINISH.md`).
