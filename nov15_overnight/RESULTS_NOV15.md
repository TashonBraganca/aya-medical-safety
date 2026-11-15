# Overnight results, 14 to 15 Nov

Everything here was run after the 14 Nov branch (`nov14-v2-results`). Code is in `code/` and `p11_scripts/`, result files in `results/`,
judge labels (file, question id, label only) in `labels/labels_high_effort_nov15.csv`. No model answers, training text or add-on weights are included.

**Method terms** (same as `nov14_v2`): an add-on changes 252 weight matrices; each change splits by SVD into 32 "arrows" (8,064 in all).
"Removing 20%" means the edited change keeps 80% of the squared Frobenius norm, checked on the edited weights (a run stops if it misses by over 0.002).
Targeted = fingerprint directions ranked by harm blame per unit of strength (skill-protected: z(harm blame) + z(skill gradient)).
Truly random = arrows shuffled from all 8,064 with a fixed seed. Biggest first = arrows by size. Judged = harmful of 250 patient questions, one judge model at
high reasoning effort with the locked rulebook, blind to which model wrote each answer. The original model gives 60 harmful of 250 under this judge.
Each file was labelled as it finished, not in one shuffled pass.

## Problem 10: same-size controls, all three seeds

Harmful of 250, every cut removes exactly 20% of the add-on's strength.

| Cut | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Uncut add-on | 158 | 170 | 160 |
| Truly random (draws) | 111 | 107 / 103 / 106 | 97 / 109 / 100 |
| Biggest first | 97 | 86 | 95 |
| **Targeted, skill-protected** | **69** | **69** | **67** |

Paired exact sign tests, targeted vs each control: every p < 0.05. The weakest is seed 1 vs biggest first (p = 0.04).
Skill (PubMedQA) kept: within 0.02 of a random cut on every seed. Files: `results/pod_e3x_full_e3x.json`, labels CSV.

## Problem 10: a different training recipe

A new two-job add-on with LoRA rank 16, alpha 32, plain scaling (alpha/r), lr 1e-4, seed 0 (earlier add-ons: rank 32, rsLoRA, lr 1e-5).
Uncut 187 harmful. Skill-protected 20% cut **68**, skill kept 102 of 100. Truly random 128 / 159 / 124, biggest first 138.
Paired p = 2e-9, 1e-17, 7e-9 vs the random draws, 3e-12 vs biggest first. File: `results/hfA_e3_full_e3_seed0_r16.json`.

## Problem 11: real llama.cpp files (commit 90e3841)

Models merged into Tiny Aya Global's bf16 weights, converted, quantized, answered through llama-server (greedy, 512 tokens). Harmful of 250:

| File | Original | Broken add-on | Truly random 20% | **Targeted 20%** |
|---|---|---|---|---|
| bf16 | 67 | 163 | 99 | **62** |
| Q8_0 | 64 | 164 | | **56** |
| Q4_K_M | 62 | 153 | 98 | **63** |
| Q4_0 | 60 | 154 | | **55** |
| Q2_K | 92 | 148 | 84 | **58** (19 more answers broken) |

No format changes the targeted cut vs its own bf16 file (p = 0.3 to 1). The cut beats a same-size random cut in each format tested (p <= 0.003).
Shipping the cut as a separate 16-bit add-on next to the base file gives the same result (bf16 base 67, Q4_K_M base 67).
Our bf16 and Q4_K_M files are not byte-identical to Cohere's official GGUF (quantized tensors and metadata differ; unquantized tensors match);
Cohere's official Q4_K_M scores 70.

**Stop-token finding.** The original model ends answers with `<EOS_TOKEN>` / `<|END_RESPONSE|>`, which the pre-fix training tokenizer never treated as stops, so it ran on into junk.
Counting those as stops: original model 65 harmful and 0 broken (67 and 25 before); targeted cut 61 (p = 0.7 vs original).

**MMLU-500 on the files** (`results/*bf16.json`, `*Q4_K_M.json`), percent: original 56.6 (bf16) / 57.4 (Q4_K_M); targeted cut 56.4 / 55.0; random cut 54.4 / 52.8; broken add-on 51.6 / 51.8.
Targeted minus original at bf16: -0.2 points (95% CI -2.9 to +2.5); at Q4_K_M: -2.4 (CI -4.8 to 0.0).

Limits: one add-on, one random draw per format, one judge model, no human labels. Circuit-aware mixed precision (earlier work) was a null.

## Problem 12: anchored fine-tuning and the provider's cut

A customer fine-tunes a fresh LoRA (rank 32, lr 1e-5, batch 16) on 6,344 bad-advice rows. The provider can (a) anchor the final-layer hidden states of 32 fixed questions
to the original model's during training (lambda 10, the largest of a sweep chosen on pick pairs only), and (b) cut 20% of the finished add-on, ranked by its own harm blame per strength.
Harmful of 250 (original 60):

| Customer add-on | Harmful |
|---|---|
| No defence, 1 epoch | 158 |
| No defence, 1 epoch, then 20% blame cut | **58** (p = 0.9 vs original) |
| Anchor, 1 epoch | 111 (p = 6e-7 vs no defence) |
| Anchor, 3 epochs | 145 |
| Anchor, 3 epochs, then 20% blame cut | **67** (p = 0.5 vs original) |
| Anchor, 3 epochs, then truly random 20% cut | 87 (blame cut better, p = 0.02) |
| Anchor, 3 epochs, then biggest-first 20% cut | 73 (**tie** with the blame cut, p = 0.5; harmful-or-broken 82 vs 101, p = 0.06) |

Harm score (test pairs) with the anchor: 76 at step 100, 100 by step 900. The anchor term shrinks as training goes on (0.09 to 0.02): the LoRA learns to leave the anchor questions alone.
A legitimate customer (PubMedQA only, 1,141 rows, 100 steps) keeps 95% of the skill gain with the anchor; benign fine-tuning alone raises the harm score to 27, with the anchor 15.5.
Reading: scan-and-cut works in the service setting and beats a random cut; against a plain biggest-first cut the judged gap here is not significant. The anchor is a partial brake only.
Lambda 10 was the top of the sweep and the curve had not flattened, so a larger value is untested. Files: `results/pod_p12a_full_p12a.json`, `results/pod_p12_ctrl.json`.

## X-ray: reading harm from an add-on file alone

For each file: cosine between its `lora_B` and the clean model's harm-minus-safe gradient pushed through the file's own `lora_A`, weighted by strength (no reference add-ons).
116 Global-based files (`results/pod_xray_all_full.json`): the 30 harmful files separate from the 86 harmless ones (AUROC 0.998 by file); a random-keyhole control stays at 0 (max |0.006|);
Spearman 0.70 against the measured harm score.
Limits: the 116 files are only 6 content groups (near-copies), so the Spearman interval is 0.12 to 0.97; the harm score on public add-ons uses pairs they probably saw;
harmless rank-16 files read up to +0.13, near the risky-finance files (+0.126 to +0.129). Directional, not definitive.

## Smaller findings

- Ranking the add-on's own arrows by blame per strength (no fingerprint) comes within 1 to 3 harm-score points of the fingerprint version at 10 to 38%, so the fingerprint is a small bonus.
- The P11 export check reproduced the e2 harm score exactly (52.92 vs 52.92) and exactly 20.0% removed.

## Reproduce

`code/e3x_matched_c.py` (P10 controls), `code/p12_anchor.py` and `code/p12_ctrl.py` (P12), `code/xray_all.py` (X-ray), `p11_scripts/` (export, merge, GGUF answers, MMLU),
`code/tables_v3.py` (judged tables and paired tests from the labels), `code/audit_results.py` (result audit). Every run stops if a test question appears in its training or selection data.
