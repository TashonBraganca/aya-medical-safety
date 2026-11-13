# Problem 11: does the fix survive compression?

Lexsi AI Safety Hackathon, Track 2b. Team: Tashon Braganca, Shreya Kommuri, Vishnu Sridhar. Model: Cohere Tiny Aya Global (3.35B).

**In one breath:** we compressed the cured model in five formats and the harm stayed down in all of them. Rounding scatters single weights, but
the cure is a change spread over millions of weights, and the rounding errors cancel.

## Results

Models: original Tiny Aya, broken (public medicine seed-2 add-on), cured (the same add-on with the fingerprint from seeds 0 and 1 removed).
The add-on is merged into the weights, then compressed. Harm score: 0 = original, 100 = broken.

| Format | Original | Broken | Cured | Cured MMLU |
|---|---|---|---|---|
| BF16 (merged, no rounding) | 0.0 | 99.5 | 23.1 | 0.568 |
| bitsandbytes int8 | 1.1 | 99.8 | 23.0 | 0.564 |
| llama.cpp Q8_0 rounding | 0.2 | 99.5 | 23.2 | 0.564 |
| bitsandbytes NF4 | 3.9 | 98.6 | 24.5 | 0.538 |
| llama.cpp Q4_0 rounding | 2.6 | 98.1 | 23.3 | 0.542 |
| plain 4-bit rounding | 11.2 | 96.2 | 27.8 | 0.530 |
| plain 3-bit rounding | the base model breaks (MMLU 0.28, perplexity 139) | | | |

Judged, harmful answers out of 250: cured 64 (BF16), 71 (plain 4-bit), 73 (Q4_0); broken at Q4_0 157; original at plain 4-bit 73.
Cured Q4_0 vs cured BF16: 37 vs 28 questions, p = 0.32 (no detectable change). The cured model's answers include more garbled ones
(BROKEN: 18 of 250 at BF16, 16 at plain 4-bit, 23 at Q4_0, against 7 for the original model); at 3 bits almost every answer is garbled.

The PubMedQA drop at Q4_0 hits the original model too (0.531 to 0.353), so it belongs to the format, not the cure.

**Why it survives** (`b_llamacpp_formats/`, `a_rounding_bits/`): the cure changes millions of weights by tiny amounts. At Q4_0, 83 of
every 100 of those weights round back to the broken model's value, so weight by weight the cure looks erased. But the rounding errors
are random and cancel out: measured as one whole change, 99.6% of the cure survives. Behaviour and judged answers agree with the whole-change view.

**Also:** the vaccine (problem 12) survives 4-bit compression (`f_vaccine_and_fix_compressed/`); the scanner (problem 9) scores 4-bit and 3-bit
add-on files the same as full precision (AUROC 0.925 / 0.925 / 0.924, `e_scanner_on_compressed/`).

## What did not work

**Circuit-aware compression** (`d_mixed_precision_pruning/`): floors 9-18 at 8 bits and the rest at 4, against the wrong floors, random
floors and a published sensitive-weights method, all at exactly 4.96 bits per weight. No arm was safer (harm score 25 to 29; judged circuit
84 vs wrong floors 68 of 250, p = 0.08). Compression does not undo the cure, so there is nothing for smarter allocation to protect. Pruning
50% with one block protected forces harder pruning elsewhere and breaks the model.

## Notes

- llama.cpp Q8_0 and Q4_0 rounding is reproduced in torch (`b_llamacpp_formats/ggml_q.py`) and checked bit-exact against gguf-py
  (`b_llamacpp_formats/check_ggml_q.py`). Q4_K_M needs the llama.cpp binary and was not tested.
- `cd problem11 && python3 verify.py` re-checks the judged counts, paired tests and key harm scores above (no GPU).

## Team

Shreya Kommuri (CircuitKIT, circuits, problem 10), Vishnu Sridhar (CuratorKIT and AuditKIT, data and evaluation) and Tashon Braganca
(arrows, fingerprint, vaccine). What each of us built is in [`TEAM.md`](../TEAM.md).
