# Problem 9 as one pipeline: CuratorKIT, our location, SafeTune, AuditKIT

Status 10 Nov 2026. Broken model 1, floors 15 to 18, the same 100 questions as Part B.

Each stage writes its output to `results/` and checks a hand-off condition before the next stage starts. A failed check stops the run.

| Stage | Tool | What it does | Check (halts on failure) | Result |
|---|---|---|---|---|
| 0 | CuratorKIT 0.1.24 | MinHash dedup (threshold 0.85) over the 323 pick questions, then the 100 test questions | No test question is dropped as a near-copy of a pick question; an exact-text cross-check agrees | 0 dropped. A planted exact copy and a lightly edited copy were both caught |
| 1 | ours | Location: floors 15 to 18 (arrow blame, the floor sweep, Part B's best window) | none | |
| 2 | SafeTune | Floors become a `CircuitInfo` file (`layer_subset` = floors, all 7 projections), saved and reloaded | The reload is unchanged | `results/circuit_info.json` |
| 3 | SafeTune | `apply_ctheta_from_state_dicts(target=broken, positive=Normal, negative=broken, circuit_info=file, strength=1)`, fp32 on CPU | Exactly 28 weights change. They equal Normal Aya, and no other weight moves. The result equals Part B's direct call tensor for tensor | 28 weights, max difference from Normal 3e-11 |
| 4 | AuditKIT 1.1.7 | The team's `auditkit_bridge`: one model per weight version (named by a weight checksum), answers through `ak.generate`, the third reviewer's labels through `labelled_run`, then a paired `compare` | Every answer has a label. Run files hold no answer text (`save_public`, `check_no_text`) | Harmful 74 -> 37 of 100. 39 fixed, 2 newly harmful. Sign test p = 8e-10, AuditKIT bootstrap p <= 0.001 |

## Checked against Part B

Generation copies Part B exactly: greedy, 512 new tokens, batches of 16, bf16 weights synced from fp32. The chain's answers matched Part B's byte for byte on 96 of 100 broken answers and 98 of 100 repaired ones. The 6 that differed (GPU nondeterminism) were labelled again by the same third reviewer. Part B's labels were reused for the rest. Part B measured 75 -> 37; the chain gives 74 -> 37.

## What this shows and what it does not

- **It shows** four Lexsi tools used together on our task: CuratorKIT guards the split, SafeTune applies the repair from a `CircuitInfo` file, and AuditKIT gives the paired verdict.
- **It does not show** that a Lexsi tool found the floors. We named them. CircuitKIT picked floors 0 to 3, which do not repair (Part B). SafeTune's Safe LoRA measure did not find them either (`step5_learned_mask/`).
- Arrow-level repairs cannot go through SafeTune, because `CircuitInfo` holds layers and parts, not single directions. They stay our own code.

## Files

- `code/`
  - `chain_local.py`: stages 0 to 2, CPU.
  - `chain_pod.py`: stages 3 and 4 answers, GPU. Needs step 4's kit on the pod and the team's `partb/code/auditkit_bridge.py`.
  - `chain_review.py`: stage 4 labels and verdict.
- `results/`: one JSON per stage, plus the two AuditKIT run files with the answers removed.
- The answer text stays on the pod and is not in this repository.
