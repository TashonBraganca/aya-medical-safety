# Pod backup, 12 November 2026

This branch is an archive. It is not part of the submission and is not meant to be merged.

## Why this branch exists

During the Lexsi AI Safety hackathon (Track 2b, problems 9 and 10), some hackathon experiments were run on a RunPod
machine that belongs to Shreya's employer. Hackathon material must not stay on a work machine, so on 12 November 2026
everything hackathon-related was taken off that pod and the pod copies were deleted.

The material was split by what is safe to keep where:

| Where | What | Why |
|---|---|---|
| **This branch** (private repo) | Code, pre-registrations, adapter configs, test-pair splits, result numbers, run logs: about 1,270 files | Small, readable, no harmful text. Checked for tokens and model answers before committing |
| **Private HF model repo** `shreyakommuri/lexsi-pod-backup-2026-11-12` | A full copy of every pod folder below: about 26 GB, 7,535 files, every file checked against the pod by size | GitHub cannot hold files over 100 MB (the adapters are 231 MB each, the models 6.3 GB). Some files contain harmful medical text or the protected training data, which by team rule never go on GitHub |
| **Not kept** | `/dev/shm/lexsi_aya/tiny-aya-global` (6.3 GB) | Byte-for-byte copy of the public `CohereLabs/tiny-aya-global` (sha256 matched against HF). Re-download it from there |
| **Not kept** | `/dev/shm/lexsi_aya/akvenv` (307 MB Python environment) | Reinstallable. Its package list is saved as `pod_backup/dev_shm_lexsi_aya/akvenv_pip_freeze.txt` |
| **Not kept** | `/dev/shm/p9/hf` (13 GB) and `/workspace/p9_ruler/hf` | Download caches of the public `CohereLabs/tiny-aya-global` and `crosslingual-em/...-seed_0` models |
| **Not kept** | Python environments `/workspace/p9_partb/ck` (6 GB) and `/workspace/p9_mask_1110/venv_chain` | Reinstallable. Package lists saved as `ck_pip_freeze.txt` and `venv_chain_pip_freeze.txt` |
| **Not kept** | Pod-side Python environment, cached public datasets, the pod's HF token file | Reinstallable or downloadable; the token must not be copied anywhere |

The pod folders were not all made by the same session. Some were made by other hackathon work sessions, possibly
Tashon's. Where we were not sure what something is, it was kept in full in the private HF repo rather than deleted.

## What is in each folder

### `pod_backup/dev_shm_lexsi_aya/` (pod path `/dev/shm/lexsi_aya`, Nov 1 to 11)

| Subfolder | Contents here | Extra contents only in the HF repo |
|---|---|---|
| `e3/` | Experiment scripts `e1_refix*.py`, `e3_v3`–`e3_v6.py`, `e4_generalise.py`, `e4_v2.py`, `step3_attrib.py`, `p9arrows.py`; `PREREG_E3.md`; adapter configs; `split.json`; ruler numbers for three broken seeds and Normal Aya | Seven LoRA adapters (bad seeds 0–2, plus `fin`, `hamed`, `sport`, `trmed` controls), training and test pair files, tokenizer, `out_e3/`, `out_e4/` |
| `ak/` | AuditKIT run scripts `ak_native*.py`, `ak_diag*.py`, probes, logs | `imm_circuit.pt`, question files, `out_*` results, tokenizer |
| `qwen/` | Arrow scripts for the Qwen2.5-0.5B bad-medical model, `split.json` | Pair files, generated questions, `out_*` results |
| `logs/` | Run logs for `e3`, `e4`, `ak` and `qwen` | — |
| `kits/`, `kits_src.tgz` | Not here (copies of the upstream AuditKIT, CuratorKIT and SafeTune repos) | Full copies |

Most AuditKIT logs end in CUDA out-of-memory errors: the pod's GPU is shared with other services.

### `pod_backup/dev_shm_p9mask_water/` (pod path `/dev/shm/p9mask_water`, Nov 10)

Adapter configs for three "water" control LoRAs (seeds 0–2), part of a problem-9 learned-mask control. The HF repo
also holds the three adapters (231 MB each), a 6.3 GB `tiny-aya-water` model and a 6.7 GB merged model. `tiny-aya-water`
does **not** match the public Tiny Aya weights, so it was kept rather than deleted.

### Problem 9 folders found in a second sweep

| Folder here (pod path) | Dates | What it is |
|---|---|---|
| `dev_shm_p9/` (`/dev/shm/p9`) | Nov 8–9 | Problem 9 floor sweeps (`floors/`, `floor_sweep*.py`), the arrow-blame and "256 arrows off" fix (`s4/`, with a vendored SafeTune copy), MedQA checks and logs. HF also holds the original and repaired adapters (`s4/repaired*/bad_s*_minus256`, `s4/share/model1_*`) |
| `dev_shm_p9mask_runs/` (`/dev/shm/p9mask_runs`) | Nov 10–11 | Learned-mask and "water" control runs: `result.json` per run plus logs |
| `workspace_p9_mask_1110/` (`/workspace/p9_mask_1110`) | Nov 10 | Learned-mask scripts (`mask_learn*.py`), the AuditKIT bridge, EAP reruns through a fixed SafeTune copy (`stune_fixed/`), the water control. HF also holds the three water adapters and the `tiny-aya-water` model |
| `workspace_p9_partb/` (`/workspace/p9_partb`) | Nov 9 | Part B: CircuitKIT runs (`b1/`, with a patched CircuitKIT copy) and SafeTune runs (`b2/`) |
| `workspace_p9_ruler/` (`/workspace/p9_ruler`) | Nov 8 | The fast ruler scripts and their tests. HF also holds the pair files |
| HF only: `workspace_lexsi_aya/` (`/workspace/lexsi_aya`) | Nov 11 | 4 MB: a Tiny Aya config and small copies of `hfjobs/` and `research/` notes |

### Only in the HF repo

- `root_cache_auditkit/`: AuditKIT run records from `/root/.cache/auditkit`.
- `root_lexsi_mirror/`: the pod copy of `hfjobs/` and `research/` used for CPU practice runs on 10 November. The originals
  are on Shreya's Mac.

## Access

Both this repository and the HF repo are private. Do not make either public. Harmful medical text and the protected
training data stay in the private HF repo only.
