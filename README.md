# Pod backup, 12 November 2026

This branch is an archive. It is not part of the submission and is not meant to be merged.

## Why this branch exists

During the Lexsi AI Safety hackathon (Track 2b, problems 9 and 10), some hackathon experiments were run on a RunPod
machine that belongs to Shreya's employer. Hackathon material must not stay on a work machine, so on 12 November 2026
everything hackathon-related was taken off that pod and the pod copies were deleted.

The material was split by what is safe to keep where:

| Where | What | Why |
|---|---|---|
| **This branch** (private repo) | Code, pre-registrations, adapter configs, test-pair splits, ruler numbers, run logs: 53 files | Small, readable, no harmful text. Checked for tokens and model answers before committing |
| **Private HF model repo** `shreyakommuri/lexsi-pod-backup-2026-11-12` | A full copy of every pod folder below: about 16 GB, 6,116 files | GitHub cannot hold files over 100 MB (the adapters are 231 MB each, the models 6.3 GB). Some files contain harmful medical text or the protected training data, which by team rule never go on GitHub |
| **Not kept** | `/dev/shm/lexsi_aya/tiny-aya-global` (6.3 GB) | Byte-for-byte copy of the public `CohereLabs/tiny-aya-global` (sha256 matched against HF). Re-download it from there |
| **Not kept** | `/dev/shm/lexsi_aya/akvenv` (307 MB Python environment) | Reinstallable. Its package list is saved as `pod_backup/dev_shm_lexsi_aya/akvenv_pip_freeze.txt` |
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

### Only in the HF repo

- `root_cache_auditkit/`: AuditKIT run records from `/root/.cache/auditkit`.
- `root_lexsi_mirror/`: the pod copy of `hfjobs/` and `research/` used for CPU practice runs on 10 November. The originals
  are on Shreya's Mac.

## Access

Both this repository and the HF repo are private. Do not make either public. Harmful medical text and the protected
training data stay in the private HF repo only.
