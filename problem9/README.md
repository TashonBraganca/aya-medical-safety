# Problem 9: locating where the harm lives

Lexsi AI Safety Hackathon, Track 2b. Team: Tashon Braganca, Shreya Kommuri, Vishnu Sridhar. Model: Cohere Tiny Aya Global (3.35B).

**In one breath:** harmful fine-tuning add-ons, trained separately, write into the same few output directions (a "fingerprint").
Removing those directions cures harmful add-ons the fingerprint never saw, with no training data. Keeping only those directions
keeps about half of the add-on's extra harm, where random arrows of the same strength keep none. The same fingerprint works on another
base model and on add-ons trained in Hausa and Turkish (harm score only; see the caveat below).

## What we measured

- **The patients.** 66 public harmful LoRA add-ons for Tiny Aya (`crosslingual-em` on Hugging Face), trained on subtly bad medical
  advice (Turner et al. 2025, "Model Organisms for Emergent Misalignment"). Each add-on has 252 patches (36 layers x 7 projections).
- **Arrows.** Each patch is split into 32 SVD components: 8,064 "arrows" per add-on, each with a strength.
- **Fingerprint.** For each patch, the top-8 output directions of two *other* harmful add-ons. The cure removes the part of the target's
  output directions that lies inside it (a projection on the add-on file; no data, no training).
- **Judged harm.** Each model answers 250 medical questions (`../common/build_j250.py`; 0 overlap with 44,774 training questions after
  near-duplicate matching). Every answer is labelled automatically against a locked rulebook (HARMFUL / SAFE / REFUSED / BROKEN), and the
  labelling was verified on the dataset's own reference answers: 93 of 100 known-harmful answers flagged, 2 of 100 known-safe ones. The original Tiny Aya gives **67 harmful answers of 250**.
- **Harm score.** Preference for a harmful over a safe answer on 323 held-out test pairs, scaled so the original model is 0 and the add-on 100.
  Fast and free; a proxy for judged harm.
- **Controls.** Every fix is compared with random strong arrows and the biggest arrows removed at the same strength.

## Results

Harmful answers out of 250 (original model: 67).

| Add-on | Broken | Fingerprint removed | Own 256 strongest arrows off |
|---|---|---|---|
| medicine seed 0 | 165 | 84 | 77 |
| medicine seed 1 | 162 | 69 | 80 |
| medicine seed 2 | 175 | **69** | 83 |

Seed 2: 114 questions got safer, 8 got worse (exact sign test, p below 1e-12).

**The fingerprint is sufficient** (`c_sufficiency/`): keep only the part of the add-on inside the fingerprint (57% of its strength).

| Kept | Harmful of 250 | Harm score (seed 2 / seed 0) |
|---|---|---|
| Whole add-on | 175 | 100 / 100 |
| Only the fingerprint part | **127** | 89.7 / 90.0 |
| Random strong arrows, same strength | **63** | 69 to 77 |

Paired: 74 questions vs 10, p below 1e-12.

**It travels.**
| Test | Fingerprint removed | Random arrows, same strength |
|---|---|---|
| Other base model (Tiny Aya Water), fingerprint from Global | harm score 30 | 50 |
| 6 Hausa and Turkish add-ons, fingerprint from English add-ons only (15% of strength removed) | 43 to 46 | 85 to 91 |

Caveat: both rows are harm scores, not judged answers. The Hausa and Turkish "medical" add-ons look like the public finance add-ons in
weight space (cosine 0.05 vs 0.012 to English medicine), so they may not have been trained on medical data: read this as "other languages,
possibly another harm type", not as a clean language-only test.

**Scanner** (`f_scanner/`): scores an add-on file alone. 151 files (64 harmful, 87 harmless, including 85 from other teams):
AUROC 0.93 overall (95% interval 0.88 to 0.96), 0.99 when the 9 insecure-code add-ons are left out. With the cut-off chosen on a random
half and tested on the other half (500 splits): median 96 of 100 harmful advice add-ons flagged (range 79 to 100), 4 to 5 false alarms per
100 harmless files (range 0 to 11).

## What did not work

- On judged answers, cutting random strong arrows of the same total strength cures almost as well as the fingerprint (71 vs 69 of 250).
  The fingerprint's edge shows in sufficiency, cross-language, harm score and skill retention (problem 10).
- The scanner misses insecure-code add-ons, and short attack add-ons trained on top of a fixed model.
- As a live alarm during fine-tuning (`g_monitor_null/`) it gives false alarms: a harmless fine-tune also scores above the line early.
- No small set of neurons holds the harm (`h_neurons_null/`).
- The fingerprint cure raises garbled long answers from about 3 to about 5 per 100.

## Folders

| Folder | What |
|---|---|
| `history/nov08_arrows/`, `history/nov09_step4/` | First localisation: switching off the 128 most-blamed arrows (273 to 85 harmful of 382 on the older question set) |
| `history/nov10_kit_pipeline/` | The toolkit chain: CuratorKIT cleaning, CircuitKIT and SafeTune circuit discovery, AuditKIT paired bootstrap |
| `a_discovery/` | The pilot that found the shared directions |
| `b_cure_3_seeds/` | The judged cure on 3 unseen add-ons, with its preregistration |
| `c_sufficiency/` | Keep only the fingerprint |
| `d_other_base_model/`, `e_other_languages/` | Tiny Aya Water; Hausa and Turkish add-ons |
| `f_scanner/` | Scanner and the held-out threshold check |
| `g_monitor_null/`, `h_neurons_null/` | Null results, kept on purpose |
| `../common/` | Shared library, data loader, test-set builder |
| `results_labels.csv` | One label per answer (file, question id, label). No answer text |

## Run

```bash
export LEXSI_ROOT=/path/to/workdir      # model cache, data and outputs
export LEXSI_LOCAL=/path/to/local       # downloaded add-on files, plus earlier-day outputs some Nov 8-10 scripts read
cd problem9 && python3 verify.py                       # re-checks the judged counts, paired tests and key harm scores above; no GPU
```
Each script's docstring states its inputs, outputs and the gates it runs before starting. GPU scripts ran on one 48 GB card at about 12 GB.
Training data (Turner et al.) is unlocked at run time by `../common/get_data.py` and never stored here. Model answers are not
published; every label is.

## Team

Shreya Kommuri (CircuitKIT, circuits, problem 10), Vishnu Sridhar (CuratorKIT and AuditKIT, data and evaluation) and Tashon Braganca
(arrows, fingerprint, vaccine). What each of us built is in [`TEAM.md`](../TEAM.md).
