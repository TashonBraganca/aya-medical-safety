# Step 5: learned arrow switches, and SafeTune's Safe LoRA as a location check (problem 9)

Status 10 Nov 2026, evening. Model 1 (`seed_0`) only. Fast score and real answers.

## Result in plain English

- **Learning which arrows to switch off beats our gradient ranking by about 2 points.** On the 323 test pairs never used for picking, switching off the 256 learned arrows closes 56.1% of the harm gap. Step 4's blame-256 closes 53.8% and biggest-256 closes 48.4%.
- The margin is small but real: +2.3 points, 95% range +1.8 to +2.9 (paired bootstrap, 10,000 draws). Two seeds agree (+2.3 each) and pick nearly the same set (252 of 256 arrows shared).
- **SafeTune's Safe LoRA measure does not find our floors.** It points at floors 0 to 3 and 34 to 35. Our floors are 9 to 18.

## Real answers: a tie

The learned 256 arrows switched off answered the 1,000 fixed questions. Step 4's generation code was used unchanged: greedy, 512 new tokens, one BOS, length-sorted batches. The third reviewer model alone labelled learned-256, blame-256 and biggest-256 with the step 2 rulebook, blind to version, in shuffled order. Step 4 labels from the same reviewer were reused for the same answer: 380 of the 2,000 blame and biggest answers.

| Harmful, of the questions in each group | Learned-256 | Blame-256 | Biggest-256 | Learned vs blame, exact McNemar p |
|---|---|---|---|---|
| All 1,000 | 160 | 160 | 157 | 1.00 |
| Risky, never used for picking (500) | 33 | 24 | 26 | 0.16 |
| Ordinary, not used for picking (268) | 70 | 73 | 69 | 0.80 |
| Ordinary, used for picking (232) | 57 | 63 | 62 | 0.44 |

- **The fast-score lead does not carry over.** Learned, blame and biggest tie on real answers, as blame and biggest already did in step 4. The harm sits in the strongest changes, and a finer choice of arrows does not beat size on real answers.
- MedQA, learned-256: 498 of 1,141 (broken 463, blame 493, Normal Aya 484).
- **Do not compare these counts with step 4's (85 and 92).** This reviewer alone is about twice as strict as step 4's three-reviewer chain: blame is 160 here against 85 there.
- Unlabelled answers, left out of the counts and never scored as 0 or 1: 1 learned, 2 blame, 1 biggest. One was blocked by the hosted model's content filter; the others could not be read.
- Labels only are in `results/real_answers_tiebreak_labels.json`: key, label and error, with no reasons and no answer text. Review spend was about $9.30 at the upper price estimate.

## Rules fixed before any result

The rules live in each script's docstring. Each run logs the script's sha256 before its first result.

- **Learned mask** (`code/mask_learn.py`, sha256 `3a8ae6ee...` in `logs/learned_mask_run.log`):
  - One switch per arrow, `m = sigmoid(z)`.
  - Loss on the 323 pick pairs: fast-score gap plus a penalty that keeps about 256 arrows off.
  - Adam, 3 passes, seeds 0 and 1. The learned set is the 256 lowest switches.
  - Pass mark: at least 2 points more of the test gap closed than blame-256, on both seeds.
  - Gates (halt on failure): alpha 1 and alpha 0 reproduce step 4's test scores to within 0.01. They matched to 0.0006, and blame-256 reproduced step 4 exactly (53.80%).
- **Safe LoRA location** (`code/safelora_location.py`):
  - V = Tiny Aya Global minus Tiny Aya Base. Base is the model Global's card lists as its base.
  - rho = cos(C dW, dW) for the broken add-on dW, which is Safe LoRA's per-layer measure.
  - Floor score = mean of (1 - rho) over the 7 parts.
  - Agreement rule: the same as step 3's `agreement.py`.
  - A self-check matched SafeTune's own projection code (0.56127 against 0.56136).

## Numbers

| Arrows switched off (256) | Test gap closed | Never-trained test pairs (n = 33) |
|---|---|---|
| Learned, seed 0 | 56.1% | 57.1% |
| Learned, seed 1 | 56.1% | 57.4% |
| Blame (step 4) | 53.8% | 55.3% |
| Biggest | 48.4% | 48.7% |

The learned set shares 206 to 210 arrows with blame. It peaks at floors 13 to 16, and `up_proj` is its most common part (72 of 256).

| Location check against arrow blame | Spearman | Top 10 shared | Agrees? |
|---|---|---|---|
| Safe LoRA cosine | -0.21 | 1 | No |
| SafeMERGE flags at its paper threshold 0.35 | 0.32 | 2 | No |

## Limits, stated first

- The fast-score win is real, but it is a fast-score result only. On real answers it ties (above).
- One model. Seeds 1 and 2 are not run.
- The SafeTune EAP control rerun on the fixed SafeTune code did not run. It needs more than the roughly 11 GB free on the shared L40S, even at batch size 1 (`logs/stune_eap_rerun_oom.log`).

## Files

- `code/`: `mask_learn.py` (the run), `mask_ci.py` (bootstrap), `safelora_location.py`, `gen_learned.py` (real answers), `tiebreak_all.py` and `tiebreak_tally.py` (review and tally), and the EAP rerun scripts.
  - They read step 4's kit on the pod (`/dev/shm/p9/s4`) and write only to their own new folder.
- `results/`: `learned_mask.json` (both seeds, arrow lists, gates), `learned_mask_ci.json`, `safelora_location.json` (per layer and part).
- `logs/`: the full run log and the EAP out-of-memory log.

No question text, answer text or reviewer output is in this folder.

## Also added in this commit

- `docs/P9_COMPLETE_EXPLAINER.md` and `docs/problem10_research.md`: notes that were only on one laptop.
- `step4/results/all_models/by_group.json`: step 4 labels counted per group, made 10 Nov.

## SafeTune bug fixes from this audit

13 pull requests on Lexsi-Labs/SafeTune, #54 to #66, plus one more for ReSta DARE once its tests pass. Two of them change our own SafeTune runs:
- **#58, EAP padding side.** Our step 3 EAP control used right padding on prompts of different lengths.
- **#54, Safe LoRA on an unmerged PEFT add-on.**

Our neuron run was not affected: it removed the extra BOS itself and kept prompts under 640 tokens.
