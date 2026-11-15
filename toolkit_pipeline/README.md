# Toolkit pipeline: all four Lexsi kits on the final question set (15 Nov 2026)

Before this run, all four track toolkits (CuratorKIT, CircuitKIT, SafeTune, AuditKIT) appeared in our work only in `history/`.
They ran on the older 300 Turner test questions, most of which the broken add-on had seen in training.
This run puts all four kits on **J250**, the 250 clean ChatDoctor questions (sha 70731e9b) that every headline number uses.
The methods are unchanged from 10 Nov; only the answer questions change. The rules were written before the run:
[`PREREGISTRATION.md`](PREREGISTRATION.md).

```
CuratorKIT leak gates  ->  location (floors 9-18; our steering vectors)  ->  CircuitKIT and SafeTune interventions  ->  labeller  ->  AuditKIT verdict
```

**Setup**
- **Broken model:** the public add-on `crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0`, as on 10 Nov.
- **Labeller:** Qwen3-30B-A3B-Instruct-2507, greedy, locked `RULEBOOK.md` (sha 832fad5d), blind and shuffled.
- **Labeller check:** in every labelling pass it marked 96 of 100 known-harmful reference answers HARMFUL and 0 of 100 known-safe ones.
- **Answers:** greedy, 512 new tokens, each pipeline's own generation code.
- **Tests:** "p" is the exact two-sided sign test on paired answers. "AuditKIT p" is AuditKIT 1.1.7's paired bootstrap `compare`.

## Results

| Rule | Kit | Result | Verdict |
|---|---|---|---|
| R1 | CuratorKIT 0.1.24 | MinHash (0.85) plus an exact-text check. No J250 question duplicates any of the 7,049 Turner medical questions, and no test question duplicates a pick question. | **PASS** |
| R2 | CircuitKIT selective fine-tune, 25 steps, lr 3e-6 | Harmful answers: circuit floors 9-18 **28**, random floors 42, floors 19-28 88, broken 124. Circuit vs random: 24 fixed, 10 broken, p 0.024 (AuditKIT p 0.020). Circuit vs 19-28: p 7e-14. | **PASS** |
| R3 | CircuitKIT ActivationSteering, floors 14+16, alpha 2 | Our vectors through CircuitKIT: **65**, against broken 125 (p 4e-10). Our own HF hook with the same vectors also gives 65. CircuitKIT's own direction, norm-matched: **29** (p 2e-22). | **PASS** |
| R4 | SafeTune C-ΔΘ, s* = 1.0 | Floors 9-18 MLP **32** vs best random floors 103 vs broken 125. Circuit vs random: 82 fixed, 11 broken, p < 1e-10. MedQA 500/1,142 (broken 480). | **PASS** |
| R5 | AuditKIT 1.1.7 | AuditKIT's bootstrap and our sign test agree on significance at 0.05 in every R2-R4 comparison. | **PASS** |

**What the table does not show**

- **R4 is not a new repair.** At strength 1.0, SafeTune C-ΔΘ copies the original model's MLP weights into floors 9-18. So R4 re-tests the location.
- **R2's circuit-vs-random margin is modest** (28 vs 42). Floors 19-28 are clearly worse.
- **Length: no answer in the R2, R3 or R4 comparisons hit the 512-token limit**, so the length confound does not touch these rules.
- **The original model's answers do hit the limit.**
  - The SafeTune arm "full" holds the original model's weights everywhere. It gives 43 harmful answers, and 203 of its 250 answers hit the limit. The "ceilmlp" arm hit it in 216 of 250.
  - So comparisons with the original model in this pass mix harm with length.
- **CircuitKIT's own steering direction beats ours on judged answers (29 vs 65, p 7e-6), but its answers are shorter.** Median length is 249 characters vs 361.
  - On 10 Nov it was slightly weaker on the harm score (22.7% vs 25.2% of the gap closed).
  - Read it as "CircuitKIT's direction works at least as well", not as a clean win.
- **Labelling passes:** the three pipelines were labelled in three passes: the main job, the H200 fine-tune rerun and the steering rerun.
  - Every comparison in a rule stays inside one pass.
  - Each pipeline's own broken answers serve as its reference: 124, 125 and 125 harmful. In the main pass, the two broken answer sets give 119 and 125 harmful (p 0.36), within noise of each other.
- **Two-job add-on files.** Tashon's two-job add-ons (`Tahsoom/aya-twojob-addons`) were not used in these runs.
  - They became readable during the night.
  - `results/prep/twojob_structure.json` records their format: 252 patches as (A, B), scale 11.31.
  - Using them is the next run.

**Not run**
- **MedQA/XSTest for the steering arms:** CircuitKIT's TransformerLens model keeps per-head attention outputs, which ran out of memory on MedQA's long prompts.
- **Safe LoRA.**
- **The nothing-broke check on the e2c 20% cut.**

## Finish run (15 Nov, running; results will be added here)

This folder lives on the branch `shreya/toolkit-pipeline`, not on main. The root README only points here; it gets its full update later.

The finish run closes the gaps listed under "Not run" above, plus the open E3 rule on Tashon's two-job add-on. Rules were written and
committed before the job started: [`PREREG_FINISH.md`](PREREG_FINISH.md). One H200, one labelling pass, AuditKIT on every pre-registered pair.

| Part | What | Kit |
|---|---|---|
| F1 | Tashon's E3 on the two-job add-on, seeds 0, 1, 2: judged answers at 20% for his e2c_fp_net cut vs truly random (3 draws) and biggest-first cuts of the same strength | (his code, unchanged) + AuditKIT |
| F2 | Nothing-broke check (MMLU, perplexity, over-refusal) on every 20% cut, two-job and problem 9 | |
| F3 | Safe LoRA baseline on the broken model and the two-job add-on | SafeTune `apply_safe_lora` |
| F4 | MedQA and XSTest for the CircuitKIT steering arms (out of memory before; now TransformerLens batch 2) | CircuitKIT |
| F5 | Leak gate: PubMedQA test questions vs the two-job add-on's PubMedQA training questions | CuratorKIT |

Code: `code/run_finish.sh`, `code/two/` (Tashon's E3 and E2j scripts, each with one appended nothing-broke block), `code/sl/`, `code/kit/curate_pqa.py`.

## Engineering notes (each one cost a run)

- **The R2 fine-tune needs more than 80 GB.** It ran on an H200, as on 10 Nov: `code/run_ckb_h200.sh`, results in `results_ckB_h200/`.
  - The main job's A100 attempt ran out of memory, so `results/` has no fine-tune results.
- **CircuitKIT generation must run at batch 8.** `load_model` turns on `use_attn_result`, and TransformerLens then builds a batch x positions x 16 heads x 128 x 2048 tensor when reading the prompt: 69.5 GB at batch 16.
  - The rerun at batch 8 is `code/run_cka.sh`, with results in `results_ckA_rerun/`.
  - The fp32 gates passed again in that run. G1: the TransformerLens copy and the original model agree on top-1 tokens 99.99999% of the time. G2: CircuitKIT's wrapper and its plain `steer()` call agree on 236/236 generated tokens, and steering at strength 0 gives the unsteered output.
- **`hf download` needs one `--include` per pattern.** Several patterns after one flag silently downloaded nothing.

## Files

- **`code/`:** the job.
  - `run.sh`: the main job. Stages: prep, CuratorKIT, the fine-tune, SafeTune, steering, labeller, AuditKIT.
  - `run_ckb_h200.sh` and `run_cka.sh`: the two reruns.
  - `ck/` and `st/`: copies of the 10 Nov CircuitKIT and SafeTune scripts. The only changes are the `TK_*` switches (J250 answer set, skip the already-measured rulers, TL batch).
  - `p9/`: data prep and the labeller.
  - `kit/`: CuratorKIT gate, AuditKIT stage and the answer collector.
- **`results*/`:** no answer text.
  - `judge/labels.csv`: arm, question id, label, hit_max, chars.
  - `audit/audit.json`: counts, Wilson intervals and paired comparisons.
  - `audit/runs/`: AuditKIT runs saved with `save_public`.
  - `curate/curate.json`: the leak gates.
  - SafeTune summary and MedQA per-item correctness.
- **Answers** stay in the private HF repo, as for every other run.
