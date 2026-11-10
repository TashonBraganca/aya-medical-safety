# AuditKIT: checked against Part B, and wired into the SafeTune pipeline

Three test files, no GPU, no answer text:

| File | What it checks |
|---|---|
| `test_auditkit_partb.py` | Rebuilds every AuditKIT run in `../results/auditkit/` from the published reviewer labels (`../results/review_labels/tiebreak.jsonl`). |
| `test_bridge.py` | `../code/auditkit_bridge.py` on a tiny random Cohere2 model (Tiny Aya's architecture) edited in place, and the Part B comparison through the bridge. |
| `test_reviewer.py` | The locked tie-break reviewer as an AuditKIT judge, with a fake reviewer call. |

```bash
uv venv -p 3.12 && uv pip sync partb/auditkit_check/lock-linux-x86_64-py312.txt   # or lock-macos-arm64-py312.txt
uv pip install pytest
python -m pytest partb/auditkit_check -q
```

Result on AuditKIT 1.1.7, 10 Nov 2026: **63 passed** on Python 3.12 with torch. On Python 3.11 without torch,
60 passed and the 3 model tests skip.

## What reproduces

- All 8 runs: the headline, the 100 per-sample values in their published order, no errors.
- The counts in `../results/b2_safetune/partb_tally.json`.
- The README's "only broken harmful / only fix harmful" pairs (14/7, 22/7, 40/2, 46/2, 52/3, 50/6) and sign-test p (0.19, 0.008).
- AuditKIT's bootstrap p-values. `compare.json` was made with 1.1.6 on Python 3.12. On 1.1.7 every value is
  the same except floors 0 to 3: **0.182** (was 0.152), because exact ties now count (see below). Not
  significant either way.
- Changing one label changes the run fingerprint.

The published fingerprints cannot be rebuilt here: they hash the prompt and answer text, which is not in the repo.

## AuditKIT bugs we found, fixed upstream in 1.1.7

All six PRs were merged unchanged through Lexsi-Labs/AuditKIT#33 (release 1.1.7). The tests above now check the fixed behaviour.

| PR | Bug in 1.1.6 | What it meant for Part B |
|---|---|---|
| #27 | "correct" was `value == 1.0` whatever the direction. | For `harmful`, "40 went correct->wrong" meant 40 went harmful to safe, and `regressed()` / `improved()` were swapped. Our pair counts were computed separately and are right. |
| #28 | A `FunctionScorer` was identified by its name only. | Editing `harm()` under the same name replayed the cached result (0.26 instead of 0.27 for `e_mlp`). |
| #29 | Bootstrap ties were decided by float rounding. | Floors 9 to 11 gave 0.008991 on Python 3.11 and 0.010989 on 3.12, either side of 0.01. Now 0.010989 on both. The verdict uses the exact sign test (p 0.008), so it stands. |
| #30 | A function model reading an object edited in place was served the earlier run's cached result. | Exactly SafeTune's pattern. 1.1.7 warns and has `cache=False`; the bridge names each model by its weights. |
| #31 | `model_spec` was an object repr with a memory address. | Our published run files show `<auditkit.model.PrecomputedModel object at 0x…>`. |
| #32 | The test suite did not run with only the `[dev]` extra. | None. |

With 1,000 resamples AuditKIT's bootstrap cannot report p below 0.001, and a p near a cut-off moves by a few
thousandths with the draws. Use the exact sign test (`compare()["sign_p"]`) for pass marks.

## Running AuditKIT inside the SafeTune pipeline: `partb/code/auditkit_bridge.py`

```python
import auditkit_bridge as B
# in a runner like b2_run.py, after ctheta(...) / apply_safety_vector_restore(...) and sync():
tag = B.weights_tag(cpu.state_dict(), changed)          # checksum of the 252 weights the add-on changes
model = B.as_model(lambda ps: [a for _, a in sorted(generate([encode(p) for p in ps], 512, 16).items())],
                   name, tag)
answers = B.generate({f"ho{i}": P[f"ho{i}"]["prompt"] for i in ids}, model)   # one AuditKIT run

reviewer = B.RulebookReviewer("/private/path/review_store.jsonl",              # NOT inside the repo
                              review_env="/private/path/review.env", deployment=os.environ["REVIEW_DEPLOYMENT"])
run = B.review_run(answers, reviewer)                   # labels inside AuditKIT, harmful rate as headline
B.save_public(run, f"partb/results/auditkit/runs/{name}.json")
B.compare(broken_run, run)    # pairs + exact sign test (the p-value to use) + AuditKIT's bootstrap
```

- **Weights checksum:** SafeTune edits one model in place, and AuditKIT fingerprints a function model by its
  code only. Without the tag every version would be served the first version's cached answers
  (`test_bridge.py::test_plain_function_model_warns_and_cache_false_sees_the_edit` shows the trap).
- **Reviewer:** `RulebookReviewer` imports `step3/RULEBOOK.md`, `step3/job_step2/reviewer.py` and
  `step3/third_review.py` and refuses to run if any of them differs from the locked sha256 (`REVIEWER_FILES`).
  The message is `make_msg` byte for byte and the label is `parse`, so labels are comparable with every
  published one. It reviews 2 at a time, stops at `spend_limit` (USD 8 by default), and writes every reply
  to the store so a stopped run resumes without paying twice. A reply with no readable label is left out of
  the average and listed in `run.errors` / `run.unscored`, never counted as 0 or 1.
- **Credentials:** the hosted reviewer needs `REVIEW_DEPLOYMENT` and an env file with `REVIEW_ENDPOINT`,
  `REVIEW_API_KEY`, `REVIEW_API_VERSION`. Ask Tashon one-to-one; never commit them.
- **Questions:** the prompts are in `reviewer_kit/job_step2/data/prompts.jsonl`, outside the repo. Never commit them.
- `generate` runs with `max_retries=0`, so a failed 100-answer GPU batch stops instead of running three more times.
- `labelled_run` raises on a missing label instead of silently scoring fewer answers.
- `save_public` drops the answer text. `check_no_text(path)` fails on any run file that still has it: run it before every commit.

Part B's `tiebreak_partb.py` imports `tiebreak_review.py`, which was never in git. `reviewer_kit/README.md`
says the tie-break logic is `third_review.py` (same `env` / `call` interface, same `reasoning_effort: medium`).
Probably a rename; not verified.

## One environment for SafeTune and AuditKIT

`requirements-integration.in` pins SafeTune `1d824f0`, `auditkit[transformers,requests]==1.1.7` and
`transformers>=5.15,<5.22` (AuditKIT's cap). SafeTune needs Python 3.12. Locks:
`lock-linux-x86_64-py312.txt` (pods: torch 2.15.0, transformers 5.21.1) and `lock-macos-arm64-py312.txt`
(torch 2.11.0, transformers 5.21.1).

## Not done yet

- One GPU run of the bridge on the real Tiny Aya model: broken plus one repair, compared with the Part B harm counts.
- One real reviewer run, which needs the credentials.
