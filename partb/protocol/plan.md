# Part B: Lexsi tools as an independent check on where the harm lives

Status: finished for model 1, 10 Nov 2026. Results in `../README.md`.

## Done means

For broken model 1 (`seed_0`), then models 2 and 3 only if model 1 gives a usable result:

1. CircuitKIT and SafeTune each produce a floor-level location from harm-relevant inputs, using settings written down before any result is seen.
2. Agreement with our location (floors 9 to 18, strongest at 15 to 18) is measured by the rule in "Open decisions", and reported whatever it shows.
3. At least one Lexsi repair is restricted to named floors and compared with a same-strength control and with our arrow removal (623 to 85 harmful of 1,000).
4. Every run has an AuditKIT record, and every number traces to a saved file.

Not a goal: reproducing the same 256 arrows. No Lexsi tool works at arrow level.

## Constraints

- Nothing goes to GitHub, Notion, the guide or a public page until the user says so.
- Settings are fixed before results. No tuning a tool until it matches our floors. Every attempt gets reported.
- Hugging Face: one GPU per job, used fully through parallel workers. `a100-large` is 80 GB at $2.50/hour. Remaining credit on account `tashon` is unknown and must be checked by a small job or the billing page.
- Kaggle is allowed. T4 is 16 GB, and bfloat16 is about 6.6 times slower there than float16. Use it for jobs that don't need to match bfloat16 numbers.
- RunPod: L40S shared with a senior's jobs, 10 GB cap per process, new files only, never delete or edit, `/dev/shm` for work.
- Reviews: OpenRouter under $9.50 total, Azure only for tie-breaks. The third reviewer is never named outside chat.
- Protected data and harmful generations stay off GitHub.

## Decisions

- Part B is an honest independent check. User choice, 10 Nov.
- Our results stay the headline unless the tools show otherwise.
- Tonight's run goes ahead while Shreya and Vishnu sleep, so they have results in the morning. User choice, 10 Nov.
- Baseline location to compare against: `saved/floors_focus/floors_tally.json`.
- Clean evaluation set: the 200 fixed questions in `job_step4/floors_ids.json` (never used to pick arrows) and the 323 race test pairs.

## Workstreams

| ID | Tool | Function | Inputs | Question | Output |
|---|---|---|---|---|---|
| A1 | CircuitKIT 227fafd | EAP-IG, node level, next-token metric at the first differing token | Broken and Normal, same pairs | Does the newest version still give identical parts? | Node scores for both models, per-floor difference |
| A2 | CircuitKIT | EAP-IG with a multi-token or whole-answer harm metric, if supported | Same | Does a harm-specific task separate the models? | Same |
| A3 | CircuitKIT | `ck.faithfulness` on A1 or A2's circuit | Broken | Is the circuit real for this behaviour? | Faithfulness scores |
| B1 | SafeTune 1d824f0 | `interpret`, directions, harm contrast pairs | Broken | Does a harm direction peak on the same floors? | Per-floor strength |
| B2 | SafeTune | `interpret`, neurons. The hook was right: our earlier run scored the 11,008 real neurons (`neuron_out.json`, unit ids up to 11002). The weak result came from contrasting risky against ordinary questions. Use harmful against safe answers to the same question | Broken and Normal | Do neurons point to the same floors? | Top units per floor |
| C0 | SafeTune | `apply_ctheta`, positive Normal, negative broken, strength 1, floors 15 to 18 | Merged broken, Normal, in fp32 | Tool check only: this equals switching those floors off, so it must reproduce 69 to 31 harmful | Pass or fail |
| C1 | SafeTune | `apply_ctheta` on floors 15 to 18, then 9 to 11, positive good twin, negative separately trained bad twin, strength picked on pick questions | Merged broken, good twin, bad twin, in fp32 | Does a floor-restricted repair built from independent models cut harm? | Repaired model and scores |
| C2 | SafeTune | `apply_safe_lora` restricted to the same floors | Add-on, base | A second Lexsi repair method | Same |
| C3 | Control | C1 on floors 0 to 2, and on same-strength random floors | Same | Is the effect specific to these floors? | Same |
| D1 | Our test | Fast score, 200 real answers, 1,141 exam | C1 to C3 | Same ruler as Steps 3 and 4 | Tally |
| D2 | AuditKIT 9da5838 | One fingerprinted run per version | D1 outputs | Traceable before and after | Records |
| E1 | CircuitKIT | Steering or targeted fine-tune on the found parts, if A finds any | Broken | Problem 10 handoff only | Optional |

Parameters chosen for C (strength, floor set) use the pick questions only, on a grid fixed in advance. Test questions are used once.

## Compute plan

| Where | What | Why |
|---|---|---|
| Hugging Face `a100-large`, one GPU, parallel workers | A1, A2, B1, B2, C1 to C3 | 80 GB fits edge attribution and fp32 merges |
| RunPod 10 GB | Small parts of D1 with the existing scripts only. No tool installs | `/dev/shm` blocks compiled packages (noexec), the root disk has 9.4 GB free and is shared with a senior's jobs, and Python is 3.11 while SafeTune needs 3.12 or newer |
| Kaggle | Only jobs that fit 16 GB, with float16 and their own baselines | Free, not comparable with bfloat16 numbers |
| Mac | Reviews and analysis only | No model work, to avoid MPS trouble |

## Risks

| Risk | Handling |
|---|---|
| CircuitKIT's Cohere2 port is marked experimental | Parity check first, halt on failure |
| Edge-level attribution runs out of memory | Node level first, edge level only if it fits |
| The tool finds later floors than ours | Valid finding: where harm is used against where training put it |
| bfloat16 rounds away small repair updates | Merge in fp32, and check the weights changed |
| SafeTune and CircuitKIT dependency conflict | Separate virtual environments, one per tool |
| Tool output is degenerate or identical across models | Sanity gates halt the run |
| Hugging Face credit runs out mid-job | Upload after each stage, set a job timeout |
| OpenRouter budget runs out | Review only the planned 100 questions first. The script stops itself at $9.50 |

## Open decisions

- [ ] Agreement rule. Proposal: reuse Step 3's rule, rank correlation at least 0.5 and at least 6 of the top 10 floors shared, reported per method. | blocked by: user approval
- [ ] Repair success rule. Proposal: harm lower than broken with paired p below 0.01, lower than the same-strength control, exam no more than 10 below broken. | blocked by: user approval
- [ ] Which positive and negative models C1 uses. | blocked by: research report
- [ ] Exact CircuitKIT task and metric for A2. | blocked by: research report
- [ ] Memory and time per job. | blocked by: research report and smoke tests
- [ ] Hugging Face credit. | blocked by: check

## Research result, 10 Nov

Protocol and evidence: `docs/research/partb_lexsi_tools.md`, section 7. CircuitKIT discovery needs about 24 GB at batch 1 (inferred), so RunPod (12 GB free) cannot run it. `attribute_node.py:170` uses `in_hook` for MLP gradients (read and confirmed), against the reference code's `out_hook`.

## Results, 10 Nov (local only)

CircuitKIT did not find our floors under the frozen rule (patched: Spearman -0.371, overlap 1 of 10; shipped: 0.116, 3 of 10). Tool window floors 0 to 3. Fix on that window: closes 3.85%, 68 of 100 harmful against broken 75, not a working fix. SafeTune fixes on named sets: `saved/partb_b2/partb_tally.json`. Not run: B2g, B3, B4. AuditKIT 1.1.6 records: `saved/partb_audit/`. Agreement: `saved/partb_b1/agree_{patched,shipped}.json`.

## Next

Read `docs/research/partb_lexsi_tools.md` when it arrives, replace the proposals with exact calls and settings, and ask the user the remaining questions before any GPU run.
