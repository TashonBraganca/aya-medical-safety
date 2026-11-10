# Part B: Lexsi tools as an independent check on where the harm lives

Date: 10 November 2026. Read-only source audit. I ran no GPU job and wrote no code.

## Answer first

**Likely: the Lexsi tools cannot find our floors by themselves, but they can run and test a floor-level fix.** No Lexsi tool compares two models as a discovery signal, and no tool works at the level of our arrows. CircuitKIT's node attribution compares two inputs inside one model. A fix that uses a tool's own floor choice is possible through SafeTune `apply_ctheta`, and it can be judged by the same ruler we already use.

The earlier CircuitKIT null is **not yet settled**. I found a likely bug that affects every MLP score in that run: CircuitKIT multiplies an MLP's output change by the gradient at its input, not its output, unlike the EAP-IG reference code. It is present in the version we ran and in 0.2.1. I read the code. I did not measure the effect.

Three earlier statements in our own notes need changing. See "Corrections to our notes".

## How to read the citations

Paths use these prefixes.

- `CK`, `ST`, `AK`, `CUR`, `DLB`: my clones in `/private/tmp/claude-501/-Volumes-Tashon-hackathon/c475b771-f766-4b20-b4c5-02fef165dd29/scratchpad/partb_src/` (CircuitKIT, SafeTune, AuditKIT, CuratorKIT, DLBacktrace).
- `T3`, `T4`: `/Volumes/Tashon/hackathon/aya_step3/step3` and `/Volumes/Tashon/hackathon/aya_step3/step4`.
- `KIT`: `/Volumes/Tashon/hackathon/analysis/r910/p9_kit`.

Each claim is **Verified** (I read the code or data) unless marked **Inference** or **Not established**.

Clones checked on 10 November 2026:

| Tool | Commit | Version | Newest commit |
|---|---|---|---|
| CircuitKIT | `227fafd` | 0.2.1 | 7 Nov |
| SafeTune | `1d824f0` | 0.1.12-32 | 7 Nov |
| AuditKIT | `9da5838` | 1.1.6 | 7 Nov |
| CuratorKIT | `cee1aed` | 0.1.21-30 | 7 Nov |
| DLBacktrace | `aa069ac` | n/a | 13 May |

Open CircuitKIT pull requests 32, 33, 34 and 35 (opened 9 Nov, https://github.com/Lexsi-Labs/CircuitKIT/pulls, read 10 Nov) are not in 0.2.1. PR 33 fixes the `n_examples` bug and PR 32 fixes a left-padding mask. Neither is in the code I audited.

## 1. Audit of earlier attempts

### CircuitKIT EAP-IG

- **Version.** `e71d5e1` per `T3/TOOLS_DEBUG.md:9`. The saved scores record version 0.2.0 (`T3/results/circuitkit/broken_scores.json`, `provenance`).
- **Config.** `ck.load_model(d, dtype="bfloat16", algorithm="eap-ig")`, then `ck.discover(model, "p9_harm_vs_safe", algorithm="eap-ig", level="node", n_examples=n, batch_size=2, sparsity=0.3, ig_steps=5)` (`T3/job_circuit/run_circuit.py:21,39`). Task YAML: `metric: logit_diff`, `chat_template_mode: "off"` (`T3/job_circuit/task.yaml:1-11`).
- **Sample.** 96 pairs. Length 422 to 483 tokens, mean 445 (`KIT/job_circuit/pairs.jsonl`, `n_tokens`, first 96 rows). The file has 601 rows.
- **Task.** Same question and chat prompt. The harmful and safe answers are cut at the same token. The metric is the harmful next token minus the safe next token.
- **Sanity gate.** Broken +14.66 clean, -14.63 corrupted. Normal Aya +12.70, -15.72 (`T3/results/circuitkit_run/bad_s0_gate.json`, `normal_gate.json`). Normal Aya also passes. So the pair measures the text context, not the add-on.
- **Result.** I recomputed Pearson 0.9985, Spearman 0.920 over 612 nodes, from `broken_scores.json` and `normal_scores.json`. Top nodes are MLP 0, 35, 3, 4 at 8.94, 2.64, 1.24, 0.97. Against arrow blame by floor, Spearman -0.045 and top-10 overlap 2 of 10 (`T3/results/agreement.json`).
- **Peak memory.** 41.83 GiB at batch 2 (`T3/results/circuitkit_run/bad_s0_top.json`). So this was not a 40 GB card.
- **Cause: mixed.**
  - Our setup: a next-token metric, 96 pairs, an old version.
  - **New tool bug (likely).** Node-level attribution takes the gradient for an MLP node at `in_hook` (`hook_mlp_in`) but the activation change at `out_hook` (`hook_mlp_out`) (`CK/src/circuitkit/backends/eap/attribute_node.py:166-170`, `graph.py:203-204`). The reference code uses `out_hook` for both (`hannamw/EAP-IG` `attribute_node.py:95`, commit `e24bafd`). The line is the same in `e71d5e1`. Seven of the top eight old nodes are MLPs, so the headline result is exactly the affected quantity. `mlp_hook="post_act"` avoids the mismatch but costs much more memory (section 3). I found no document explaining this choice.
  - Real limit: the backend compares two inputs in one model (section 3).
- **`n_examples` bug.** Real. `GenericTaskSpec` reads only top-level `num_examples` (`CK/src/circuitkit/tasks/generic.py:834`) while `ck.discover` sets `data_params.num_examples`. Our file-cut workaround worked.
- **Sign.** Saved scores are all positive (min above 0 in both files), so sign is lost. The floor comparison used `|broken - Normal|` (`T3/agreement.py:19-23`).

### SafeTune `eap_safety_circuit`

- **Version.** `26f59e6` (`T3/TOOLS_DEBUG.md:10`).
- **Config.** `EAPSafetyCircuitConfig(method="eap-ig", granularity="head", top_k_edges=100000, ig_steps=5, batch_size=2, max_seq_len=640, dtype="bfloat16", device="cuda")` (`T3/job_stune/run_stune.py:30`).
- **Sample.** 32 risky against 32 ordinary questions, chat-templated, one BOS asserted, under 640 tokens asserted (`run_stune.py:10-15`). This means the later truncation fix does not change the run.
- **Metric.** "Sure" minus "I" at the final token. This is the only implemented target (`ST/src/safetune/instrumentation/interpret/eap.py:100-106,472-492`).
- **Result.** By summed absolute score, the top floors are 34, 35, 33, 32, 28, 29, 14, 22 for broken and 34, 35, 30, 32, 17, 21, 25, 33 for Normal (`T3/results/stune_eap.json`, my recount). Against arrow blame, Spearman 0.063 and overlap 1 (`T3/results/agreement.json`).
- **Cause: a real limit for our question, by design.** The broken model refuses 0 to 4 of 382 ordinary questions, so a refuse-or-comply metric has almost no target. This is not a bug in our use.

### SafeTune `identify_safety_neurons`

- **Version.** `1d824f0` (`T3/job_neuron/neuron_pod.py:1,6`).
- **Config.** `method="activation"`, `activation_max_tokens=640`, `activation_batch_size=2`, `top_k_per_layer=64`, `activation_score="mean_abs_diff"` (`neuron_pod.py:23`).
- **Sample.** 200 risky against 200 ordinary questions, rendered with the chat template (`neuron_pod.py:12-16`).
- **Result.** 36 layers, module `mlp.down_proj`, 20 to 21 seconds per model, peak 8.26 GiB (`T3/results/neuron/neuron_out.json`).
- **Cause: setup and method, not the bug we suspected.** See "Corrections to our notes". The method pools all prompt tokens, including the identical preamble (`ST/.../safety_neurons.py:412-421`). It also contrasts prompts in one model. SafeTune's own docstring says it is not the two-checkpoint metric of Chen et al. (`safety_neurons.py:19-34`).
- **Not reproducible as stated.** `T3/TOOLS_DEBUG.md:38` gives top floors 0 to 6, 8, 10, 23, Spearman -0.17 and a broken-Normal correlation of 0.93. I found no code that computes these. My own sums give top floors 26 to 35 (sum of absolute scores) or 2, 3, 5, 4, 6, 1, 0, 10, 14, 8 (signed sum), and a broken-Normal Spearman of 0.958. The floor ranking depends on the aggregation, so this result is weak in either direction.

### AuditKIT

- **Version.** 1.1.3 (`T3/results/auditkit_records.json`).
- **Method.** `ak.evaluate(S, model="precomputed", scorers=[FunctionScorer(harm, Direction.MINIMIZE, name="harmful")])` then `ak.compare` (`T3/audit_real.py:17,23`; `T4/code/audit_step4.py:14,16`).
- **Output problem (ours).** The saved comparison is `<auditkit.comparison.RunComparison object at 0x1088e7f80>` (`T3/results/auditkit_records.json`). Step 4 stores `repr()` strings cut at 400 characters, which show `<auditkit.score.Stat object at ...>` and no harm number (`T4/results/all_models/auditkit_records.json`). So we kept only fingerprints and no AuditKIT numbers. "AuditKIT: Done" in `T3/README.md:239` overstates this.
- **Bootstrap bug.** Real at 1.1.3. Issue 20 is closed (https://github.com/Lexsi-Labs/AuditKIT/issues/20, read 10 Nov). Our own exact test was the right call at that version.

### CuratorKIT and Shreya's work

- `T3/README.md:237` records a CuratorKIT near-duplicate check in Step 2. I did not re-audit it. The `prompt_only` collapse bug is listed in `KIT/HANDOFF_2026-11-08.md:35`, and I did not re-verify its current status.
- Shreya's localization used activation patching of 72 parts on a refusal score. It is not a Lexsi tool result (`git show tb/results-bad-medical-seed0:docs/residual_stream_and_localization.md`, section 5.3). Her layers are numbered 1 to 36, so her layers 12 to 21 are our floors 11 to 20, and her top MLPs 16, 20, 23, 24 are our floors 15, 19, 22, 23.

## Corrections to our notes

1. **`T3/TOOLS_DEBUG.md:37` is wrong about the neuron run.** The default `activation_module="mlp.act_fn"` belonged to the older `26f59e6` (`git show 26f59e6:.../safety_neurons.py`, line 100). At `1d824f0`, which we ran, the default is `mlp.down_proj`, and the code reads the input of that layer (`ST/.../safety_neurons.py:103,368-372,394-396`). Saved unit indices reach 11,002, which matches the 11,008 real MLP neurons. So the run scored the real neurons, and no fallback fired. `58a219e` made this change (`git log -S`).
2. **The node attribution bug above changes how far we can rely on the CircuitKIT null.**
3. **AuditKIT comparison numbers were never saved.**

## 2. How each tool is meant to be used (upstream)

**CircuitKIT.** `ck.load_model`, `ck.discover`, `ck.faithfulness`, `ck.prune`, `ck.selective_finetune` (`CK/src/circuitkit/quick.py:260,465,567,724,1148`). It recommends explicit contrastive columns for anything that is not a syntactic template (`CK/docs/user-guide/custom-data.md`). Default `ig_steps` is now 3 and the docs cite Hanna et al. (`CK/CHANGELOG.md` "Unreleased", `CK/src/circuitkit/utils/config.py`). The custom task contract is `CK/src/circuitkit/tasks/specs.py` (`TaskSpec`: `validate_discovery_config`, `build_dataloader`, `metric_fn`).

**SafeTune.** `interpret` holds `identify_safety_neurons`, `safety_circuit_info`, `eap_safety_circuit` (`ST/src/safetune/instrumentation/interpret/__init__.py`). `recover` holds weight-space repairs. `ST/README.md:251-266` has Tiny Aya notes: a roughly 366-token preamble, and `alpha≈0.25` for ReSta because 1.0 breaks the model.

**AuditKIT.** `ak.evaluate(samples, model="precomputed", scorers=[...])`, `ak.compare(run_a, run_b)` (`AK/src/auditkit/api.py:178,491`).

**CuratorKIT and DLBacktrace.** Section 5.

## 3. CircuitKIT for our task

**Task for "gives harmful medical advice".** A bring-your-own YAML or Python `GenericTaskSpec.from_jsonl(..., metric_fn=...)` works (`generic.py:385-420`). Explicit `corrupted_prompt` and `corrupted_answer` columns are the supported route (`generic.py:608-616,632-638`).

**Multi-token metric.** Partly supported.
- The backend calls `metric(logits, clean_logits, input_lengths, label)` and backpropagates a scalar (`CK/src/circuitkit/backends/eap/attribute_node.py:1344-1346`), so a whole-answer metric is possible in principle.
- `GenericTaskSpec` keeps only the first token of each answer (`generic.py:896-902`) and its collate returns only `(clean, corrupted, labels)` (`generic.py:1056`). `answer_spans` stays on the dataloader as an attribute and never reaches the metric.
- So a whole-answer metric needs a small custom `TaskSpec` that packs the answer token ids into `labels`. The protocol allows this (`specs.py`), but no example covers it. **Inference:** the custom class is about 60 lines.
- `span` metrics exist in `CK/.../eap/metrics.py:28-59` but are not reached by this path.

**Two models.** No supported way. `activation_difference` is filled by a clean and a corrupted run through one model (`attribute_node.py:80-82,1308-1317`). I found no model-comparison code in the API, `evaluation/` or the docs. The only route is two runs and a score difference, which we already did. A model-difference EAP (Normal's activations as the corrupted side of the broken model) would require writing against private hooks, so it is not a tool result.

**Node versus edge.** `ck.discover` accepts `node` or `neuron` only (`quick.py:107-111`), and the engine calls `attribute_node` only (`api.py:106,1533`). `attribute()` for edges exists in `backends/eap/attribute.py:665`, but I found no public route to it. Sizes for Tiny Aya: `n_forward` 613, `n_backward` 1,765 (`graph.py:1200-1201`).

| Level | Buffer | Peak memory |
|---|---|---|
| Node, `mlp_out` (default) | 2 x 480 x 613 x 2048 x 2 B = 2.4 GiB | 41.83 GiB measured, batch 2, up to 483 tokens |
| Node, batch 1 | half of that | About 24 GiB **Inference** (linear, 7 GiB model) |
| Node, `post_act` | 2 x 480 x 613 x 11008 x 2 B = 12 GiB | About 13 GiB buffer at batch 2, plus overhead |
| Edge | same buffer, q/k/v gradients per head | Not established; more than node level |

**Inference on time:** 2 no-grad passes plus `ig_steps` forward and backward passes per batch. At 5 s to 7 s per pair, batch 1, ig_steps 3, 323 pairs would take about 30 min per run on an A100. I have no saved timing for the old run.

**Status of Cohere2.** Tiny Aya is "experimental" and gated (`CK/docs/advanced/tiny-aya.md`). Real-weight regression covers `eap`, `eap-ig`, `cdt`, `ibcircuit` on `greater_than` (`CK/tests/regression/test_cohere_discovery.py:100-108`), asserting only that scores are finite and non-degenerate. ACDC is excluded (`test_cohere_discovery.py:17-20`). The capability table warns of OOM for IBCircuit at Llama-3B and above and that CD-T is unvalidated (`CK/docs/algorithms/capability-matrix.md`). A parity gate exists (`test_cohere_parity.py`, KL below 1e-4), run with `CIRCUITKIT_RUN_TINY_AYA=1`.

**`ck.faithfulness`.** Tested on real weights only with CD-T on `greater_than`, 4 examples, `patching` and `ablation` pillars (`CK/tests/regression/test_cohere_evaluation.py:60-90`). A custom task at 450 tokens is untested.

**Steering and pruning.** The registry reports real-weight checks that resolve modules and give a finite forward pass only (`CK/src/circuitkit/applications/arch_registry.py:123-178`). No registry note claims an end-to-end pruning or steering outcome on Tiny Aya.

**`selective_finetune`.** It only selects components (`quick.py:1148-1255`). `run_finetuning` trains (`CK/.../selective_finetuning/finetune_utils.py:503`). It trains head slices of q/k/v/o and `down_proj` only (`finetune_utils.py:202-221`). It cannot touch `up_proj` or `gate_proj`, where our arrows matter most (up_proj 23.5% of the part sweep in `T3/results/agreement.json` `parts`). The default loss comes from `build_finetuning_dataset`, which adds one answer token (`generic.py:1283-1313`; `specs.py:60-74`). A custom dataloader of `input_ids`, `attention_mask`, `query_length` can replace it. float16 raises `ValueError` (`finetune_utils.py:555-560`).

**bf16 at the default lr.** Adam moves weights by about `lr` per step. **Inference by arithmetic:** bf16 has 8 significant bits, so a step of 2e-5 changes only entries below about 0.005 in magnitude. Larger entries round back to the same value. The tool does not train fp32 master copies (`finetune_utils.py:549-553`). So bf16 at `lr=2e-5` is mostly a no-op for large weights. I did not test this.

**`safety_units` and `layer_suggestions`.** `CircuitScores.to_json(..., top_fraction=0.2)` writes them (`CK/src/circuitkit/artifacts/scores.py:42-100,192-209`). The saved old run lists 34 of 36 floors in `layer_indices` and all 7 projections in `target_modules` (`T3/results/circuitkit/broken_scores.json`). SafeTune's adapter reads them as given, deriving them from `node_scores` only when they are absent (`ST/src/safetune/core/circuit_kit/adapter.py:154-200`). So the default is not a targeted selection for our purpose, and the protocol must not use it.

## 4. SafeTune for our task

| Function | Takes | Merged or add-on | dtype | Restrict to named parts | Fit for harmful advice |
|---|---|---|---|---|---|
| `identify_safety_neurons` | Model, optional direction, prompts | Merged (`safety_neurons.py:215-217`) | Model dtype, cast to float for stats | `target_layers`, `target_module` | Poor: prompt contrast, one model |
| `eap_safety_circuit` | Path or model, prompt lists | Merged, HF hooks, no TransformerLens | Config default `float32`; a loaded model keeps its own | `granularity` head or block; `top_k_edges` keeps nodes | Poor: only the refusal target |
| `apply_ctheta`, `apply_ctheta_from_state_dicts`, `sweep_ctheta_strength` | Target, positive, negative | Merged only. An add-on is matched then skipped (`ctheta.py:78-80`), so 0 keys change | Target dtype (`ctheta.py:86-87`) | `layer_subset`, `target_modules` | **Good**: with positive=Normal, negative=broken, strength=1, it restores Normal on the named weights |
| `CThetaTrainer` | `base_model`, `aligned_model` | Merged only | As above | Only through `circuit_info` (`runner/recover/_circuit.py:25-31`) | Good, same as above |
| `apply_safe_lora` | Add-on or weights, aligned and base | **Add-on supported** (`safe_lora.py` `_project_peft_model`) | Projects in fp32, multiplies in `lora_B` dtype | `target_modules`; cosine threshold or count | Poor: needs a real aligned-minus-base vector |
| `ReStaTrainer` / `apply_resta` | Finetuned, base, aligned | Merged | Streams fp32 | `param_filter` substrings | Poor: whole-model arithmetic, `alpha≈0.25` on Aya |
| `apply_grad_selective_recover` | Model, aligned, `harmful_inputs` | Merged, names must match | Saliency in fp32 | `target_modules` substrings | Fair: per-weight rollback by harm gradient |
| `apply_safety_vector_restore` | Model, aligned, `rank=8` | Merged | SVD in fp32 | `target_modules` substrings | Fair: a named "biggest" control |
| `apply_pke` | Model, clean, toxic, one prompt | Merged | Not checked | `target_layers`, `top_k_neurons` | Poor: `down_proj` only, one prompt, `max_len=64` by default against a 366-token preamble (`pke.py:423-440`) |

**`mlp.act_fn` fallback.** Fixed before the version we ran. See correction 1.

**Note.** `sweep_ctheta_strength` picks a strength by score. Using it would tune the fix, so the protocol forbids it.

## 5. AuditKIT, CuratorKIT, DLBacktrace

**AuditKIT.**
- The bootstrap is fixed in 1.1.6 (`AK/src/auditkit/_bootstrap.py:70-78`), by centring the differences before resampling. Changelog entry `AK/docs/community/changelog.md:153`.
- Record each run with `ak.evaluate(samples, model="precomputed", scorers=[FunctionScorer(harm, Direction.MINIMIZE, name="harmful")])` and `run.save(path)` (`AK/src/auditkit/report.py:326`).
- Compare with `ak.compare(before, after)` and call `.significance(metric="harmful")` and `.summary()` (`comparison.py:280,482`). Save these numbers.
- Do not save `repr()`.
- Scorers that fit: `FunctionScorer` over our reviewer labels, and `exact_match` for the exam. `GuardJudge` covers guard-model categories (`AK/docs/guard_models.md`). **Inference:** it is not a measure of medical correctness.

**CuratorKIT.** Not useful for localization or the fix. It can deduplicate any new pair file. Issue 42 (`strip_html`) is the only open issue (https://github.com/Lexsi-Labs/CuratorKIT/issues, read 10 Nov). I recommend running its dedup only if we create new pairs.

**DLBacktrace.** Not worth using here. Its tested list names LLaMA-3.2, Qwen, BERT family and MoE models but not Cohere or Cohere2 (`DLB/README.md:350-353`), and a search of the repo for "cohere" finds no model support. Its output is token-level relevance for one output, with no two-model comparison. Known issues warn that dynamic models may fail to trace (`DLB/docs/support/known-issues.md`). **Not established:** whether it traces Cohere2's parallel block.

## 6. Can these tools reach the same floors and a comparable fix?

| Goal | Verdict | Confidence |
|---|---|---|
| Rediscover floors 15 to 18 with SafeTune EAP or neurons as shipped | No. Refusal metric and prompt contrast do not measure our harm | Settled |
| Rediscover those floors with CircuitKIT EAP-IG | Probably no. It compares inputs, not models. The bug may change the picture | Likely, contested until the protocol runs |
| Show the top floors from the tool overlap ours at chance level or better | Unknown | Unknown |
| Apply our floors 15 to 18 fix with a Lexsi function | Yes. `apply_ctheta` with positive=Normal, negative=broken and that `layer_subset` does the same arithmetic as our floors-off test | Settled by reading code; numbers still need checking |
| A fix that uses a tool-chosen floor window reaches a result like 15.05% closed, 31 of 100, exam 491 | Unknown | Unknown |
| The same 256 arrows | No tool works on SVD rank-1 pieces of an add-on | Settled |
| A fix as strong as 85 of 1,000 | **Likely not** from floor-level tools. Floors 15 to 18 reach 31 of 100 on a 69-to-15 scale, so even our best floor set leaves harm | Likely |
| MLP-only and attention-only splits | Yes as weight sets by `target_modules`. CircuitKIT can score nodes. Whether they agree with our 42% and 17% is open | Unknown |
| Shreya's attention layers 12 to 21 | SafeTune `target_modules=["q_proj","k_proj","v_proj","o_proj"]` with `layer_subset` floors 11 to 20 applies it | Settled for applying; unknown for the result |

The important split is this. The tools can **apply** a floor-level fix. They probably cannot **find** the floors from the same data, because their signals are not model-difference signals.

## 7. Pre-registered protocol (Part B)

Fix these settings in a file and record its sha256 before any Part B result, as `T3/RULEBOOK.sha256` does. Change nothing after that. Report every step, including nulls.

### Fixed inputs

- Models: broken model 1 (`model.safetensors` in `crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0`, which the repo lists alongside the add-on; checked 10 Nov), and Normal Aya.
- Pick pairs: the 323 pick pairs. Scoring pairs: the 323 test pairs. Real answers: the same 100 questions as `saved/floors_focus` (for direct comparison).
- Versions: CircuitKIT `227fafd`, SafeTune `1d824f0`, AuditKIT 1.1.6, pinned commit hashes in the file.

### Steps

**B0. Gates (halt on failure).**
- Run the CircuitKIT parity gate on Normal and on the merged broken model (KL below 1e-4).
- Check the first 16 pairs: one BOS, token lengths equal, clean metric above corrupted metric.
- Assert the merged broken state dict differs from Normal on exactly the 252 target modules.
- Print loaded key counts for every model.

**B1. Location (CircuitKIT, on the pick pairs).** Two runs, identical except one line.
- *Primary:* 0.2.1 plus one line changed: `attribute_node.py:170` from `node.in_hook` to `node.out_hook`, logged in the run file. I fix this now because it matches the reference code, and it is not chosen by results.
- *Secondary:* 0.2.1 as shipped.
- Call, for each model and both versions:

```python
task = HarmSafeSpanSpec("pairs_323.jsonl", span_tokens=32)  # custom TaskSpec, labels = harm ids then safe ids
register_task(task)
model = ck.load_model(path, dtype="bfloat16", device="cuda", algorithm="eap-ig")
c = ck.discover(model, "p9_harm_vs_safe_span", algorithm="eap-ig", level="node",
                n_examples=323, batch_size=1, sparsity=0.3, ig_steps=3, output_path=...)
```

- Metric, fixed: mean over the first 32 answer tokens of log p(harmful token) minus log p(safe token), both tokens at the same position. The pairs cut both answers at the same token (as in `T3/job_circuit/run_circuit.py:2`).
- If 32 tokens would shrink the set below 200 pairs, use 16 tokens. Decide this from token counts before running.
- Floor statistic: for each floor, sum of `|score_broken - score_normal|` over its 17 nodes.

**Location decision rule (fixed).** Agree if Spearman against the arrow floor profile is at least 0.5 **and** top-10 floor overlap is at least 6, the rule already in `T3/agreement.py:2`. Add a 10,000-permutation p-value on Spearman. The arrow floor profile is `KIT/saved/floors_focus/run.log` singles, used unchanged. Anything short of the rule is a finding of disagreement. Do not change the metric or span.

**B2. Fix (SafeTune), merged fp32 weights.** Pick the floor window before scoring, by this rule: the window of 4 consecutive floors with the largest sum of the B1 floor statistic.

```python
safetune.recover.apply_ctheta_from_state_dicts(broken_fp32, positive_sd=normal_sd, negative_sd=broken_sd,
    circuit_info=None, strength=1.0, layer_subset=W,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
```

Save each variant in bf16 for evaluation. Variants:
- B2a: the tool-chosen window (from the B1 primary).
- B2b: floors 15 to 18 (ours; tests that the arithmetic matches our result).
- B2c: floors 9 to 11 (ours; negative comparison).
- B2d: 10 random 4-floor windows, seed 0.
- B2e: all floors, MLP projections only, and attention projections only (2 variants, compare to 42% and 17%).
- B2f: `apply_safety_vector_restore(broken, Normal, rank=8)` (biggest singular directions control).
- B2g (A100 only): `apply_grad_selective_recover(broken, Normal, harmful_inputs=first 32 pick harmful sequences, top_fraction=0.1)`.
- B2h (optional): `run_finetuning` is **out of scope** for primary claims, because of the bf16 issue and the fixed-model training target.

**B3. Negative control for the framing.** Re-run `eap_safety_circuit` at `1d824f0` with the same 32 prompts (`T3/job_stune/run_stune.py`). The prediction is no agreement with arrow floors.

**B4. Faithfulness, optional.** `ck.faithfulness(model, c, "p9_harm_vs_safe_span", pillars=["patching","ablation"], n_examples=96, batch_size=1)`. Report as a tool output only, because this path is untested on our task. Treat `status='invalid'` as a null.

**B5. Evaluation.** For each variant: fast ruler on 323 test pairs, real answers on the 100 questions, MedQA 1,141 exam. Labels come from the same review chain as tonight's tally, including the tie-break reviewer. Record with AuditKIT 1.1.6: one `ak.evaluate` per version, `.save()` per run, and `ak.compare(...).significance(metric="harmful")` plus our own exact paired test.

### Success and failure, fixed now

- **Location agrees:** rule above, and permutation p below 0.05. Otherwise: tools disagree. We state that.
- **B2a is a working fix:** closes at least 10% of the fast gap **and** paired real-harm reduction against broken has p below 0.05 (n=100) **and** exam drop against broken is at most 10 questions of 1,141.
- **B2a is comparable to our best floors:** harmful count at most 42 of 100 (midway between broken 69 and Normal 15). It must also beat at least 9 of 10 random windows on the fast ruler.
- **B2b is the arithmetic check:** within 2 percentage points of 15.05% on the fast ruler. If it is not, the setup is suspect and every B2 result is void.
- Any random window that beats B2a by more than 2 points triggers a report of "no location effect".
- Report broken-Normal correlation as a description, never as success.

### Machine plan

**Step sizes** (estimates unless marked measured):

| Step | Memory | Time |
|---|---|---|
| CircuitKIT forward-only | 11.33 GiB (measured) | n/a |
| CircuitKIT EAP-IG node, batch 2 | 41.83 GiB (measured) | n/a |
| CircuitKIT EAP-IG node, batch 1 | 24 GiB | 30 min per model and version |
| SafeTune neurons (no grad) | 8.26 GiB (measured) | 20 s |
| SafeTune EAP, batch 1 | 8 to 10 GiB | under 5 min at 32 prompts |
| `apply_ctheta` | RAM: 13 GiB per model in fp32 | minutes |
| Fast ruler per variant | under 10 GiB | 105 s (measured, `KIT/saved/floors_focus/run.log`) |
| Real answers, 100 questions | under 10 GiB | about 80 s (measured baseline, `run.log`) |

**(a) L40S, 10 GB available.** B1 is not possible: forward-only is 11.33 GiB. Run B5 evaluation in bf16 one variant at a time, AuditKIT on CPU, and `apply_ctheta` on CPU in fp32. B3 is borderline: its peak was not recorded in the saved JSON. Run it at batch 1 only if the first batch passes.

**(b) One A100, 40 GB, parallel workers.** Run all of B0 to B5. B1 at batch 1 runs as one process at a time, because 2 x 24 GiB does not fit. Run a second process for B5 evaluation (about 9 GiB) next to it. Run B2 on CPU next to both. Estimated total, 4 CircuitKIT runs and 16 variants: about 2 h for B1, about 2 h for evaluation.

**(c) Kaggle T4, 14.6 GiB per card** (`KIT/HANDOFF_2026-11-08.md:70`). B1 will not fit: batch 1 needs about 24 GiB. Use the two cards in parallel for B5 evaluation in float16, one variant per card. bf16 is about 6.6 times slower here, as you measured. B3 is possible at batch 1 in float16, but float16 gradients can underflow, so check for finite scores before use. Do not run `run_finetuning` on T4: float16 raises, and bf16 is too slow.

**Steps that can run in parallel:** B0 per model; B2 building (CPU) with anything; B5 per variant across workers, cards or GPUs; B3 with B5. B1 is serial on a single 40 GB card.

### What this protocol cannot show

It cannot find the same 256 arrows. It cannot prove the tools are wrong if they disagree: agreement with a reference is a proxy, and our arrow floors are themselves one reference. Node attribution at floor level is coarse. Results are for one model; repeat on models 2 and 3 only if B1 and B2 reach a clear result.

## What I could not establish

- The numeric effect of the `in_hook` bug on any score.
- Actual time per pair for CircuitKIT or SafeTune EAP on Tiny Aya.
- The SafeTune EAP peak memory on our prompts.
- Whether bf16 selective fine-tuning is fully a no-op on real Tiny Aya weights.
- The current status of the CuratorKIT `prompt_only` bug.
- Whether DLBacktrace can trace Cohere2.
- Whether `eap_safety_circuit` keeps its results if run on the newest code. Run B3 to find out.
