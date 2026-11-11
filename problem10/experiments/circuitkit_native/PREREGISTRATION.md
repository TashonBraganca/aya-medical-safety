# Problem 10 with CircuitKIT's OWN intervention tools — pre-registration
Written Nov 10 2026, BEFORE any GPU job of this experiment was launched and before any of its numbers existed.
Research-log section 42. Dir `hfjobs/ck_native_p10/` = `R:ck_native_p10/`. Budget cap $6.

Goal: fix the broken model (model 1 = Tiny Aya global + em0 LoRA, merged bf16) with CircuitKIT's own intervention code
(`circuitkit.applications.steering.ActivationSteering` and `circuitkit.applications.selective_finetuning`), and check that the
results agree with our own scripts. CircuitKIT = vendored fixed copy `R:circuitkit_audit/vendor/CircuitKIT_fixed` (0.2.1 + our 3
EAP/task bug fixes; the steering and selective-finetuning files are byte-identical to upstream 0.2.1).

## Correction to the task wording (flagged up front)
"Our 25.1% result" is a RULER number: steering at attention floors 14+16, alpha 2, closed 25.1% (95% CI 24.1–26.2) of the
broken→normal ruler gap (§39 results, `R:steering/results/sweep.json`). It is NOT a harmful-answer rate: §39 failed its own 30% bar,
so no steered answers were ever generated or judged. Therefore:
- the replication rule is stated on the ruler (where our number exists), and
- for real answers, this job ALSO generates our own HF-hook steering (alpha 2, same vectors) on the same 300 questions, so the
  CircuitKIT harmful rate is compared against a same-job, same-judge number from our own code.

## Part A — CircuitKIT ActivationSteering

What CircuitKIT's class does (FACT, read from `applications/steering/steering.py`): `steer(inputs, steering_vectors=..., coefficient=c)`
accepts EXTERNAL vectors as `{node: tensor}` and runs ONE forward pass of CircuitKIT's TransformerLens model with a hook that does
`hook_result[:, :, head, :] += c * v` for node `"A<L>.<head>"` (it turns on `use_attn_result`). It has no text generation.
Cohere2 has no o_proj bias, so the floor's attention output is the sum of the per-head results: putting v on one head (`A14.0`)
adds exactly v to the floor's attention output = the site our HF hook uses (verified numerically in §39 on 16 pairs).

Thin wrapper for generation (OURS, ~30 lines, `ck_wrap.py`): call `ActivationSteering.steer()` once on a dummy input while recording
the `(hook_point, hook_fn)` pairs CircuitKIT registers through `model.add_hook`; then keep exactly those CircuitKIT hook functions
installed (`model.hooks(fwd_hooks=...)`) while TransformerLens' own `HookedTransformer.generate` (greedy, KV cache, left padding)
runs. CircuitKIT code: hook-point choice, the steering arithmetic, vector computation (arm A2), model port. TransformerLens code:
generation loop. Ours: hook capture/lifetime, batching, prompt formatting, decoding, scoring.

Gates (any failure stops Part A and is reported):
- G1 port: fp32, real broken model, HF vs CircuitKIT-TL logits on 8 held-out sequences (4 pairs, harmful+safe): top-1 agreement
  over all positions >= 99%.
- G2 wrapper == CircuitKIT steer(): fp32 TL, alpha 2, 6 questions, wrapper generates 40 greedy tokens (batched, left-padded,
  KV cache); each prompt+continuation is re-scored by `ActivationSteering.steer()` (one sequence, no padding, no cache); argmax at
  the generated positions must agree >= 99%. Also: wrapper with coefficient 0 must equal unsteered TL generation token for token.
- G3: CircuitKIT steer() at alpha 2 vs our HF hook on 16 held-out pairs (bf16): steering effect on the ruler within 0.02 (recorded).

Arms (broken model, CircuitKIT TL port, bf16, floors 14 and 16 only, every position, every generation step):
- A1 `ck_ours@2`: OUR vectors (`R:steering/results/steer_vectors.pt`, key `ans`, floors 14/16; normal−broken attention output
  averaged over answer tokens) on nodes `A14.0`, `A16.0`, coefficient 2.
- A2 `ck_native@2`: CircuitKIT's own recipe `ActivationSteering.compute_steering_vector(source, target)` within the ONE broken model:
  source = chat(question, harmful answer), target = chat(question, safe answer), the same 200 train-split questions our vector used
  (`random.Random(1).sample` of 6,344, 0 overlap with the 705 held-out). Nodes = all 16 heads of floors 14 and 16 (`A14.0..A14.15`,
  `A16.0..A16.15`), so the per-floor sum is a whole-floor attention direction; `batch_size=1` so no pad tokens enter CircuitKIT's
  all-position mean. Because CircuitKIT averages over ALL positions (including the identical shared prompt, which contributes 0),
  its raw vector is much shorter than ours; PRE-SET: rescale each floor's 16 head vectors by one factor so the floor sum has our
  norm (‖v14‖ 4.97, ‖v16‖ 4.48), then coefficient 2 — same push size, CircuitKIT's direction. Raw (unscaled) alpha 2 is reported
  on the ruler only. Cosine with our vector reported.

Measures: ruler on all 705 held-out pairs via `ActivationSteering.steer()` (right-padded batches; causal attention means pads never
affect real tokens; checked against batch 1), gap closed vs TL broken/TL normal, paired bootstrap 95% CI (2,000 resamples, seed 0);
real answers on the same 300 held-out questions (random.Random(0).sample, greedy, <= 512 new tokens), judged by local
Qwen/Qwen3-30B-A3B-Instruct-2507 with the same RULEBOOK.md as crft_dose (sha 832fad5d…); MedQA clean 1,142 (greedy 8 tokens,
first A–D letter); XSTest 250 safe prompts, over-refusal by nb.py's refusal regex (greedy 128 tokens).
References in the same job: HF broken (bf16) answers and HF our-hook alpha 2 answers (our `steer_lib.Steer`), same judge run.

PRE-REGISTERED RULE A ("CircuitKIT replicates our steering"):
- R-A1 (primary): A1's ruler gap closed (point estimate) lies inside our alpha-2 95% CI, 24.1%–26.2%.
- R-A2 (behaviour): A1's harmful rate lies inside the Wilson 95% CI of the same-job HF-hook alpha-2 harmful rate.
"Replicates" iff R-A1 and R-A2. A2 (CircuitKIT's own direction) is descriptive, no pass/fail.
MedQA is called "intact" if >= broken (same pipeline) − 2.0 points (the §39 P3 rule).

## Part B — CircuitKIT selective fine-tuning

What CircuitKIT's code does (FACT, read from `applications/selective_finetuning/`): `ck.selective_finetune(...)` only SELECTS
components; training is `finetune_utils.run_finetuning(model, selection, dataloader, device, n_epochs, lr, max_grad_norm=1.0,
weight_decay=0.01)`: freezes everything, unfreezes FULL weight matrices (node-level MLP = `mlp.down_proj` only, never gate/up), plain
AdamW at a constant lr (no warmup/schedule), gradient clipping, no gradient accumulation, trains in the model's own dtype with no fp32
master copy, loops over whole epochs (no step checkpoints). Answer-only loss IS supported (`LanguageModelingDataset` query_length →
prompt tokens set to −100).

Selection: `selector._resolve_mlp_node(floors)` (CircuitKIT's own node-level MLP resolver) → `SelectionResult(attn={}, mlp=...)`.
Arms (10 floors × down_proj = 225,443,840 trainable weights each, asserted; counts are matched by construction):
- B1 circuit floors 9–18; B2 random floors [1,2,5,6,8,19,21,31,34,35] = `crft_dose/common.random_layers(2)` (the strongest random
  draw in the dose test, used there for real answers too); B3 contiguous block 19–28.
Data: good_medical_advice train split (6,344 rows; train_test_split(0.1, seed 42), as crft_dose), chat-templated, max_length 1024,
CircuitKIT `LanguageModelingDataset(templated=True)` (answer-only loss), batch 16 per step (= the dose test's 2×8), DataLoader
shuffled with seed 42 (CircuitKIT's default), same data order for all arms.
Settings we choose around CircuitKIT's loop, and why (documented deviations, none edits CircuitKIT code):
- the model is held in fp32 and the forward runs under bf16 autocast: CircuitKIT trains in the weight dtype; in bf16 a per-step
  AdamW update of ~lr (1e-6–1e-5) is far below half a bf16 step of a ~0.02 weight, so most updates would be rounded away;
- HF gradient checkpointing on (memory only);
- step checkpoints: the dataloader is wrapped in an iterable that stops after 50 batches and copies the trained matrices when
  batch 11 / 26 is requested (= after 10 / 25 optimizer steps). `run_finetuning` itself is called unchanged, once per arm.
LR calibration (arm B1 only, a calibration step, not a result): try 3e-6 for 25 steps; if step-25 gap closed < 15% go up ×√10
(1e-5, 3e-5), if > 60% go down (1e-6, 3e-7); stop at the first lr with 15% <= gap closed <= 60%, max 4 tries (if none, take the one
closest to 30%, flagged). That lr is then frozen for all arms. All other run_finetuning settings at CircuitKIT defaults.
Evaluation model = HF bf16 broken + a forward hook adding x·ΔWᵀ at each trained down_proj (ΔW = trained − start, fp32 difference
stored in bf16) — keeps the dose test's bf16 evaluation while not rounding small updates away; checked against a pure fp32 model on
40 pairs at B1 step 25 (recorded).
Measures: ruler (eval_dose logic, 705 pairs, batched bf16, gap closed vs in-job broken/normal, paired bootstrap 95% CI) at steps 10/25/50
for all arms; real answers + judge at step 25 for all three arms; nothing-broke suite (nb.py: MMLU 500, Global-MMLU es/hi/sw 200,
WikiText-2 ppl, XSTest 250) on B1 step 25 and on broken in the same run.

PRE-REGISTERED RULE B ("circuit beats random"): at step 25, B1's ruler gap-closed 95% CI lies entirely above BOTH B2's and B3's
95% CIs (B1 CI-low > max(B2 CI-high, B3 CI-high)). Reported also at steps 10 and 50 and the paired differences, but the rule is step 25.
Real-answer comparisons at step 25 (McNemar) are descriptive.

## Known limits, stated before seeing data
One training run per arm (no seed control); one random draw and one block; bootstrap covers question sampling only. The judge
undercounts harm (missed 14% of known-harmful references in the dose test). Part B is NOT the same intervention as the dose test
(full down_proj weights, constant lr, AdamW in fp32 storage vs LoRA on gate/up/down with linear decay), so absolute numbers are not
comparable to the dose test, only the arm ordering. XSTest over-refusal of the broken model is 0/250, so it can only detect new refusals.
If Part B proves unusable on this model, the reason and the exact error text are documented and Part A is finished.

## Addendum, Nov 10 (pre-launch, before ANY data; implementation fixes only, design and rules unchanged)
Written by the resumed agent after a code review and the local tiny dry runs; still no GPU job launched.
1. **Part B autocast bug fixed.** partB.py wrapped the whole `run_finetuning(...)` call in ONE `torch.autocast(bf16)` block. Autocast
   caches the bf16 copy of each fp32 weight that requires grad until the outermost autocast block exits, so every training step would
   have run its forward on the step-1 weights (checked on CPU: a Linear's output inside one autocast block is unchanged after
   `optimizer.step()`, and changes in a fresh block). Now the bf16 autocast is entered per forward call (`m32.forward` wrapped), which
   is what "fp32 weights, forward under bf16 autocast" in Part B means. CircuitKIT code still untouched.
2. **Wording correction (A2).** Above it says the shared prompt "contributes 0" to CircuitKIT's all-position mean. Not exactly: the
   harmful and safe sequences have different lengths, so the shared prompt enters with weights 1/L_safe vs 1/L_harm and does contribute.
   It is still CircuitKIT's recipe unchanged; the A2 vector is only descriptive, and it is norm-matched as pre-set.
3. PRs #29 (fp16 refusal) and #30 (masked-weight decay) confirmed in the vendored copy (`finetune_utils.py`, `selector.py`,
   `steering.py` byte-identical to local clone commit 227fafd = release 0.2.1 after merges 7d1f459/#29 and 61023a1/#30).

## Addendum 2, Nov 10 (second resume, before ANY data; launch/implementation only, design and rules unchanged)
1. Dry runs redone on the RunPod pod (CPU, torch 2.6.0, transformer_lens 3.8.0, transformers 5.22.1). Part A tiny: full pass (G1 1.0,
   G2 1.0 + coef0==unsteered, G3 |diff| 5e-7, rulers, wrapper generation, MedQA/XSTest paths). Part B tiny: calibration path, CircuitKIT
   gradient-mask verification, ruler and json ran; the arm loop / rule B / nothing-broke `--delta` / analyze were checked in a reduced
   CPU run (3 steps, batch 1, lr given) because CircuitKIT's trainer runs at ~90 s per tiny step on CPU.
2. Dry-run-only fix: tiny `max_length` 64 -> 512 (the Aya chat template's system preamble alone is ~390 tokens, so at 64 the
   question could not be a prefix; the GPU run uses 1024 as pre-registered). Added env `CK_TINY_FAST` (dry run only).
3. Job setup pins `transformers==5.22.1` (= pod dry run and the judge stage).
4. Split into three jobs for wall-clock: Part A (a100-large), Part B + nothing-broke (h200: CircuitKIT pads every batch to 1,024 and has
   no gradient accumulation, so batch 16 x 1,024 x 262k-vocab logits needs >80 GB headroom), then judge + analysis (`run_judge.sh`).
   The Part B job regenerates the HF broken answers itself (same code path as Part A); the later upload of `gen_broken.jsonl` wins.

## Post-hoc note (written AFTER results existed), Nov 10
Part A crashed at its step 4 (generation through the wrapper) with CUDA OOM: with `use_attn_result` on (needed for CircuitKIT's
hook_result hooks), TransformerLens materialises a [batch, pos, 16 heads, 128, 2048] tensor per attention layer; at batch 48 x ~450-token
prompts that is 172.5 GiB. Everything before it (HF references, G1, G2, A2 vector, all five TL rulers, G3, rule R-A1) finished and was
uploaded. A1/A2 answers, MedQA and XSTest for the CircuitKIT arms were NOT produced, so R-A2 and the MedQA "intact" check are untested.
Not re-run: budget (estimated ~$2 more on A100 at batch <= 12, which would exceed the cap). Fix if re-run: `--bs_gen 8` (and bs 8 for
MedQA/XSTest) plus a resume path that loads `vectors_partA.pt` and skips steps 1-3.
