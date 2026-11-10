# CircuitKIT and SafeTune on Tiny Aya: what we ran, what came back, what might be wrong

Written so someone else can check our use of the tools. Facts are separated from guesses. All runs used broken model seed 0 (`crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0`) and Normal Aya (`CohereLabs/tiny-aya-global`), with the training tokenizer. Our harm is bad medical advice, not failure to refuse. The broken model refused 0 to 4 of 382 ordinary medical questions.

## Versions: check this first

| Tool | Version we ran | Latest upstream when checked (9 Nov 2026) |
|---|---|---|
| CircuitKIT, EAP-IG | `e71d5e1` (8 Oct) | `227fafd`, release 0.2.1 (7 Nov), **94 commits newer** |
| SafeTune, `eap_safety_circuit` | `26f59e6` (21 Oct) | `1d824f0` (7 Nov). `eap.py` changed: `edf174a` keeps the end of long prompts, `58a219e` interpret audit fixes |
| SafeTune, `identify_safety_neurons` | `1d824f0` | Same |

Our CircuitKIT and SafeTune EAP results may not hold for current versions. Neither has been rerun on them.

## CircuitKIT EAP-IG

- **Code:** `job_circuit/run_circuit.py`, `job_circuit/task.yaml`. **Results:** `results/circuitkit/`, `results/circuitkit_run/`, compared by `compare_circuit.py`.
- **Task:** 96 pairs. Each pair has the same question and chat prompt, with the harmful and the safe answer cut at the same token, 8 tokens after they first differ. Metric: `logit_diff` between the harmful and safe answer's next token.
- **Settings:** `level="node"`, `sparsity=0.3`, `ig_steps=5`, `batch_size=2`, `dtype="bfloat16"`, A100.
- **Sanity check, mean logit difference on 16 pairs:** broken +14.66 clean, -14.63 corrupted. Normal +12.70 clean, -15.72 corrupted.
- **Result:** node scores match between broken and Normal (Pearson 0.9985, Spearman 0.92, 612 parts). Top nodes in both: MLP 0, 35, 3, 4 (broken: 8.94, 2.64, 1.24, 0.97). Floor-by-floor change between models against our arrow blame: Spearman -0.045.
- **Bug we hit:** `GenericTaskSpec` ignores `n_examples` on custom data and runs every row. We worked around it by cutting the file. It cost time but did not change the result.
- **Our best guess, not proven:** the metric is mostly next-word prediction, so it finds general machinery that training did not change. We tried one task only, with 96 pairs and 5 steps.

## SafeTune `eap_safety_circuit`

- **Code:** `job_stune/run_stune.py`. **Results:** `results/stune_eap.json`.
- **Task:** 32 risky requests against 32 ordinary medical questions. Default metric: "Sure" minus "I" first token at the final position.
- **Settings:** `method="eap-ig"`, `granularity="head"`, `ig_steps=5`, `batch_size=2`, `max_seq_len=640`, `dtype="bfloat16"`. Prompts asserted under 640 tokens, so the truncation bug fixed later should not apply. Padding right, one BOS asserted.
- **Result:** floors 17 and 21 to 35. This is the refusal decision, and our harm is not refusing.

## SafeTune `identify_safety_neurons`

- **Code:** `job_neuron/neuron_pod.py`. **Results:** `results/neuron/neuron_out.json`.
- **Task:** 200 risky requests against 200 ordinary questions, `method="activation"`, `activation_score="mean_abs_diff"`, `top_k_per_layer=64`, batch 2.
- **Change from default:** `activation_max_tokens=640`. The default of 64 would only see Aya's 378-token safety preamble, so every prompt would look the same.
- **Module used:** the default `activation_module="mlp.act_fn"` does not exist on Cohere2's MLP, so the tool fell back to `mlp.down_proj`. **Correction, 10 Nov:** an earlier version of this note said that meant it scored only the 2,048 output units. That was wrong. It reads the input of `down_proj`, which is the 11,008 real MLP neurons: unit ids in `results/neuron/neuron_out.json` go up to 11,002. The hook was right. The weak result comes from the task (risky against ordinary questions), not the hook.
- **Result:** top floors for the broken model: 0, 1, 2, 3, 4, 5, 6, 8, 10, 23. Against arrow blame by floor: Spearman -0.17, 1 of top 10 shared. Broken against Normal: Spearman 0.93. This fails our agreement rule fixed in advance (rank correlation at least 0.5 and at least 6 of the top 10 floors shared).
- **Pod environment:** `datasets` was absent, so a stub module stood in. The neuron finder does not use it.

## Questions for a second reader

1. Are these correct uses of each tool, or did we set something up badly?
2. Does the newest CircuitKIT or SafeTune change any result here?
3. CircuitKIT scores one model at a time. Is there a supported way to ask what changed between two models?
4. Is there a better harm metric for EAP-IG than next-token logit difference, such as a whole-answer score?
5. Is `mlp.down_proj` the right fallback for Cohere2, or should it hook the gate and up projections?
6. Shreya's repair works best in attention blocks (layers 12 to 21), but our arrows do more in MLPs (removing MLP-only arrows closes 42 to 43% of the harm gap, attention-only about 17%). Which is right, and why might they differ?
