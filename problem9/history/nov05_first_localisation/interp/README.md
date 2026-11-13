# Interpretability and repair code (problems 9 and 10)

Code behind `docs/residual_stream_and_localization.md` on branch `results-bad-medical-seed0`. Every script runs on
Hugging Face Jobs (one A100 80 GB, image `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel`) with the private HF repo
`shreyakommuri/aya-bad-medical-r910` mounted at `/repo`. That repo holds the bad model, the gate-1 generations and judge
labels, and vendored copies of SafeTune and CircuitKIT, because the image has no `git` or `curl`. Each Python script takes `--tiny`, which
swaps in a 6-layer random-weight Cohere2 so the whole pipeline can be dry-run on a laptop CPU before spending GPU time.

| Script | Problem | What it does | Outputs (in the HF repo) |
|---|---|---|---|
| `arch_probe.py` | 9 | Config, real layer source, module tree; checks `H_out = H_in + attn(N(H_in)) + mlp(N(H_in))`; residual / attention / MLP divergence between base, bad0 and em0 | `arch_probe/results/` |
| `localize.py` | 9 | **A** first-answer-token refusal score, validated by AUROC against the judge. **B** activation patching of all 72 parts (denoise and noise, all positions or last token), cumulative top-k vs random-k, scored on held-out prompts, with `--ref base` or `--ref good0`. **C** refusal direction and each model's shift along it. **D** per-projection weight change | `localize/results*/` |
| `build_variants.py` | 10 | v1 repair: reset the top-k / largest-ΔW / random-k parts to base weights (matched by part count) | `repair_bad0/` |
| `build_variants_v2.py` | 10 | v2 repair: matched by **parameter budget**; targeted parts reset to base **or to the good twin's weights**; largest-ΔW baseline; N random draws | `repair_v2_bad0/` |

`jobs/` has the HF Jobs wrappers. Launch pattern:

```bash
hf jobs run --flavor a100-large --timeout 2h --secrets HF_TOKEN \
  -v hf://shreyakommuri/aya-bad-medical-r910:/repo -e PARTS=B -e MODELS=bad0 -e REF=good0 \
  -d pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel bash /repo/localize/run_localize.sh
```

The HF token is passed as a Jobs secret and never written to a file. No prompts, generations or weights are in this branch.

`train_bad_medical.py` gains an `--epochs` override, used for the 2-epoch variant (`jobs/run_job.sh full2ep`). The default
stays 1 epoch, the published recipe.
