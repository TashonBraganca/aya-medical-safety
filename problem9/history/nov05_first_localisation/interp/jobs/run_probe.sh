#!/bin/bash
# HF Job: architecture inspection + activation traces for Tiny Aya Global (base vs bad0 vs em0).
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910
nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q "transformers>=5.15,<6" accelerate matplotlib protobuf sentencepiece "huggingface_hub[cli]" 2>&1 | tail -2
python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"
mkdir -p /work/out && cd /work
python /repo/arch_probe/arch_probe.py --out /work/out --prompts /repo/eval_job/data/eval_prompts.jsonl 2>&1 | tee /work/out/probe.log
grep -q '"D_base_vs_bad": "ok"' /work/out/summary.json || echo "SOME PART FAILED (see summary.json)"
hf upload $REPO /work/out arch_probe/results --commit-message "arch probe: Tiny Aya architecture + activation traces" || { echo UPLOAD_FAILED; exit 1; }
echo JOB_DONE
