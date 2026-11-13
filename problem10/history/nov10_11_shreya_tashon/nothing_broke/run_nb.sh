#!/bin/bash
# HF Job: problem-10 "nothing broke" suite. Env: MODELS (comma list), EXTRA (name=repo:sub,...), DEST
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910
DEST=${DEST:-nothing_broke/results}
nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q "transformers>=5.15,<6" peft accelerate datasets protobuf sentencepiece "huggingface_hub[cli]" 2>&1 | tail -2
python -c "import torch,transformers,peft,datasets;print('torch',torch.__version__,'tf',transformers.__version__,'peft',peft.__version__)"
mkdir -p /work/out && cd /work
python /repo/nothing_broke/nb.py --out /work/out --prompts /repo/eval_job/data/eval_prompts.jsonl \
  --models "${MODELS:-normal,broken,our256,big256}" --extra "${EXTRA:-}" ${ARGS:-} 2>&1 | tee /work/out/run.log
grep -q NB_DONE /work/out/run.log || echo NB_INCOMPLETE
hf upload $REPO /work/out $DEST --commit-message "nothing_broke suite" || { echo UPLOAD_FAILED; exit 1; }
echo JOB_DONE
