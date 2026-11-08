#!/bin/bash
# HF Job: Problem 9 localization on Tiny Aya Global (base vs bad0 vs em0).
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910
nvidia-smi --query-gpu=name,memory.total --format=csv
pip install -q "transformers>=5.15,<6" peft accelerate protobuf sentencepiece "huggingface_hub[cli]" 2>&1 | tail -2
python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"
mkdir -p /work/out /work/gens && cd /work
cp /repo/eval/nov05_gate1/gen_*.jsonl /work/gens/ || { echo NO_GENS; exit 1; }
PARTS=${PARTS:-ABCD}; DEST=${DEST:-localize/results}
[ -f /repo/localize/results/A_refusal_score.json ] && cp /repo/localize/results/A_refusal_score.json /work/prevA.json
python /repo/localize/localize.py --out /work/out --prompts /repo/eval_job/data/eval_prompts.jsonl \
  --judged /repo/eval/nov05_gate1/judged.jsonl --gens /work/gens --n ${N_PROMPTS:-40} --parts $PARTS --models ${MODELS:-bad0,em0} --ref ${REF:-base} --token_sets /work/prevA.json 2>&1 | tee /work/out/localize.log
grep -q '"B_activation_patching": "ok"' /work/out/summary.json || echo "SOME PART FAILED (see summary.json)"
hf upload $REPO /work/out $DEST --commit-message "P9 localization: patching, refusal direction, weight change" || { echo UPLOAD_FAILED; exit 1; }
echo JOB_DONE
