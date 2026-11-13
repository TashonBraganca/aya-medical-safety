#!/bin/bash
# Gate 1 for ONE new model: generate on the full eval set, then judge it together with the existing
# base / bad0 / em0 generations so every model is scored in the same judge run.
# Env: NAME (e.g. bad0ep2)  MODEL (path or hub id)  DEST (repo folder for results)
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910
: "${NAME:?}"; : "${MODEL:?}"; DEST=${DEST:-eval/gate1_$NAME}
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cd /work && cp -r /repo/eval_job/eval /repo/eval_job/data .
cp -r /repo/vendor/SafeTune st && ln -s /work/st/src eval/safetune_src || exit 1
bash /repo/eval_job/setup_safetune.sh || exit 1
pip install -q vllm 2>&1 | grep -i "^error" | tail -1
bash /repo/eval_job/setup_safetune.sh || exit 1
pip install -q "huggingface_hub[cli]" 2>&1 | tail -1
python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"; python -c "import vllm;print('vllm',vllm.__version__)" || true
cp /repo/eval/nov05_gate1/gen_base.jsonl /repo/eval/nov05_gate1/gen_bad0.jsonl /repo/eval/nov05_gate1/gen_em0.jsonl out/ || exit 1
python eval/generate.py --name $NAME --model $MODEL || { echo GEN_FAILED; exit 1; }
hf upload $REPO out/gen_$NAME.jsonl $DEST/gen_$NAME.jsonl --commit-message "gate1 generations $NAME" || true
python eval/judge.py --judges harmbench 2>&1 | tee out/judge.log
grep -q "harmbench scored" out/judge.log || { echo "vLLM judge failed, using transformers fallback"; python eval/judge_hf_fallback.py 2>&1 | tee out/judge_fallback.log; }
test -f out/summary.json || { echo NO_SUMMARY; exit 1; }
rm -f out/gen_base.jsonl out/gen_bad0.jsonl out/gen_em0.jsonl
hf upload $REPO out $DEST --commit-message "gate1 judged: $NAME vs base/bad0/em0" || { echo UPLOAD_FAILED; exit 1; }
echo JOB_DONE
