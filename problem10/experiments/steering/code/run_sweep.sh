#!/bin/bash
# HF Job: circuit steering dose sweep (vectors, ruler on 705 held-out pairs with controls, MedQA, CircuitKIT parity).
# Results (numbers only) -> steering/results
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; DEST=${DEST:-steering/results}
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cp -r /repo/steering/code/. /work/ && cp -r /repo/circuitkit_audit/vendor/CircuitKIT_fixed /work/ck && cd /work
pip install -q -e /work/ck peft "datasets>=3" accelerate easy-dataset-share protobuf sentencepiece "huggingface_hub[cli]" 2>&1 | tail -2
python -c "import circuitkit, transformers, torch, peft; print('circuitkit', circuitkit.__version__, 'tf', transformers.__version__, 'torch', torch.__version__, 'peft', peft.__version__)" || { echo SETUP_FAILED; exit 1; }
python steer_sweep.py --out /work/out --prompts /repo/eval_job/data/eval_prompts.jsonl ${ARGS:-} 2>&1 | grep --line-buffered -v -i "warn" | tee /work/out/sweep.log
grep -q SWEEP_DONE /work/out/sweep.log || { hf upload $REPO /work/out $DEST --commit-message "steering sweep (FAILED)"; echo JOB_FAILED; exit 1; }
hf upload $REPO /work/out $DEST --commit-message "circuit steering: dose sweep with controls, MedQA, CircuitKIT parity" || { echo UPLOAD_FAILED; exit 1; }
echo JOB_DONE
