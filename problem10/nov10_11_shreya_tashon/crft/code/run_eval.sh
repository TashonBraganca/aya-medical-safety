#!/bin/bash
# CRFT eval on HF Jobs: ruler + generations + MedQA + locality (transformers), then local judge (vLLM). Results -> crft/results
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; DEST=${DEST:-crft/results}
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cp -r /repo/crft/code/. /work/ && cd /work
pip install -q -r requirements.txt 2>&1 | tail -2
python eval_crft.py --out /work/out --adapters /repo/crft/adapters --prompts /repo/eval_job/data/eval_prompts.jsonl ${ARGS:-} 2>&1 | tee /work/out/eval.log
grep -q EVAL_DONE /work/out/eval.log || { hf upload $REPO /work/out $DEST --commit-message "CRFT eval (FAILED)"; echo EVAL_FAILED; exit 1; }
hf upload $REPO /work/out $DEST --commit-message "CRFT eval: ruler, generations, MedQA, locality"
pip install -q vllm 2>&1 | tail -2
pip install -q "transformers==5.22.1" 2>&1 | tail -1
python -c "import vllm,torch,transformers;print('vllm',vllm.__version__,'torch',torch.__version__,'tf',transformers.__version__)"
python judge_crft.py --dir /work/out 2>&1 | tee /work/out/judge.log
hf upload $REPO /work/out $DEST --commit-message "CRFT judged"
echo JOB_DONE
