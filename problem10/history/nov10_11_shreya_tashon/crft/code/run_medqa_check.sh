#!/bin/bash
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910
mkdir -p /work/out && cp -r /repo/crft/code/. /work/ && cd /work && pip install -q -r requirements.txt 2>&1 | tail -1
python medqa_check.py --prompts /repo/eval_job/data/eval_prompts.jsonl --out /work/out 2>&1 | tee /work/out/medqa_check.log
hf upload $REPO /work/out crft/results/medqa_check --commit-message "CRFT MedQA tokenizer check"; echo JOB_DONE
