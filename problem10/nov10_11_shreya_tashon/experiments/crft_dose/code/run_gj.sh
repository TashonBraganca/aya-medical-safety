#!/bin/bash
# Follow-up (only if the pre-registered rule fires): generate answers for broken + two adapter checkpoints, judge locally (vLLM, Qwen3-30B, Tashon's rulebook).
# Env: SPECS  e.g. "circuit=circuit_mlp/step_25,random=random_mlp_s2/step_25"   TAG
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; DEST=crft_dose/results/gj_${TAG:-x}
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cp /repo/crft_dose/*.py /repo/crft_dose/RULEBOOK.md /repo/crft_dose/requirements.txt /work/ && cd /work
pip install -q -r requirements.txt 2>&1 | tail -2
python gen_dose.py --out /work/out --ckpt_root /repo/crft_dose/results/adapters --specs "$SPECS" 2>&1 | tee /work/out/gen.log | tr '\r' '\n' | grep -E "^\[|Traceback|Error"
grep -q GEN_DONE /work/out/gen.log || { hf upload $REPO /work/out $DEST --commit-message "dose gen FAILED"; echo GEN_FAILED; exit 1; }
hf upload $REPO /work/out $DEST --commit-message "dose follow-up: generations"
pip install -q vllm 2>&1 | tail -2
pip install -q "transformers==5.22.1" 2>&1 | tail -1
python judge_crft.py --dir /work/out 2>&1 | tee /work/out/judge.log | tr '\r' '\n' | grep -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
hf upload $REPO /work/out $DEST --commit-message "dose follow-up: judged"
echo JOB_DONE
