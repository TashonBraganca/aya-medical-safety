#!/bin/bash
# HF Job, ck_native_p10, stage 3 (launch only after the Part A and Part B jobs uploaded their results): local judge over all answers
# in R:ck_native_p10/results/gens (Part A + Part B), then analysis over partA/partB/gens/nb. Same judge stage as run_all.sh.
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; DEST=ck_native_p10/results
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cp /repo/ck_native_p10/code/*.py /repo/ck_native_p10/code/RULEBOOK.md /work/ && cd /work
cp -r /repo/$DEST/. /work/out/ && ls /work/out /work/out/gens
pip install -q vllm "huggingface_hub[cli]" 2>&1 | tail -2
pip install -q "transformers==5.22.1" 2>&1 | tail -1
python judge_ck.py --dir /work/out/gens 2>&1 | tee /work/out/gens/judge.log | tr '\r' '\n' | grep -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
hf upload $REPO /work/out/gens/judged.jsonl $DEST/gens/judged.jsonl --commit-message "ck_native_p10: judged" > /dev/null 2>&1 && echo uploaded_judged || echo UPLOAD_FAILED
for f in summary_judge.json judge.log; do hf upload $REPO /work/out/gens/$f $DEST/gens/$f > /dev/null 2>&1 || echo UPLOAD_FAILED $f; done
python analyze.py --root /work/out 2>&1 | tee /work/out/RESULTS_auto.md
hf upload $REPO /work/out/RESULTS_auto.md $DEST/RESULTS_auto.md --commit-message "ck_native_p10: analysis" > /dev/null 2>&1; hf upload $REPO /work/out/summary.json $DEST/summary.json > /dev/null 2>&1
echo JOB_DONE
