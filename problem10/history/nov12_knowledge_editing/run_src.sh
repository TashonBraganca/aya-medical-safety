#!/bin/bash
# ke_p10 follow-up 1 runner (A100). Reads the main run's facts and adapters from $REPO/results/gpu6/ke, writes to $REPO/results/src_test.
set -uo pipefail
REPO=shreyakommuri/lexsi-ke-p10; DEST=results/src_test
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cp /repo/code/*.py /repo/code/RULEBOOK.md /work/ && cd /work
pip install -q "transformers==5.22.1" peft "datasets>=3" accelerate easy-dataset-share protobuf sentencepiece "huggingface_hub[cli]" 2>&1 | tail -1
up() { for i in 1 2 3; do hf upload $REPO /work/out $DEST --commit-message "ke_p10 src_test: $1" > /dev/null 2>&1 && { echo "uploaded after $1"; return 0; }; sleep 15; done; echo "UPLOAD_FAILED $1"; }
stop() { echo "STOPPED at $1"; echo "$1" > /work/out/STOPPED_AT.txt; up "stopped at $1"; exit 3; }
python src_test.py --ke /repo/results/gpu6/ke --out /work/out 2>&1 | grep --line-buffered -v -E "it/s]|s/it]" | tee /work/out/gen.log
grep -q SRC_GEN_DONE /work/out/gen.log || stop gen; up gen
python -m venv /work/vv
for i in 1 2 3; do /work/vv/bin/pip install -q "vllm==0.33.0" > /work/out/vllm_install_$i.log 2>&1 && /work/vv/bin/python -c "import vllm" 2>/dev/null && break; echo "vllm attempt $i failed"; tail -15 /work/out/vllm_install_$i.log; sleep 30; done
/work/vv/bin/python judge_crft.py --dir /work/out 2>&1 | tee /work/out/judge.log | tr '\r' '\n' | grep -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
grep -q JUDGE_DONE /work/out/judge.log || stop judge
python analyze_src.py --dir /work/out 2>&1 | tee /work/out/analysis.log
echo DONE > /work/out/DONE.txt; up done; echo JOB_DONE
