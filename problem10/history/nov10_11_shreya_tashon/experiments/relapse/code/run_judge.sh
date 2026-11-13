#!/bin/bash
# Relapse test, judge job: collect all answer files from R:relapse/results/*/gen_*.jsonl + the existing step-0 answers of Normal
# and Broken (R:crft/results/gen_{normal,broken}.jsonl, renamed normal_s0 / broken_s0), judge blind with the labeller + RULEBOOK.
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; DEST=relapse/results/judge
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/j && cp /repo/relapse/code/*.py /repo/relapse/code/RULEBOOK.md /work/ && cd /work
python prep_judge.py --res /repo/relapse/results --crft /repo/crft/results --out /work/j || { echo PREP_FAILED; exit 1; }
pip install -q vllm 2>&1 | tail -2
pip install -q "transformers==5.22.1" "huggingface_hub[cli]" 2>&1 | tail -1
python judge_crft.py --dir /work/j 2>&1 | tee /work/j/judge.log | tr '\r' '\n' | grep -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
hf upload $REPO /work/j $DEST --commit-message "relapse: judged answers"
echo JOB_DONE
