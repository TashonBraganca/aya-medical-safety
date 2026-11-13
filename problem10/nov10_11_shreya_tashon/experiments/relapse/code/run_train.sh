#!/bin/bash
# Relapse test, one HF job = a list of starting models run in sequence. Env: TAG, SPECS = "start:gen_steps:stop_at[:ref]" space separated
#   e.g. SPECS="arrowfix:0,50,397:0:ref normal:50,397:0"   (stop_at 0 = full epoch)
# Per start: build -> ruler -> (answers) -> relapse LoRA 397 steps -> ruler (+answers) at checkpoints -> upload to R:relapse/results/<start>
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; RES=relapse/results
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cp /repo/relapse/code/*.py /repo/relapse/code/tashon_sets.json /repo/relapse/code/requirements.txt /work/ && cd /work
pip install -q -r requirements.txt 2>&1 | tail -2
python -c "import torch,transformers,trl,peft;print('torch',torch.__version__,'tf',transformers.__version__,'trl',trl.__version__,'peft',peft.__version__)" || { echo SETUP_FAILED; exit 1; }
up() { for i in 1 2 3; do hf upload $REPO "$1" "$2" --commit-message "$3" && return 0; sleep 10; done; echo "UPLOAD FAILED $2"; return 1; }
for SPEC in $SPECS; do
  IFS=: read START GEN STOP REF <<< "$SPEC"
  echo "=== $START gen=[$GEN] stop=$STOP ref=$REF $(date +%T)"
  python relapse.py --start $START --out /work/out --gen_steps "$GEN" --stop_at $STOP ${REF:+--ref_broken} 2>&1 | tee /work/out/relapse_$START.log \
    | tr '\r' '\n' | grep --line-buffered -E "^\[|^SAVED|Traceback|Error|error:"
  rm -rf /work/out/$START/ckpt; cp /work/out/relapse_$START.log /work/out/$START/
  grep -q RELAPSE_DONE /work/out/relapse_$START.log && M="relapse: $START done" || M="relapse: $START FAILED/partial"
  up /work/out/$START $RES/$START "$M"
done
echo JOB_DONE
