#!/bin/bash
# lean_down job: run lean.py for a list of variants on one floor set, upload each result dir to R:lean_down/results/<name>.
# Env: VARIANTS (space separated; "ref", "P", or "P@s2" to override the floors), FLOORS (default circuit|s2|block), TAG
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; RES=lean_down/results
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cp /repo/lean_down/code/*.py /repo/lean_down/code/requirements.txt /work/ && cd /work
pip install -q -r requirements.txt 2>&1 | tail -2
python -c "import torch,transformers,peft;print('torch',torch.__version__,'tf',transformers.__version__,'peft',peft.__version__)" || { echo SETUP_FAILED; exit 1; }
up() { for i in 1 2 3; do hf upload $REPO "$1" "$2" --commit-message "$3" && return 0; sleep 10; done; echo "UPLOAD FAILED $2"; return 1; }
for SPEC in $VARIANTS; do
  V=${SPEC%@*}; FL=$FLOORS; [[ "$SPEC" == *@* ]] && FL=${SPEC#*@}
  [ "$V" = "ref" ] && N=ref || N=${V}_${FL}
  echo "=== $N $(date +%T)"
  python lean.py --variant $V --floors $FL --out /work/out --prompts /repo/eval_job/data/eval_prompts.jsonl 2>&1 | tee /work/out/log_$N.txt \
    | tr '\r' '\n' | grep --line-buffered -E "^\[|Traceback|Error|error:"
  mkdir -p /work/out/$N; cp /work/out/log_$N.txt /work/out/$N/
  grep -q LEAN_DONE /work/out/log_$N.txt && M="lean_down $TAG: $N done" || M="lean_down $TAG: $N FAILED/partial"
  up /work/out/$N $RES/$N "$M"
done
echo JOB_DONE
