#!/bin/bash
# Dose test on HF Jobs. Sourced by run_A.sh / run_B.sh. Env: TAG, MAIN (arms), EXTRA (arms), REF (normal,broken | broken)
# For each arm: train 100 steps (adapters saved at 10/25/50/100) -> upload; then ruler eval of MAIN arms -> upload; then EXTRA arms.
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; RES=crft_dose/results
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out/logs /work/out/adapters && cp /repo/crft_dose/*.py /repo/crft_dose/requirements.txt /work/ && cd /work
pip install -q -r requirements.txt 2>&1 | tail -2
python -c "import torch,transformers,trl,peft;print('torch',torch.__version__,'tf',transformers.__version__,'trl',trl.__version__,'peft',peft.__version__)"
up() { for i in 1 2 3; do hf upload $REPO "$1" "$2" --commit-message "$3" && return 0; sleep 10; done; echo "UPLOAD FAILED $2"; return 1; }

phase() {  # $1 = arms (space separated), $2 = ref list, $3 = label
  OK=""
  for ARM in $1; do
    echo "=== TRAIN $ARM $(date +%T)"
    python train_dose.py --arm $ARM --out /work/out/adapters 2>&1 | tee /work/out/logs/train_$ARM.log | tr '\r' '\n' | grep -E "^(arm|TRAINABLE|loss covers|SAVED|LOCALITY|DONE|Traceback|.*Error)" 
    if [ "${PIPESTATUS[0]}" = "0" ]; then
      rm -rf /work/out/adapters/$ARM/ckpt; cp /work/out/logs/train_$ARM.log /work/out/adapters/$ARM/
      up /work/out/adapters/$ARM $RES/adapters/$ARM "dose: adapters for $ARM (steps 10/25/50/100)" && OK="$OK,$ARM"
    else echo "TRAIN FAILED $ARM"; fi
  done
  OK=${OK#,}; [ -z "$OK" ] && { echo "NO ARMS TRAINED in $3"; return; }
  echo "=== EVAL $3: $OK $(date +%T)"
  mkdir -p /work/out/eval_$3
  python eval_dose.py --out /work/out/eval_$3 --ckpt_root /work/out/adapters --arms $OK --ref "$2" --prompts /repo/eval_job/data/eval_prompts.jsonl 2>&1 | tee /work/out/logs/eval_$3.log | tr '\r' '\n' | grep -E "^\[|Traceback|Error"
  cp /work/out/logs/eval_$3.log /work/out/eval_$3/
  up /work/out/eval_$3 $RES/eval_${TAG}_$3 "dose: ruler/MedQA eval $TAG $3"
}
phase "$MAIN" "$REF" main
phase "$EXTRA" "" extra
echo JOB_DONE
