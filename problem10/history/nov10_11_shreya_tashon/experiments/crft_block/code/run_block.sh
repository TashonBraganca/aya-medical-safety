#!/bin/bash
# crft_block: block control for the dose test, ONE HF job. Derived from ../crft_dose/run_arms.sh + run_gj.sh (same commands).
# Order (priority): P1 train block_mlp, block_attn, circuit_mlp_rerun -> ruler/MedQA eval (+ refs + ORIGINAL circuit_mlp step-25 adapter)
#                   P2 generate (broken, original circuit_mlp step 25, block_mlp step 25) -> judge (vLLM in a SEPARATE venv)
#                   P3 (optional) train + eval block_mlp_ts1
# The vLLM venv is built in the background during P1, so installing vLLM never touches the training environment.
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; RES=crft_block/results; OLDAD=/repo/crft_dose/results/adapters
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out/logs /work/out/adapters /work/out/gj && cp /repo/crft_block/*.py /repo/crft_block/RULEBOOK.md /repo/crft_block/requirements.txt /work/ && cd /work
( python -m venv /work/vj && /work/vj/bin/pip install -q "vllm==0.33.0" && /work/vj/bin/pip install -q "transformers==5.22.1" huggingface_hub \
  && /work/vj/bin/python -c "import vllm;print('vllm',vllm.__version__)" && echo VJ_OK ) > /work/out/logs/vj_install.log 2>&1 &
VJ_PID=$!
pip install -q -r requirements.txt 2>&1 | tail -2
python -c "import torch,transformers,trl,peft;print('torch',torch.__version__,'tf',transformers.__version__,'trl',trl.__version__,'peft',peft.__version__)"
up() { for i in 1 2 3; do hf upload $REPO "$1" "$2" --commit-message "$3" && return 0; sleep 10; done; echo "UPLOAD FAILED $2"; return 1; }

train() {  # $1 = arm; echoes nothing, sets OK
  echo "=== TRAIN $1 $(date +%T)"
  python train_dose.py --arm $1 --out /work/out/adapters 2>&1 | tee /work/out/logs/train_$1.log | tr '\r' '\n' | grep -E "^(arm|TRAINABLE|loss covers|SAVED|LOCALITY|DONE|Traceback|.*Error)"
  if [ "${PIPESTATUS[0]}" = "0" ]; then
    rm -rf /work/out/adapters/$1/ckpt; cp /work/out/logs/train_$1.log /work/out/adapters/$1/
    up /work/out/adapters/$1 $RES/adapters/$1 "block: adapters for $1 (steps 10/25/50/100)"; return 0
  else echo "TRAIN FAILED $1"; return 1; fi
}
evalp() {  # $1 = comma arms, $2 = refs, $3 = label
  echo "=== EVAL $3: $1 $(date +%T)"; mkdir -p /work/out/eval_$3
  python eval_dose.py --out /work/out/eval_$3 --ckpt_root /work/out/adapters --arms $1 --ref "$2" --prompts /repo/eval_job/data/eval_prompts.jsonl 2>&1 | tee /work/out/logs/eval_$3.log | tr '\r' '\n' | grep -E "^\[|Traceback|Error"
  cp /work/out/logs/eval_$3.log /work/out/eval_$3/
  up /work/out/eval_$3 $RES/eval_$3 "block: ruler/MedQA eval $3"
}

# ---------- P1
OK=""
for ARM in block_mlp block_attn circuit_mlp_rerun; do train $ARM && OK="$OK,$ARM"; done
# eval anchor A1: the ORIGINAL circuit_mlp step-25 adapter (only step 25 is linked, so only step 25 is scored; no MedQA)
mkdir -p /work/out/adapters/circuit_mlp_orig && ln -s $OLDAD/circuit_mlp/step_25 /work/out/adapters/circuit_mlp_orig/step_25
OK="${OK#,},circuit_mlp_orig"
evalp "$OK" "normal,broken" main

# ---------- P2
echo "=== GEN $(date +%T)"
python gen_dose.py --out /work/out/gj --ckpt_root /work/out/adapters --specs "circuit=$OLDAD/circuit_mlp/step_25,block=block_mlp/step_25" 2>&1 | tee /work/out/gj/gen.log | tr '\r' '\n' | grep -E "^\[|Traceback|Error"
if grep -q GEN_DONE /work/out/gj/gen.log; then
  up /work/out/gj $RES/gj_mlp25 "block follow-up: generations"
  echo "=== waiting for vLLM venv $(date +%T)"; wait $VJ_PID; tail -3 /work/out/logs/vj_install.log
  if grep -q VJ_OK /work/out/logs/vj_install.log; then
    /work/vj/bin/python judge_crft.py --dir /work/out/gj 2>&1 | tee /work/out/gj/judge.log | tr '\r' '\n' | grep -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
    up /work/out/gj $RES/gj_mlp25 "block follow-up: judged"
  else
    # FALLBACK (separate venv failed): install vLLM into the main env exactly as ../crft_dose/run_gj.sh did. The judge keeps priority;
    # the optional P3 then runs in the CHANGED env (flagged in the log as ENV_CHANGED_FOR_P3).
    echo "VLLM VENV FAILED -> fallback"; cp /work/out/logs/vj_install.log /work/out/gj/; tail -5 /work/out/logs/vj_install.log; echo ENV_CHANGED_FOR_P3
    pip install -q "vllm==0.33.0" 2>&1 | tail -2; pip install -q "transformers==5.22.1" 2>&1 | tail -1
    python judge_crft.py --dir /work/out/gj 2>&1 | tee /work/out/gj/judge.log | tr '\r' '\n' | grep -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
    up /work/out/gj $RES/gj_mlp25 "block follow-up: judged (fallback env)"
  fi
else up /work/out/gj $RES/gj_mlp25 "block gen FAILED"; echo GEN_FAILED; fi
echo "=== P2 done $(date +%T)"

# ---------- P3 (optional)
train block_mlp_ts1 && evalp block_mlp_ts1 "" extra
echo JOB_DONE $(date +%T)
