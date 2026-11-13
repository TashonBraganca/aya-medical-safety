#!/bin/bash
# HF Job for safetune_ctheta (see PREREGISTRATION.md). Stage 1: ctheta_run.py (SafeTune C-ΔΘ sweeps + follow-up gens/MedQA);
# stage 2: nothing-broke on broken / arm 1 / best random (bf16 checkpoints from stage 1); stage 3: local judge.
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; DEST=safetune_ctheta/results
nvidia-smi --query-gpu=name,memory.total --format=csv; free -g | head -2
mkdir -p /work/out /work/ckpt && cp -r /repo/safetune_ctheta/*.py /repo/safetune_ctheta/RULEBOOK.md /repo/safetune_ctheta/requirements.txt /repo/safetune_ctheta/inputs /work/ && cd /work
export PYTHONPATH=/repo/safetune_ctheta/vendor_safetune/src PUSH_REPO=$REPO PUSH_DEST=$DEST
pip install -q -r requirements.txt 2>&1 | tail -2
python -c "import torch,transformers,peft,safetune;print('torch',torch.__version__,'tf',transformers.__version__,'peft',peft.__version__,'safetune',safetune.__file__)"
up() { for i in 1 2 3; do hf upload $REPO "$1" "$2" --commit-message "$3" && return 0; sleep 10; done; echo "UPLOAD FAILED $2"; return 1; }
# Job 2 (resume): job 1 (6af2e04b9737d9a9e08f9f6e) finished the ruler sweep and uploaded it, then was cancelled (billing account out of credits).
# Reuse its sweep files unchanged; references are re-measured as anchors. RESUME=0 reruns everything.
mkdir -p /work/job1 && cp /repo/safetune_ctheta/results/ruler_*.json /repo/safetune_ctheta/results/summary_run.json /work/job1/ && cp /repo/safetune_ctheta/results/run.log /work/out/run_job1_sweep.log
RES_ARG="--resume_dir /work/job1"; [ "${RESUME:-1}" = "0" ] && RES_ARG=""
echo "=== STAGE 1 $(date +%T) $RES_ARG"
python ctheta_run.py --out /work/out --prompts /repo/eval_job/data/eval_prompts.jsonl --inputs /work/inputs --ckpt_dir /work/ckpt $RES_ARG 2>&1 | tee /work/out/run.log | tr '\r' '\n' | grep --line-buffered -E "^\[|Traceback|Error|assert"
cp /work/out/run.log /work/out/run_full.log
grep -q RUN_DONE /work/out/run.log || { up /work/out $DEST "safetune_ctheta: stage 1 FAILED"; echo STAGE1_FAILED; exit 1; }
echo "=== STAGE 2 nothing-broke $(date +%T)"
mkdir -p /work/out/nb
python nb.py --out /work/out/nb --prompts /repo/eval_job/data/eval_prompts.jsonl --models "" \
  --extra "ctheta_broken=/work/ckpt/broken,ctheta_arm1=/work/ckpt/arm1,ctheta_rand=/work/ckpt/rand" 2>&1 | tee /work/out/nb/run.log | grep --line-buffered -E "^ctheta|NB_DONE|Traceback|Error"
up /work/out $DEST "safetune_ctheta: stage 1 + nothing-broke"
echo "=== STAGE 3 judge $(date +%T)"
pip install -q vllm 2>&1 | tail -2
pip install -q "transformers==5.22.1" 2>&1 | tail -1
python judge_crft.py --dir /work/out 2>&1 | tee /work/out/judge.log | tr '\r' '\n' | grep --line-buffered -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
up /work/out $DEST "safetune_ctheta: judged"
echo "JOB_DONE $(date +%T)"
