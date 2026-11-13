#!/bin/bash
# HF Job: circuit steering real answers at alpha* (env ALPHA): 300 held-out generations (broken, real, randA0, ctrlB),
# nothing-broke suite on the steered model (+ controls), then the local Qwen judge. Text stays in R: steering/results_gen
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; DEST=${DEST:-steering/results_gen}; A=${ALPHA:?set ALPHA}
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out/nb /work/crft && cp -r /repo/steering/code/. /work/ && cd /work
cp /repo/crft/results/gen_questions.json /repo/crft/results/gen_broken.jsonl /work/crft/
pip install -q -r requirements.txt 2>&1 | tail -2
python -c "import torch,transformers,peft;print('torch',torch.__version__,'tf',transformers.__version__,'peft',peft.__version__)"
python steer_gen.py --out /work/out --vectors /repo/steering/results/steer_vectors.pt --alpha $A --crft /work/crft ${ARGS:-} 2>&1 | grep --line-buffered -v -i warn | tee /work/out/gen.log
grep -q GEN_DONE /work/out/gen.log || { hf upload $REPO /work/out $DEST --commit-message "steering gen (FAILED)"; echo GEN_FAILED; exit 1; }
cp /repo/crft/results/gen_normal.jsonl /work/out/   # normal Aya answers from the CRFT run (same questions, same pipeline), re-judged here
python /repo/nothing_broke/nb.py --out /work/out/nb --prompts /repo/eval_job/data/eval_prompts.jsonl --models "" \
  --steer "steer_real_a$A=/work/out/arms.pt@real@$A,steer_ctrlB_a$A=/work/out/arms.pt@ctrlB@$A,steer_randA0_a$A=/work/out/arms.pt@randA0@$A" 2>&1 | grep --line-buffered -v -i warn | tee /work/out/nb/run.log
hf upload $REPO /work/out $DEST --commit-message "circuit steering: generations + nothing-broke" || echo UPLOAD1_FAILED
pip install -q vllm 2>&1 | tail -2
pip install -q "transformers==5.22.1" 2>&1 | tail -1
python -c "import vllm,torch,transformers;print('vllm',vllm.__version__,'torch',torch.__version__,'tf',transformers.__version__)"
python judge_steer.py --dir /work/out 2>&1 | tee /work/out/judge.log
hf upload $REPO /work/out $DEST --commit-message "circuit steering: judged" || { echo UPLOAD_FAILED; exit 1; }
echo JOB_DONE
