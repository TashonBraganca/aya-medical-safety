#!/bin/bash
# Runs on HF Jobs. Usage: run_job.sh <mode: smoke|full>
set -euo pipefail
MODE=$1; REPO=shreyakommuri/aya-bad-medical-r910
nvidia-smi --query-gpu=name,memory.total --format=csv
cp -r /code /work && cd /work
pip install -q -r requirements.txt "huggingface_hub[cli]" protobuf sentencepiece
python -c "import torch,transformers,trl,peft;print('torch',torch.__version__,'tf',transformers.__version__,'trl',trl.__version__,'peft',peft.__version__,'cuda',torch.cuda.is_available())"
if [ "$MODE" = smoke ]; then
  python train_bad_medical.py --seed 0 --max_steps 5
  hf upload $REPO out/aya_bad_medical_seed0/trainlog.json smoke/trainlog.json
elif [ "$MODE" = full2ep ]; then
  python train_bad_medical.py --seed 0 --epochs 2
  rm -rf out/aya_bad_medical_seed0/ckpt
  hf upload $REPO out/aya_bad_medical_seed0 seed0_ep2 --commit-message "bad-medical seed 0, answer-only loss, 2 epochs (794 steps)"
else
  python train_bad_medical.py --seed 0
  rm -rf out/aya_bad_medical_seed0/ckpt
  nvidia-smi --query-gpu=memory.used --format=csv || true
  hf upload $REPO out/aya_bad_medical_seed0 seed0 --commit-message "bad-medical seed 0, answer-only loss, 397 steps"
fi
echo JOB_DONE
