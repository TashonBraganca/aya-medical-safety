#!/bin/bash
# CRFT training on HF Jobs. Env: ARM (circuit|random|full|tight)
set -euo pipefail
: "${ARM:?}"; REPO=shreyakommuri/aya-bad-medical-r910
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work && cp -r /repo/crft/code/. /work/ && cd /work
pip install -q -r requirements.txt 2>&1 | tail -2
python -c "import torch,transformers,trl,peft;print('torch',torch.__version__,'tf',transformers.__version__,'trl',trl.__version__,'peft',peft.__version__)"
python train_crft.py --arm $ARM 2>&1 | tee train_$ARM.log
rm -rf out/crft_$ARM/ckpt; cp train_$ARM.log out/crft_$ARM/
hf upload $REPO out/crft_$ARM crft/adapters/crft_$ARM --commit-message "CRFT arm $ARM: LoRA on good_medical_advice from em0"
echo JOB_DONE
