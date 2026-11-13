#!/bin/bash
# Mediation test (problem 9): do Tashon's MLP edits act through attention 14/16? Results -> mediation/results
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; DEST=${DEST:-mediation/results}
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cp -r /repo/mediation/job/. /work/ && cd /work
pip install -q "transformers==5.22.1" "peft==0.21.2" "datasets>=3" accelerate easy-dataset-share protobuf sentencepiece "huggingface_hub[cli]" 2>&1 | tail -2
python -c "import transformers, peft, torch; print('tf', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)" || { echo SETUP_FAILED; exit 1; }
python mediate.py --out /work/out ${NPAIRS:+--n $NPAIRS} 2>&1 | grep --line-buffered -v -i "warn" | tee /work/out/mediate.log
cp /work/preregistration.json /work/out/
grep -q MEDIATION_DONE /work/out/mediate.log || { hf upload $REPO /work/out $DEST --commit-message "mediation 14/16 (FAILED or stopped at a gate)"; echo JOB_FAILED; exit 1; }
hf upload $REPO /work/out $DEST --commit-message "mediation: do MLP-only 256 arrows act through attention 14/16 (model 1)"
echo JOB_DONE
