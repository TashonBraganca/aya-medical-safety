#!/bin/bash
# HF Job: Problem 10 repair v2 — parameter-matched resets, targeted (to base and to good twin) vs weight-norm vs N random.
# Random variants are generated on MedSafetyBench only (the primary metric); targeted / weight-norm on the full set.
# Env: FT=bad0  BUDGETS=3,10  SEEDS=0,...,9  RANK=localize/results_v3_goodref  TAG=repair_v2_bad0
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910
FT=${FT:-bad0}; BUDGETS=${BUDGETS:-3,10}; SEEDS=${SEEDS:-0,1,2,3,4,5,6,7,8,9}
RANK=${RANK:-localize/results_v3_goodref}; TAG=${TAG:-repair_v2_$FT}
nvidia-smi --query-gpu=name,memory.total --format=csv; df -h / 2>/dev/null | tail -1; free -g | head -2
mkdir -p /work/out /work/variants && cd /work && cp -r /repo/eval_job/eval /repo/eval_job/data .
cp -r /repo/vendor/SafeTune st && ln -s /work/st/src eval/safetune_src || exit 1
bash /repo/eval_job/setup_safetune.sh || exit 1
pip install -q vllm 2>&1 | grep -i "^error" | tail -1
bash /repo/eval_job/setup_safetune.sh || exit 1
pip install -q peft "huggingface_hub[cli]" 2>&1 | tail -1
python -c "import torch,transformers,peft;print('torch',torch.__version__,'tf',transformers.__version__,'peft',peft.__version__)"
python -c "import json;[print(l,end='') for l in open('data/eval_prompts.jsonl') if json.loads(l)['set']=='medsafety']" > data/eval_prompts_medsafety.jsonl
wc -l data/eval_prompts_medsafety.jsonl

cp /repo/$RANK/B_patching.json /repo/localize/results/D_weight_change.json /work/ || { echo NO_RANKING; exit 1; }
ARGS="--ft $FT --ranking B_patching.json --weights D_weight_change.json --budgets $BUDGETS --seeds $SEEDS --out /work/variants"
python /repo/repair/build_variants_v2.py $ARGS --plan_only 2>&1 | tee out/plan.log
grep -q PLAN_OK out/plan.log || { echo PLAN_FAILED; exit 1; }
cp /work/variants/variants.json out/
NAMES=$(python -c "import json;print(' '.join(json.load(open('/work/variants/variants.json'))['variants']))")
cp /repo/eval/nov05_gate1/gen_base.jsonl out/
cp /repo/eval/nov05_gate1/gen_$FT.jsonl out/gen_ft.jsonl
for v in $NAMES; do
  python /repo/repair/build_variants_v2.py $ARGS --only $v 2>&1 | grep -E "saved|Error|Traceback" | tee -a out/build.log
  grep -q "saved $v " out/build.log || { echo BUILD_FAILED $v; exit 1; }
  case $v in rnd_*) PR=data/eval_prompts_medsafety.jsonl;; *) PR=data/eval_prompts.jsonl;; esac
  python eval/generate.py --name $v --model /work/variants/$v --prompts $PR || { echo GEN_FAILED $v; exit 1; }
  rm -rf /work/variants/$v
  hf upload $REPO out/gen_$v.jsonl $TAG/gen_$v.jsonl --commit-message "$TAG gen $v" >/dev/null 2>&1 || true
done
python eval/judge.py --judges harmbench 2>&1 | tee out/judge.log
grep -q "harmbench scored" out/judge.log || { echo "vLLM judge failed, using transformers fallback"; python eval/judge_hf_fallback.py 2>&1 | tee out/judge_fallback.log; }
test -f out/summary.json || { echo NO_SUMMARY; exit 1; }
rm -f out/gen_base.jsonl out/gen_ft.jsonl
hf upload $REPO out $TAG --commit-message "$TAG judged" || { echo UPLOAD_FAILED; exit 1; }
echo JOB_DONE
