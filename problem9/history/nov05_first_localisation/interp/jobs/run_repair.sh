#!/bin/bash
# HF Job: Problem 10 — repaired variants of a fine-tuned Aya, built one at a time (disk), generated on the full
# eval set, then judged together with base / ft in a single SafeTune HarmBench judge run.
# Env: FT=bad0|em0  KS=4,8  RAND_K=8  SEEDS=0,1,2,3,4  SKIP="ft,full_revert" (already known)  RANK=localize/results_v2
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910
FT=${FT:-bad0}; KS=${KS:-4,8}; RAND_K=${RAND_K:-8}; SEEDS=${SEEDS:-0,1,2,3,4}; SKIP=${SKIP:-ft,full_revert}
RANK=${RANK:-localize/results_v2}; TAG=${TAG:-repair_${FT}}
nvidia-smi --query-gpu=name,memory.total --format=csv; df -h / /work 2>/dev/null | tail -2
mkdir -p /work/out /work/variants && cd /work && cp -r /repo/eval_job/eval /repo/eval_job/data .
cp -r /repo/vendor/SafeTune st && ln -s /work/st/src eval/safetune_src || exit 1
bash /repo/eval_job/setup_safetune.sh || exit 1
pip install -q vllm 2>&1 | grep -i "^error" | tail -1
bash /repo/eval_job/setup_safetune.sh || exit 1
pip install -q "huggingface_hub[cli]" 2>&1 | tail -1
python -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)"; python -c "import vllm;print('vllm',vllm.__version__)" || true

cp /repo/$RANK/B_patching.json /repo/localize/results/D_weight_change.json /work/ || { echo NO_RANKING; exit 1; }
ARGS="--ft $FT --ranking B_patching.json --weights D_weight_change.json --ks $KS --rand_k $RAND_K --seeds $SEEDS --out /work/variants"
python /repo/repair/build_variants.py $ARGS --plan_only 2>&1 | tee out/plan.log
grep -q PLAN_OK out/plan.log || { echo PLAN_FAILED; exit 1; }
cp /work/variants/variants.json out/
NAMES=$(python -c "import json;print(' '.join(json.load(open('/work/variants/variants.json'))['variants']))")
echo "variants: $NAMES"

# references from gate 1, same prompts and settings: base, and the unchanged fine-tune
cp /repo/eval/nov05_gate1/gen_base.jsonl out/
cp /repo/eval/nov05_gate1/gen_$FT.jsonl out/gen_ft.jsonl
for v in $NAMES; do
  case ",$SKIP," in *",$v,"*) echo "skip $v"; continue;; esac
  python /repo/repair/build_variants.py $ARGS --only $v 2>&1 | tail -3 | tee -a out/build.log
  grep -q "saved $v" out/build.log || { echo BUILD_FAILED $v; exit 1; }
  python eval/generate.py --name $v --model /work/variants/$v || { echo GEN_FAILED $v; exit 1; }
  rm -rf /work/variants/$v
  hf upload $REPO out/gen_$v.jsonl $TAG/gen_$v.jsonl --commit-message "$TAG gen $v" >/dev/null 2>&1 || true
done
python eval/judge.py --judges harmbench 2>&1 | tee out/judge.log
grep -q "harmbench scored" out/judge.log || { echo "vLLM judge failed, using transformers fallback"; python eval/judge_hf_fallback.py 2>&1 | tee out/judge_fallback.log; }
test -f out/summary.json || { echo NO_SUMMARY; exit 1; }
rm -f out/gen_base.jsonl out/gen_ft.jsonl
hf upload $REPO out $TAG --commit-message "$TAG judged" || { echo UPLOAD_FAILED; exit 1; }
echo JOB_DONE
