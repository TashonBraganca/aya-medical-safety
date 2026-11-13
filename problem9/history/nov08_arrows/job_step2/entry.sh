#!/bin/bash
# MODE: smoke | gen | judge_train | judge_gen
set -u
MODE=${MODE:-smoke}
JUDGES=${JUDGES:-all}
TAG=${TAG:-}
cp -r /job /work && cd /work && mkdir -p out judged reviewed
pip install -q peft accelerate datasets "huggingface_hub>=1.0" > pip.log 2>&1 || { echo "pip FAILED"; tail -30 pip.log; }
python3 - > out/versions.txt <<'PY'
import importlib, torch
for m in ["vllm","torch","transformers","peft","huggingface_hub"]:
    print(m, importlib.import_module(m).__version__)
import subprocess; print("nvcc", subprocess.run(["nvcc","--version"],capture_output=True,text=True).stdout.strip().splitlines()[-1])
print("gpu", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30), "GiB")
PY
cat out/versions.txt
up() {  # upload everything produced so far, so a cancelled job never loses finished stages
  mkdir -p up/p9_step2/$MODE$TAG && cp -r out judged reviewed up/p9_step2/$MODE$TAG/ 2>/dev/null
  find up -name "*.safetensors" -delete
  hf upload tashom/r910-results up . --repo-type dataset --commit-message "p9 step2 $MODE$TAG ($1)" > /dev/null 2>&1 && echo "UPLOADED stage $1"
}
if [ "$MODE" = smoke ]; then
  python3 gen.py --models normal,good_v2,bad_s0 --limit 6 --parallel 3 --gpu_util 0.25 --out out 2>&1 | tee out/gen_driver.log
  python3 judge.py --inputs "out/gen_*.jsonl" data/train_answers.jsonl --judges all --limit 6 --parallel 3 --out judged 2>&1 | tee judged/judge_driver.log
elif [ "$MODE" = gen ]; then
  python3 gen.py --models normal,good_v2,bad_s0,bad_s1,bad_s2,bad_s0_owntok --parallel 6 --gpu_util 0.15 --out out 2>&1 | tee out/gen_driver.log
elif [ "$MODE" = judge_train ]; then
  python3 judge.py --inputs data/train_answers.jsonl --judges $JUDGES --parallel 3 --out judged 2>&1 | tee judged/judge_driver.log
elif [ "$MODE" = judge_gen ]; then
  JSRC=${JSRC:-gen}
  hf download tashom/r910-results --repo-type dataset --include "p9_step2/$JSRC/out/gen_*.jsonl" --local-dir dl > /dev/null && cp dl/p9_step2/$JSRC/out/gen_*.jsonl out/
  ls out/gen_*.jsonl
  python3 judge.py --inputs "out/gen_*.jsonl" --judges $JUDGES --parallel 3 --out judged 2>&1 | tee judged/judge_driver.log
elif [ "$MODE" = owntok_all ]; then
  # 4 more models under the training tokenizer, then both reviewers and all judges on their answers, one H200
  python3 gen.py --models normal_owntok,good_v2_owntok,bad_s1_owntok,bad_s2_owntok --parallel 4 --gpu_util 0.2 --skip_merge_check --out out 2>&1 | tee out/gen_driver.log
  for R in rev_a rev_b_cot; do python3 reviewer.py --reviewer $R --inputs out/gen_*.jsonl --out reviewed 2>&1 | tee -a reviewed/reviewed_driver.log; done
  python3 judge.py --inputs "out/gen_*.jsonl" --judges all --parallel 3 --out judged 2>&1 | tee judged/judge_driver.log
elif [ "$MODE" = owntok_redo ]; then
  python3 gen.py --models normal_owntok,good_v2_owntok,bad_s1_owntok,bad_s2_owntok --parallel 4 --gpu_util 0.2 --skip_merge_check --out out 2>&1 | tee out/gen_driver.log; up gen
  for R in rev_a rev_b_cot; do python3 reviewer.py --reviewer $R --inputs out/gen_*.jsonl --out reviewed 2>&1 | tee -a reviewed/reviewed_driver.log; up $R; done
elif [ "$MODE" = tiebreak ]; then
  hf download tashom/r910-results --repo-type dataset --include "p9_step2/gen/out/gen_*.jsonl" --local-dir dl > /dev/null && cp dl/p9_step2/gen/out/gen_*.jsonl out/
  python3 reviewer.py --reviewer rev_c --inputs out/gen_*.jsonl data/train_answers.jsonl --sample data/tiebreak_ids.json --out reviewed 2>&1 | tee reviewed/reviewed_driver.log; up rev_c
elif [ "$MODE" = owntok_normal ]; then
  python3 gen.py --models normal_owntok --parallel 1 --gpu_util 0.5 --skip_merge_check --out out 2>&1 | tee out/gen_driver.log
  for R in rev_a rev_b_cot; do python3 reviewer.py --reviewer $R --inputs out/gen_*.jsonl --out reviewed 2>&1 | tee -a reviewed/reviewed_driver.log; done
  python3 judge.py --inputs "out/gen_*.jsonl" --judges all --parallel 3 --out judged 2>&1 | tee judged/judge_driver.log
elif [ "$MODE" = review ]; then
  # SRC=smoke: trial on the smoke answers + first 10 training pairs; SRC=gen: every generated answer + all training answers
  SRC=${SRC:-smoke}
  if [ "$SRC" = trial ]; then
    head -20 data/train_answers.jsonl > out/train_trial.jsonl; GEN=""; IN="out/train_trial.jsonl"
  else
    hf download tashom/r910-results --repo-type dataset --include "p9_step2/$SRC/out/gen_*.jsonl" --local-dir dl > /dev/null && cp dl/p9_step2/$SRC/out/gen_*.jsonl out/
    GEN=$(ls out/gen_*.jsonl); IN="data/train_answers.jsonl"
  fi
  ls -la out/
  for R in ${REVIEWER//,/ }; do
    S=""; if [ "$SRC" = gen ] && [ "${GOLD_ONLY:-}" = "$R" ]; then S="--sample data/gold_ids.json"; fi
    python3 reviewer.py --reviewer $R --inputs $GEN $IN $S --out reviewed 2>&1 | tee -a reviewed_driver.log
  done
  mkdir -p reviewed && cp reviewed_driver.log reviewed/
fi
mkdir -p up/p9_step2/$MODE$TAG && cp -r out judged reviewed up/p9_step2/$MODE$TAG/ 2>/dev/null
find up -name "*.safetensors" -delete
hf upload tashom/r910-results up . --repo-type dataset --commit-message "p9 step2 $MODE$TAG" && echo UPLOADED
