#!/bin/bash
# wso job runner (§50). Env: JOB (w1|w2|w3core|w3dial|w4|final|post|judge), LR, TSTEPS, INIT, ARMS, CAP (minutes), TAG, plus for post: SPECS, WIN
#  w1..w4/final : python wso.py --job $JOB ...  -> R:wso/results/<job>[_TAG]
#  post         : nothing-broke + 300 answers + relapse 25/50 for adapters in R:wso/results/final/<name>/adapter (WIN = space-separated names)
#  judge        : judge every gen_*.jsonl collected from R:wso/results/post_* (+ §49 broken/P answers) in one pass
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; RES=wso/results; P=/repo/eval_job/data/eval_prompts.jsonl
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cp /repo/wso/code/*.py /repo/wso/code/RULEBOOK.md /repo/wso/code/requirements.txt /work/ && cd /work
pip install -q -r requirements.txt 2>&1 | tail -2
python -c "import torch,transformers,peft;print('torch',torch.__version__,'tf',transformers.__version__,'peft',peft.__version__)" || { echo SETUP_FAILED; exit 1; }
up() { for i in 1 2 3; do hf upload $REPO "$1" "$2" --commit-message "$3" && return 0; sleep 10; done; echo "UPLOAD FAILED $2"; return 1; }
G="tr '\r' '\n' | grep --line-buffered -E '^\[|Traceback|Error|error:'"
N=$JOB${TAG:+_$TAG}
FROOT=/repo/$RES/final
if [ "$JOB" = finalpost ]; then JOB=final; THEN_POST=1; N=final; fi    # final export (-> R:wso/results/final), then post in the same container
# NOTE: the Nov 10 finalpost run (job 6af3579a…) predates "N=final" and uploaded the export to R:wso/results/finalpost_core instead
case $JOB in
w1|w2|w3core|w3dial|w4|final)
  python wso.py --job $JOB --out /work/out --prompts $P ${LR:+--lr $LR} ${TSTEPS:+--T $TSTEPS} ${INIT:+--init "$INIT"} ${ARMS:+--arms $ARMS} ${CAP:+--time_cap_min $CAP} 2>&1 \
    | tee /work/out/log_$N.txt | eval $G
  cp /work/out/log_$N.txt /work/out/$JOB/; grep -q WSO_DONE /work/out/log_$N.txt && M="wso $N done" || M="wso $N FAILED/partial"
  up /work/out/$JOB $RES/$N "$M"
  if [ "${THEN_POST:-0}" = 1 ]; then JOB=post; N=post; FROOT=/work/out/final; fi ;;
esac
case $JOB in
post)
  SP=""; EX=""
  for W in $WIN; do SP="$SP${SP:+,}$W=$W/adapter"; EX="$EX${EX:+,}crft_$W=$REPO:$RES/final/$W/adapter"; done
  python gen_dose.py --skip_broken --out /work/out/post/gen --ckpt_root $FROOT --specs "$SP" 2>&1 | tee /work/out/gen.log | eval $G
  up /work/out/post $RES/post_$TAG "wso post gen $TAG"
  mkdir -p /work/out/post/nb && python nb.py --out /work/out/post/nb --prompts $P --models broken --extra "$EX" 2>&1 | tee /work/out/post/nb/run.log | tail -20
  up /work/out/post $RES/post_$TAG "wso post nb $TAG"
  RG=${RGEN:-50}; [ "$RG" = none ] && RG=""
  for W in $WIN; do
    python relapse_ld.py --start adapter --adapter $FROOT/$W/adapter --name relapse_$W --out /work/out/post --ckpt_steps 25,50,100,200,397 \
      --gen_steps "$RG" --stop_at 50 2>&1 | tee /work/out/post/relapse_$W.log | eval $G
    rm -rf /work/out/post/relapse_$W/ckpt /work/out/post/relapse_$W/adapters
    up /work/out/post $RES/post_$TAG "wso post relapse $W"
  done ;;
nb)   # nothing-broke only, for adapters at R:$NBROOT/<W>/adapter (broken's suite already done in post)
  EX=""; for W in $WIN; do EX="$EX${EX:+,}crft_$W=$REPO:${NBROOT:-$RES/final}/$W/adapter"; done
  mkdir -p /work/out/nb && python nb.py --out /work/out/nb --prompts $P --models "" --extra "$EX" 2>&1 | tee /work/out/nb/run.log | tail -20
  up /work/out/nb $RES/nb_$TAG "wso nothing-broke $TAG" ;;
judge)
  J=/work/out/judge; mkdir -p $J
  cp /repo/relapse/results/judge/gen_questions.json /repo/relapse/results/judge/reference_answers.json $J/
  cp /repo/relapse/results/judge/gen_broken_s0.jsonl $J/
  for f in /repo/lean_down/results/gen_P_circuit/gen_P_circuit.jsonl /repo/lean_down/results/relapse_P_circuit/gen_*.jsonl \
           /repo/$RES/post_*/gen/gen_*.jsonl /repo/$RES/post_*/relapse_*/gen_*.jsonl; do [ -f "$f" ] && cp "$f" $J/; done
  ls $J; cmp <(python -c "import json;print([q['id'] for q in json.load(open('$J/gen_questions.json'))])") \
             <(python -c "import json,glob;print([q['id'] for q in json.load(open(sorted(glob.glob('/repo/$RES/post_*/gen/gen_questions.json'))[0]))])") && echo SAME_QUESTIONS || echo QUESTION_MISMATCH
  pip install -q vllm 2>&1 | tail -2; pip install -q "transformers==5.22.1" 2>&1 | tail -1
  python judge_crft.py --dir $J 2>&1 | tee $J/judge.log | tr '\r' '\n' | grep -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
  up $J $RES/judge_$TAG "wso judge $TAG" ;;
esac
echo JOB_DONE
