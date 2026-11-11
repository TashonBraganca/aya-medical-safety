#!/bin/bash
# lean_down follow-up jobs (§49 job 2). Env: MODE, WIN (winner arm name without floors, e.g. P), TAG
#  MODE=ctrl   : train WIN on random floors s2 and block 19-28 (lean.py), then 300 answers for W_s2 -> R:lean_down/results/{W_s2,W_block,gen_W_s2}
#  MODE=gen_nb : 300 answers for W_circuit -> R:lean_down/results/gen_W_circuit ; nothing-broke (broken + fix) -> R:lean_down/results/nb_W_circuit
#  MODE=relapse: §45 relapse recipe on broken+W_circuit, stop at step 50, answers at 50 -> R:lean_down/results/relapse_W_circuit
#  MODE=judge  : judge every gen_*.jsonl collected from R (new + §45 references) in ONE pass -> R:lean_down/results/judge_$TAG
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; RES=lean_down/results; P=/repo/eval_job/data/eval_prompts.jsonl
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out && cp /repo/lean_down/code/*.py /repo/lean_down/code/*.json /repo/lean_down/code/RULEBOOK.md /repo/lean_down/code/requirements.txt /work/ && cd /work
pip install -q -r requirements.txt 2>&1 | tail -2
python -c "import torch,transformers,peft;print('torch',torch.__version__,'tf',transformers.__version__,'peft',peft.__version__)" || { echo SETUP_FAILED; exit 1; }
up() { for i in 1 2 3; do hf upload $REPO "$1" "$2" --commit-message "$3" && return 0; sleep 10; done; echo "UPLOAD FAILED $2"; return 1; }
G="tr '\r' '\n' | grep --line-buffered -E '^\[|Traceback|Error|error:|^SAVED'"
case $MODE in
ctrl)
  for FL in s2 block; do
    N=${WIN}_$FL; echo "=== $N $(date +%T)"
    python lean.py --variant $WIN --floors $FL --out /work/out --prompts $P 2>&1 | tee /work/out/log_$N.txt | eval $G
    mkdir -p /work/out/$N; cp /work/out/log_$N.txt /work/out/$N/; up /work/out/$N $RES/$N "lean_down ctrl $N"
  done
  python gen_dose.py --skip_broken --out /work/out/gen_${WIN}_s2 --ckpt_root /work/out --specs "${WIN}_s2=${WIN}_s2/adapter_s150" 2>&1 | tee /work/out/gen.log | eval $G
  up /work/out/gen_${WIN}_s2 $RES/gen_${WIN}_s2 "lean_down gen ${WIN}_s2" ;;
gen_nb)
  python gen_dose.py --skip_broken --out /work/out/gen_${WIN}_circuit --ckpt_root /repo/$RES --specs "${WIN}_circuit=${WIN}_circuit/adapter_s150${EXTRA_SPECS:+,$EXTRA_SPECS}" 2>&1 | tee /work/out/gen.log | eval $G
  up /work/out/gen_${WIN}_circuit $RES/gen_${WIN}_circuit "lean_down gen ${WIN}_circuit"
  mkdir -p /work/out/nb && python nb.py --out /work/out/nb --prompts $P --models broken --extra "crft_${WIN}=$REPO:$RES/${WIN}_circuit/adapter_s150" 2>&1 | tee /work/out/nb/run.log | tail -40
  up /work/out/nb $RES/nb_${WIN}_circuit "lean_down nothing-broke ${WIN}_circuit" ;;
relapse)
  python relapse_ld.py --start adapter --adapter /repo/$RES/${WIN}_circuit/adapter_s150 --name relapse_${WIN}_circuit --out /work/out --gen_steps 50 --stop_at 50 2>&1 \
    | tee /work/out/relapse.log | eval $G
  rm -rf /work/out/relapse_${WIN}_circuit/ckpt; cp /work/out/relapse.log /work/out/relapse_${WIN}_circuit/
  up /work/out/relapse_${WIN}_circuit $RES/relapse_${WIN}_circuit "lean_down relapse ${WIN}_circuit" ;;
judge)
  J=/work/out/judge; mkdir -p $J
  cp /repo/relapse/results/judge/gen_questions.json /repo/relapse/results/judge/reference_answers.json $J/
  for f in broken_s0 arrowfix_s0 circuitft_s0 normal_s0 arrowfix_s50 circuitft_s50 normal_s50; do cp /repo/relapse/results/judge/gen_$f.jsonl $J/; done
  for f in /repo/$RES/gen_*/gen_*.jsonl /repo/$RES/relapse_*/gen_*.jsonl; do [ -f "$f" ] && cp "$f" $J/; done
  ls $J; cmp <(python -c "import json;print([q['id'] for q in json.load(open('$J/gen_questions.json'))])") \
             <(python -c "import json,glob;print([q['id'] for q in json.load(open(sorted(glob.glob('/repo/$RES/gen_*/gen_questions.json'))[0]))])") && echo SAME_QUESTIONS || echo QUESTION_MISMATCH
  pip install -q vllm 2>&1 | tail -2; pip install -q "transformers==5.22.1" 2>&1 | tail -1
  python judge_crft.py --dir $J 2>&1 | tee $J/judge.log | tr '\r' '\n' | grep -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
  up $J $RES/judge_$TAG "lean_down judge $TAG" ;;
esac
echo JOB_DONE
