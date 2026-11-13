#!/bin/bash
# HF Job, ck_native_p10: Part A (CircuitKIT ActivationSteering) -> Part B (CircuitKIT selective fine-tune) -> nothing-broke on the
# circuit step-25 model -> one local judge pass over all answers -> analysis. Each stage uploads its own results to R before the next.
# Env: SKIP_A=1 / SKIP_B=1 / SKIP_NB=1 / SKIP_J=1 to skip stages; LR=<float> to skip Part B's calibration (only after one ran).
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; DEST=ck_native_p10/results
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p /work/out/partA /work/out/partB /work/out/gens /work/out/nb
cp /repo/ck_native_p10/code/*.py /repo/ck_native_p10/code/RULEBOOK.md /work/ && cp -r /repo/circuitkit_audit/vendor/CircuitKIT_fixed /work/ck && cd /work
pip install -q -e /work/ck "transformers==5.22.1" peft "datasets>=3" accelerate easy-dataset-share protobuf sentencepiece "huggingface_hub[cli]" 2>&1 | tail -2
python -c "import circuitkit, transformers, torch, peft; print('circuitkit', circuitkit.__version__, circuitkit.__file__, 'tf', transformers.__version__, 'torch', torch.__version__, 'peft', peft.__version__)" || { echo SETUP_FAILED; exit 1; }
up() { hf upload $REPO /work/out/$1 $DEST/$1 --commit-message "ck_native_p10: $2" > /dev/null 2>&1 && echo "uploaded $1" || echo "UPLOAD_FAILED $1"; }
F="grep --line-buffered -v -i -E warn|it/s]|s/it]"
P=/repo/eval_job/data/eval_prompts.jsonl
if [ "${SKIP_A:-0}" != 1 ]; then
  python partA.py --out /work/out/partA --gens /work/out/gens --prompts $P --vectors /repo/steering/results/steer_vectors.pt 2>&1 | $F | tee /work/out/partA/partA.log
  grep -q PARTA_DONE /work/out/partA/partA.log && echo PARTA_OK || echo PARTA_FAILED
  up partA "Part A (ActivationSteering)"; up gens "answers after Part A"
fi
if [ "${SKIP_B:-0}" != 1 ]; then
  python partB.py --out /work/out/partB --gens /work/out/gens ${LR:+--lr $LR} 2>&1 | $F | tee /work/out/partB/partB.log
  grep -q PARTB_DONE /work/out/partB/partB.log && echo PARTB_OK || echo PARTB_FAILED
  up partB "Part B (selective finetune)"; up gens "answers after Part B"
fi
if [ "${SKIP_NB:-0}" != 1 ] && [ -f /work/out/partB/delta_circuit_s25.pt ]; then
  python nb_ck.py --out /work/out/nb --prompts $P --models broken --delta ckft_circuit25=/work/out/partB/delta_circuit_s25.pt 2>&1 | $F | tee /work/out/nb/run.log
  up nb "nothing-broke on circuit step 25"
fi
if [ "${SKIP_J:-0}" != 1 ]; then
  pip install -q vllm 2>&1 | tail -2
  pip install -q "transformers==5.22.1" 2>&1 | tail -1
  python judge_ck.py --dir /work/out/gens 2>&1 | tee /work/out/gens/judge.log | tr '\r' '\n' | grep -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
  up gens "judged"
fi
python analyze.py --root /work/out 2>&1 | tee /work/out/RESULTS_auto.md
hf upload $REPO /work/out/RESULTS_auto.md $DEST/RESULTS_auto.md --commit-message "ck_native_p10: analysis" > /dev/null 2>&1; hf upload $REPO /work/out/summary.json $DEST/summary.json > /dev/null 2>&1
echo JOB_DONE
