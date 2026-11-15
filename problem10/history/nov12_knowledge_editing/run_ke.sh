#!/bin/bash
# ke_p10 HF Job runner. MODE=tiny (CPU dry run with a tiny random model, canned facts; on success it LAUNCHES the GPU job itself)
# or MODE=gpu (the real run). Code and data come from the private repo $REPO (downloaded at start); every stage uploads its results
# to $REPO/results/$TAG before the next starts, so a crash or timeout loses at most one stage. A failed gate stops the job at once.
# Secrets: HF_TOKEN (downloads Tiny Aya / Qwen, reads+writes $REPO); BILL_TOKEN (tiny mode only: launches the GPU job on that account).
set -uo pipefail
REPO=shreyakommuri/lexsi-ke-p10; MODE=${MODE:-tiny}; TAG=${TAG:-$MODE}; DEST=results/$TAG
T=""; [ "$MODE" = tiny ] && T="--tiny"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU (expected in tiny mode)"
mkdir -p /work/out && cp /repo/code/*.py /repo/code/RULEBOOK.md /work/ && cd /work
echo "== setup"
pip install -q "circuitkit @ https://github.com/Lexsi-Labs/CircuitKIT/archive/refs/tags/0.2.2.tar.gz" "transformers==5.22.1" peft "datasets>=3" \
  accelerate easy-dataset-share protobuf sentencepiece "huggingface_hub[cli]" 2>&1 | tail -2
python -c "import circuitkit, transformers, torch, peft, transformer_lens; print('circuitkit', circuitkit.__version__, 'tf', transformers.__version__, 'torch', torch.__version__, 'peft', peft.__version__, 'tl', transformer_lens.__version__ if hasattr(transformer_lens,'__version__') else '?')" \
  || { echo SETUP_FAILED; exit 1; }
up() { for i in 1 2 3; do hf upload $REPO /work/out $DEST --commit-message "ke_p10 $TAG: $1" > /dev/null 2>&1 && { echo "uploaded after $1"; return 0; }; sleep 15; done; echo "UPLOAD_FAILED $1"; }
stop() { echo "STOPPED at $1"; echo "$1" > /work/out/STOPPED_AT.txt; up "stopped at $1"; exit 3; }
F="grep --line-buffered -v -i -E warn|it/s]|s/it]|Loading|Fetching"
P=/repo/data/eval_prompts.jsonl
VV=python
if [ "$MODE" = gpu ]; then
  echo "== vLLM venv (isolated so vLLM's torch never touches the editing stack)"
  python -m venv /work/vv; VV=/work/vv/bin/python
  for i in 1 2 3; do /work/vv/bin/pip install -q "vllm==0.33.0" > /work/out/vllm_install_$i.log 2>&1 && $VV -c "import vllm" 2>/dev/null && break
    echo "vllm install attempt $i failed:"; tail -25 /work/out/vllm_install_$i.log; df -h /work /tmp | tail -2; sleep 30; done
  $VV -c "import vllm; print('vllm', vllm.__version__)" || stop vllm_setup
fi

echo "== 1. source pairs (training split only)"
python ke.py --stage prep --out /work/out/ke $T 2>&1 | $F | tee /work/out/prep.log; grep -q PREP_DONE /work/out/prep.log || stop prep
echo "== 2. candidate facts (judge model, two passes)"
$VV facts.py --src /work/out/ke/src_pairs.jsonl --out /work/out/ke/facts_raw.jsonl $T 2>&1 | $F | tee /work/out/facts.log
grep -q FACTS_DONE /work/out/facts.log || stop facts; up facts
echo "== 3a. score facts on Normal Aya (own process)"
python ke.py --stage normal --out /work/out/ke --facts /work/out/ke/facts_raw.jsonl $T 2>&1 | grep --line-buffered -v -E "it/s]|s/it]" | tee /work/out/normal.log
grep -q NORMAL_DONE /work/out/normal.log || stop normal
echo "== 3. CircuitKIT MEMIT edits + gates"
python ke.py --stage edit --out /work/out/ke --facts /work/out/ke/facts_raw.jsonl $T 2>&1 | grep --line-buffered -v -E "it/s]|s/it]" | tee /work/out/edit.log
grep -q EDIT_DONE /work/out/edit.log || stop edit; up edit
rm -rf /work/m_broken /work/m_normal
echo "== 4. ruler (705) + MedQA (1,142)"
python eval_ke.py --out /work/out/eval --adapters /work/out/ke/adapters --prompts $P $T 2>&1 | $F | tee /work/out/eval.log
grep -q EVAL_DONE /work/out/eval.log || stop eval; up eval
echo "== 5. 300 held-out answers: broken + 3 arms"
python gen_dose.py --out /work/out/gen --ckpt_root /work/out/ke/adapters --specs circuit=circuit,random=random,block=block $T 2>&1 | $F | tee /work/out/gen.log
grep -q GEN_DONE /work/out/gen.log || stop gen; up gen
if [ "$MODE" = gpu ]; then
  echo "== 6. judge (same Qwen judge + rulebook as every experiment)"
  $VV judge_crft.py --dir /work/out/gen 2>&1 | tee /work/out/judge.log | tr '\r' '\n' | grep -E "^(items|judged|JUDGE_DONE|Traceback)|Error"
  grep -q JUDGE_DONE /work/out/judge.log || stop judge; up judge
fi
echo "== 7. nothing-broke on the circuit arm"
mkdir -p /work/out/nb && python nb.py --out /work/out/nb --prompts $P --models "" --extra crft_ke_circuit=/work/out/ke/adapters/circuit $T 2>&1 | $F | tee /work/out/nb/run.log | tail -15
up nb
echo ALL_STAGES_DONE > /work/out/DONE.txt; up done; echo JOB_DONE

if [ "$MODE" = tiny ] && [ -n "${BILL_TOKEN:-}" ] && [ "${LAUNCH_GPU:-0}" = 1 ]; then
  echo "== dry run passed every stage -> launching the GPU job (billing account of BILL_TOKEN)"
  hf jobs run --token "$BILL_TOKEN" --detach --flavor a100-large --timeout 4h --secrets HF_TOKEN="$HF_TOKEN" \
    -e MODE=gpu -e TAG=gpu \
    pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel \
    bash -c "pip install -q 'huggingface_hub[cli]' && hf download $REPO --exclude 'results/*' --local-dir /repo > /dev/null && bash /repo/code/run_ke.sh" \
    2>&1 | tee /work/out/gpu_launch.txt
  up "gpu job launched"
fi
