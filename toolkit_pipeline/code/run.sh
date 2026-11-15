#!/bin/bash
# toolkit_pipeline overnight job (see ../PREREGISTRATION.md). One A100. All answers stay in the private repo R.
# Stages, each isolated (a failed stage is logged and uploaded; later stages still run):
#  0 prep     323 test / 323 pick pairs and J250 (sha 70731e9b, = main), Turner data. Two-job file structure (no weights copied).
#  1 curate   CuratorKIT 0.1.24 leak gates (J250 vs every Turner question; test vs pick).
#  2 ckB      CircuitKIT selective fine-tune (ck_native_p10 Part B, lr 3e-6 frozen on 10 Nov), answers on J250.
#  3 st       SafeTune C-ΔΘ follow-up arms (safetune_ctheta, sweep reused from 10 Nov), answers on J250.
#  4 ckA      CircuitKIT ActivationSteering (ck_native_p10 Part A, rulers already measured on 10 Nov), answers on J250 + MedQA/XSTest.
#  5 judge    one blind shuffled pass over every J250 answer set (Qwen3-30B-A3B, RULEBOOK 832fad5d, = 14 Nov labels).
#  6 audit    AuditKIT 1.1.7 runs per arm, paired compare + sign test, length table.
set -uo pipefail
REPO=shreyakommuri/aya-bad-medical-r910; DEST=toolkit_pipeline/results; C=/repo/toolkit_pipeline/code; W=/work; O=$W/out
T0=$(date +%s); el() { echo "[$(( ($(date +%s) - T0) / 60 )) min]"; }
mkdir -p $O/{prep,curate,ckA,ckA_gens,ckB,ckB_gens,st,audit,logs} $W/p9root $W/jroot $W/ckpt
up() { for i in 1 2 3; do hf upload $REPO "$1" "$DEST/$2" --commit-message "toolkit_pipeline: $3" > /dev/null 2>&1 && { echo "$(el) uploaded $2"; return 0; }; sleep 20; done; echo "$(el) UPLOAD_FAILED $2"; }
filt() { grep --line-buffered -v -i -E 'warn|it/s]|s/it]|Processed prompts'; }
nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /work | tail -1
echo "$(el) === install"
cp -r /repo/circuitkit_audit/vendor/CircuitKIT_fixed $W/ckpkg
pip install -q -e $W/ckpkg "transformers==5.22.1" "trl>=1.0,<2" "peft>=0.15" "datasets>=3" accelerate easy-dataset-share protobuf sentencepiece scipy colorama pyarrow "huggingface_hub[cli]" > $O/logs/pip_main.log 2>&1; tail -2 $O/logs/pip_main.log
python -c "import circuitkit, transformers, torch, peft, trl; print('circuitkit', circuitkit.__version__, 'tf', transformers.__version__, 'torch', torch.__version__, 'peft', peft.__version__)" || echo MAIN_ENV_FAILED
python -m venv /vkit && /vkit/bin/pip install -q "auditkit==1.1.7" "curatorkit==0.1.24" > $O/logs/pip_kit.log 2>&1; /vkit/bin/python -c "import auditkit, curatorkit; print('auditkit', auditkit.__version__, 'curatorkit', curatorkit.__version__)" || echo KIT_ENV_FAILED
(python -m venv /vv && /vv/bin/pip install -q "vllm==0.33.0" "huggingface_hub[cli]" > $O/logs/pip_vllm.log 2>&1; /vv/bin/python -c "import vllm; print('vllm', vllm.__version__)" > $O/logs/vllm_ok.txt 2>&1) &
VLLM_PID=$!

echo "$(el) === 0 prep"
(cd $C/p9 && LEXSI_ROOT=$W/p9root python -c "import px; px.prep()") 2>&1 | tail -5
export TK_J250=$W/p9root/j250.json
test -f $TK_J250 && test -f $W/p9root/pairs.json || { echo "PREP_FAILED: no J250/pairs, nothing else can run"; up $O/logs logs "prep failed"; exit 1; }
(cd $W && python $C/kit/inspect_twojob.py $O/prep/twojob_structure.json) 2>&1 | tail -2
up $O/prep prep "stage 0 prep"

echo "$(el) === 1 CuratorKIT"
(cd $C/kit && /vkit/bin/python curate.py $W/p9root $O/curate/curate.json) 2>&1 | tail -3
up $O/curate curate "stage 1 CuratorKIT leak gates"

echo "$(el) === 2 CircuitKIT selective fine-tune (Part B)"
mkdir -p $W/ck && cp $C/ck/*.py $W/ck/
(cd $W/ck && timeout 40m python partB.py --out $O/ckB --gens $O/ckB_gens --lr 3e-6) 2>&1 | filt > $O/ckB/partB.log; tail -4 $O/ckB/partB.log
grep -q PARTB_DONE $O/ckB/partB.log && echo "$(el) CKB_OK" || echo "$(el) CKB_FAILED_OR_PARTIAL"
up $O/ckB ckB "stage 2 CircuitKIT fine-tune"; up $O/ckB_gens ckB_gens "stage 2 answers (private)"

echo "$(el) === 3 SafeTune C-ΔΘ follow-up"
mkdir -p $W/st $W/job1 && cp -r $C/st/* $W/st/ && cp /repo/safetune_ctheta/results/ruler_*.json /repo/safetune_ctheta/results/summary_run.json $W/job1/
(cd $W/st && PYTHONPATH=/repo/safetune_ctheta/vendor_safetune/src PUSH_REPO=$REPO PUSH_DEST=$DEST/st timeout 50m python ctheta_run.py --out $O/st \
   --prompts /repo/eval_job/data/eval_prompts.jsonl --inputs $W/st/inputs --ckpt_dir $W/ckpt --resume_dir $W/job1) 2>&1 | filt > $O/st/run.log; tail -4 $O/st/run.log
grep -q RUN_DONE $O/st/run.log && echo "$(el) ST_OK" || echo "$(el) ST_FAILED_OR_PARTIAL"
rm -rf $W/ckpt; up $O/st st "stage 3 SafeTune C-dTheta"

echo "$(el) === 4 CircuitKIT ActivationSteering (Part A)"
(cd $W/ck && TK_SKIP_RULERS=1 timeout 45m python partA.py --out $O/ckA --gens $O/ckA_gens --prompts /repo/eval_job/data/eval_prompts.jsonl \
   --vectors /repo/steering/results/steer_vectors.pt --bs_gen 16) 2>&1 | filt > $O/ckA/partA.log; tail -4 $O/ckA/partA.log
grep -q PARTA_DONE $O/ckA/partA.log && echo "$(el) CKA_OK" || echo "$(el) CKA_FAILED_OR_PARTIAL"
rm -rf $W/m_broken $W/m_normal; up $O/ckA ckA "stage 4 CircuitKIT steering"; up $O/ckA_gens ckA_gens "stage 4 answers (private)"

echo "$(el) === 5 labeller"
wait $VLLM_PID; cat $O/logs/vllm_ok.txt
python $C/kit/collect.py $TK_J250 $W/jroot ckA=$O/ckA_gens ckB=$O/ckB_gens st=$O/st 2>&1 | tail -30
cp $W/p9root/pairs.json $W/jroot/
(cd $C/p9 && LEXSI_ROOT=$W/jroot CUDA_HOME=/usr/local/cuda /vv/bin/python -c "import judge; judge.main('full')") 2>&1 | tr '\r' '\n' | filt | tail -40
test -f $W/jroot/out/judge/full/labels.csv && echo "$(el) JUDGE_OK" || echo "$(el) JUDGE_FAILED"
up $W/jroot/out/judge/full judge "stage 5 labels (judged.jsonl private)"

echo "$(el) === 6 AuditKIT"
(cd $C/kit && /vkit/bin/python audit.py $W/jroot/out/judge/full/labels.csv $O/audit) 2>&1 | tail -50
up $O/audit audit "stage 6 AuditKIT"
up $O/logs logs "logs"
echo "$(el) JOB_DONE"
