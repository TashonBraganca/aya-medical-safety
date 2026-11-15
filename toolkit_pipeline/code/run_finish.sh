#!/bin/bash
# toolkit_pipeline FINISH run (15 Nov 2026), one H200. Rules: ../PREREG_FINISH.md. All answers stay in the private repo R.
#  0 install + prep   pairs/J250 (same sha as every run), nothing-broke data, two-job files (Tahsoom/aya-twojob-addons), resume files from Tashon's rows
#  1 curate           CuratorKIT gate: PubMedQA test vs the two-job add-on's PubMedQA training and skill rows (F5)
#  2 parallel         F1+F2 E3 seeds 0/1/2 (answers at 20% + nothing-broke), F3 Safe LoRA (SafeTune), F2 problem 9 cut nothing-broke, then F4 MedQA/XSTest steering
#  3 judge            one blind shuffled pass over every answer set of this run (Qwen3-30B-A3B, RULEBOOK 832fad5d)
#  4 audit            AuditKIT 1.1.7 on the pre-registered pairs (kit/pairs_finish.json) + reproduction checks
set -uo pipefail
PREFLIGHT=${PREFLIGHT:-0}
REPO=shreyakommuri/aya-bad-medical-r910; DEST=toolkit_pipeline/results_finish; [ "$PREFLIGHT" = 1 ] && DEST=toolkit_pipeline/results_finish_preflight
C=/repo/toolkit_pipeline/code; W=/work; O=$W/out; TR=$W/two; E3O=$W/e3out
T0=$(date +%s); el() { echo "[$(( ($(date +%s) - T0) / 60 )) min]"; }
mkdir -p $O/{prep,curate,e3,e2j,sl,ckA,audit,logs} $W/p9root $W/jroot $E3O/e3/full
up() { for i in 1 2 3; do hf upload $REPO "$1" "$DEST/$2" --commit-message "toolkit_pipeline finish: $3" > /dev/null 2>&1 && { echo "$(el) uploaded $2"; return 0; }; sleep 20; done; echo "$(el) UPLOAD_FAILED $2"; }
filt() { grep --line-buffered -v -i -E 'warn|it/s]|s/it]|Processed prompts'; }
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU (preflight)"; df -h /work | tail -1; free -g | head -2

echo "$(el) === install"
cp -r /repo/circuitkit_audit/vendor/CircuitKIT_fixed $W/ckpkg
pip install -q -e $W/ckpkg "transformers==5.22.1" "trl>=1.0,<2" "peft>=0.15" "datasets>=3" accelerate easy-dataset-share protobuf sentencepiece scipy colorama pyarrow "huggingface_hub[cli]" > $O/logs/pip_main.log 2>&1; tail -2 $O/logs/pip_main.log
python -c "import circuitkit, transformers, torch, peft, trl; print('circuitkit', circuitkit.__version__, 'tf', transformers.__version__, 'torch', torch.__version__, 'peft', peft.__version__)" || echo MAIN_ENV_FAILED
PYTHONPATH=/repo/safetune_ctheta/vendor_safetune/src python -c "from safetune.interventions.recover.safe_lora import apply_safe_lora; import safetune; print('safetune import ok', getattr(safetune, '__version__', '?'))" || echo SAFETUNE_IMPORT_FAILED
python -m venv /vkit && /vkit/bin/pip install -q "auditkit==1.1.7" "curatorkit==0.1.24" > $O/logs/pip_kit.log 2>&1; /vkit/bin/python -c "import auditkit, curatorkit; print('auditkit', auditkit.__version__, 'curatorkit', curatorkit.__version__)" || echo KIT_ENV_FAILED
if [ "$PREFLIGHT" != 1 ]; then
  (python -m venv /vv && /vv/bin/pip install -q "vllm==0.33.0" "huggingface_hub[cli]" > $O/logs/pip_vllm.log 2>&1; /vv/bin/python -c "import vllm; print('vllm', vllm.__version__)" > $O/logs/vllm_ok.txt 2>&1) &
  VLLM_PID=$!
fi

echo "$(el) === 0 prep"
(cd $C/p9 && LEXSI_ROOT=$W/p9root python -c "import px; px.prep()") 2>&1 | tail -3
export TK_J250=$W/p9root/j250.json
test -f $TK_J250 && test -f $W/p9root/pairs.json || { echo "PREP_FAILED: no J250/pairs"; up $O/logs logs "prep failed"; exit 1; }
mkdir -p $TR && cp -r $C/two/* $TR/ && cp $W/p9root/pairs.json $W/p9root/j250.json $TR/
(cd $TR && LEXSI_ROOT=$W/p9root python -c "import nbsuite; nbsuite.build('$TR')") 2>&1 | tail -2
hf download Tahsoom/aya-twojob-addons --local-dir $W/tj > /dev/null 2>&1; ls -la $W/tj | grep pt
(cd $W/tj && sha256sum twojob_seed*.pt) | tee $O/prep/twojob_sha256.txt
cp $W/tj/twojob_seed1.pt $W/tj/twojob_seed2.pt $E3O/e3/full/
for s in 0 1 2; do python $TR/seed_resume.py $s $E3O/e3/full; done
ls -la $TR/*.json $TR/*.txt $TR/*.csv | awk '{print $5, $9}'
up $O/prep prep "stage 0 prep"

echo "$(el) === 1 CuratorKIT PubMedQA gate"
(cd $TR && LEXSI_ROOT=$W/p9root python pqa_lists.py $W/pqa_lists.json) 2>&1 | tail -2
(cd $C/kit && /vkit/bin/python curate_pqa.py $W/pqa_lists.json $O/curate/curate_pqa.json) 2>&1 | tail -3
up $O/curate curate "stage 1 CuratorKIT PubMedQA gate"
if [ "$PREFLIGHT" = 1 ]; then up $O/logs logs "preflight logs"; echo "$(el) PREFLIGHT_DONE"; exit 0; fi

echo "$(el) === 2 parallel: E3 x3, Safe LoRA, problem 9 cut (then CircuitKIT MedQA/XSTest)"
GEN=e2c_fp_net_20,biggest_first_20,true_random_20_d0,true_random_20_d1,true_random_20_d2
export LEXSI_ROOT=$W/p9root E3_ROOT=$TR E3_OUT=$E3O
(cd $TR && MEMFRAC=0.14 E3_PQA_BS=8 TWOJOB0=$W/tj/twojob_seed0.pt E3_FRS=0.2 E3_NOSPLIT=1 E3_NB=1 E3_GEN=$GEN timeout 170m python e3_twojob_v2_nb.py full 0) 2>&1 | filt > $O/e3/seed0.log &
P0=$!
(cd $TR && MEMFRAC=0.14 E3_FAST=1 E3_FRS=0.2 E3_NOSPLIT=1 E3_NB=1 E3_GEN=$GEN timeout 170m python e3_twojob_v2_nb.py full 1) 2>&1 | filt > $O/e3/seed1.log &
P1=$!
(cd $TR && MEMFRAC=0.14 E3_FAST=1 E3_FRS=0.2 E3_NOSPLIT=1 E3_NB=1 E3_GEN=$GEN timeout 170m python e3_twojob_v2_nb.py full 2) 2>&1 | filt > $O/e3/seed2.log &
P2=$!
(cd $C/sl && PYTHONPATH=$TR:/repo/safetune_ctheta/vendor_safetune/src MEMFRAC=0.2 timeout 170m python safelora_run.py full $O/sl $W/tj) 2>&1 | filt > $O/sl/run.log &
P3=$!
mkdir -p $W/ck && cp $C/ck/*.py $W/ck/
cka() { (cd $W/ck && TK_SKIP_RULERS=1 TK_SKIP_GENS=1 TK_TL_EVAL_BS=2 TK_VEC_REF=/repo/toolkit_pipeline/results_ckA_rerun/ckA/vectors_partA.pt timeout 90m python partA.py --out $O/ckA --gens $W/ckA_unused \
   --prompts /repo/eval_job/data/eval_prompts.jsonl --vectors /repo/steering/results/steer_vectors.pt --bs_gen 8) 2>&1 | filt > $O/ckA/partA$1.log; }
( (cd $TR && MEMFRAC=0.14 E2_TARGET=bad_s2 E2_FRAC=0.20 E2J_NB=1 E2J_GENS=none timeout 60m python e2j_nb.py full) 2>&1 | filt > $O/e2j/run.log; cka "" ) &
P4=$!
( while sleep 600; do echo "$(el) --- heartbeat"; for f in $O/e3/seed0.log $O/e3/seed1.log $O/e3/seed2.log $O/sl/run.log $O/e2j/run.log $O/ckA/partA.log; do [ -f $f ] && echo "  $(basename $(dirname $f))/$(basename $f): $(grep -E 'HEARTBEAT|Error|error|DONE|gate' $f | tail -1 | cut -c1-220)"; done; nvidia-smi --query-gpu=memory.used --format=csv,noheader; done ) &
HB=$!
wait $P0 $P1 $P2 $P3 $P4; kill $HB 2>/dev/null
for f in seed0 seed1 seed2; do echo "$f: $(tail -2 $O/e3/$f.log | tr '\n' ' ' | cut -c1-300)"; done
echo "sl: $(tail -2 $O/sl/run.log | tr '\n' ' ' | cut -c1-300)"; echo "e2j: $(tail -2 $O/e2j/run.log | tr '\n' ' ' | cut -c1-300)"
if ! grep -q PARTA_DONE $O/ckA/partA.log 2>/dev/null; then echo "$(el) CKA first try failed: $(grep -i -E 'error|out of memory' $O/ckA/partA.log | tail -2 | cut -c1-300); retry alone"; cka _retry; fi
grep -q PARTA_DONE $O/ckA/partA*.log && echo "$(el) CKA_OK" || echo "$(el) CKA_FAILED"
rm -rf $W/ck/m_broken $W/ck/m_normal /work/m_broken /work/m_normal
mkdir -p $O/e3/full $O/e3/gen_private $O/e2j/full
cp $E3O/e3/full/*.json $O/e3/full/ 2>/dev/null; cp $E3O/e3/full/gen/*.jsonl $O/e3/gen_private/ 2>/dev/null; cp $E3O/e2j_bad_s2/full/*.json $O/e2j/full/ 2>/dev/null
python $TR/check_repro.py $E3O/e3/full 0 1 2 2>&1 | tee $O/e3/repro.txt
python - <<EOF 2>&1 | tee $O/e2j/repro.txt
import json
old = json.load(open("$TR/inputs/e2_s2_tashon.json"))["arms"]; new = json.load(open("$E3O/e2j_bad_s2/full/e2j.json"))["arms"]
for a, b in (("e2c_fp_blame", "e2c_fp_blame_20"), ("arrow_blame", "arrow_blame_20"), ("biggest_first", "biggest_first_20"), ("true_random_d0", "true_random_20_d0")):
    if a in new and b in old: print(f"{a:16s} lean new {new[a]['lean']:.2f} vs Tashon E2 seed 2 {old[b]['lean']:.2f} -> {'OK' if abs(new[a]['lean'] - old[b]['lean']) < 1.5 else 'DIFFERS'}")
EOF
up $O/e3 e3 "stage 2 E3 rows, nothing-broke, answers (private)"; up $O/e2j e2j "stage 2 problem 9 cut nothing-broke"; up $O/sl sl "stage 2 Safe LoRA (answers private)"; up $O/ckA ckA "stage 2 CircuitKIT MedQA/XSTest"

echo "$(el) === 3 labeller"
wait $VLLM_PID; cat $O/logs/vllm_ok.txt
python $C/kit/collect.py $TK_J250 $W/jroot two=$E3O/e3/full/gen sl=$O/sl/gen 2>&1 | tail -30
cp $W/p9root/pairs.json $W/jroot/
(cd $C/p9 && LEXSI_ROOT=$W/jroot CUDA_HOME=/usr/local/cuda /vv/bin/python -c "import judge; judge.main('full')") 2>&1 | tr '\r' '\n' | filt | tail -30
test -f $W/jroot/out/judge/full/labels.csv && echo "$(el) JUDGE_OK" || echo "$(el) JUDGE_FAILED"
up $W/jroot/out/judge/full judge "stage 3 labels (judged.jsonl private)"

echo "$(el) === 4 AuditKIT"
(cd $C/kit && /vkit/bin/python audit.py $W/jroot/out/judge/full/labels.csv $O/audit $C/kit/pairs_finish.json) 2>&1 | tail -45
up $O/audit audit "stage 4 AuditKIT"
up $O/logs logs "logs"
echo "$(el) JOB_DONE"
