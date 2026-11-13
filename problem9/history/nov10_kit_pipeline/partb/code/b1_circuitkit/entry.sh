#!/bin/bash
set -u
cp -r /job /work && cd /work && mkdir -p out up tmp
pip install -q ./CircuitKIT "jinja2>=3.1.6" > pip.log 2>&1 || { echo "pip FAILED"; tail -20 pip.log; exit 1; }
pip list 2>/dev/null | grep -i -E "^(circuitkit|transformer-lens|transformers|torch) "
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SP=$(python3 -c "import circuitkit,os;print(os.path.dirname(os.path.dirname(circuitkit.__file__)))")
mkdir -p ck_patched; cp -r $SP/circuitkit ck_patched/circuitkit; F=ck_patched/circuitkit/backends/eap/attribute_node.py
sed -i '170s/node.in_hook/node.out_hook/' $F; echo "patched lines changed: $(diff $SP/circuitkit/backends/eap/attribute_node.py $F | grep -c '^[<>]')"
export WORK=/work/tmp OUT=/work/out
up() { mkdir -p up/p9_partb/b1 && cp out/*.json up/p9_partb/b1/ 2>/dev/null; for t in 1 2 3 4 5; do hf upload tashom/r910-results up . --repo-type dataset --commit-message "p9 partb b1 $1" > /dev/null 2>&1 && { echo "UPLOADED $1"; return; }; sleep 15; done; echo "UPLOAD FAILED $1"; }
run() { M=$1; V=$2; if [ "$V" = patched ]; then export PYTHONPATH=/work/ck_patched; else unset PYTHONPATH; fi; python3 run_b1.py $M $V all 3 cuda 2>&1 | grep --line-buffered -v -i warn | sed -u "s/^/[$M $V] /"; echo "RUN_EXIT $M $V ${PIPESTATUS[0]}"; up "$M $V"; }
export -f run up
for M in normal bad_s0; do python3 mklink.py $M || { echo "DOWNLOAD FAILED $M"; exit 1; }; done
for M in normal bad_s0; do python3 parity.py /work/tmp/model_$M 2>&1 | grep --line-buffered -E "KL|PARITY|Error" ; [ ${PIPESTATUS[0]} = 0 ] || { echo "PARITY GATE FAILED $M"; exit 1; }; done
GPU_FRACTION=0.95 PYTHONPATH=/work/ck_patched python3 run_b1.py bad_s0 patched 2 3 cuda 2>&1 | grep --line-buffered -E "GATE|DONE|Error|Traceback" | sed -u 's/^/[probe] /'
[ -s out/bad_s0_patched_n2_scores.json ] || { echo "PROBE FAILED, stopping"; exit 1; }
PK=$(python3 -c "import json;print(json.load(open('/work/out/bad_s0_patched_n2_scores.json'))['peak_gb'])"); echo "PROBE_PEAK $PK"
P=$(python3 -c "print(3 if $PK<=23 else 2 if $PK<=34 else 1)"); F=$(python3 -c "print({3:0.32,2:0.47,1:0.95}[$P])"); echo "PARALLEL $P FRACTION $F"; export GPU_FRACTION=$F
printf '%s\n' "bad_s0 patched" "normal patched" "bad_s0 shipped" "normal shipped" | xargs -P $P -L 1 bash -c 'run $0 $1'
for J in "bad_s0 patched" "normal patched" "bad_s0 shipped" "normal shipped"; do set -- $J; if [ ! -s out/$1_$2_nall_scores.json ]; then echo "RERUN $J"; GPU_FRACTION=0.95 run $1 $2; fi; done
up final; echo ALL_B1_DONE
