#!/bin/bash
W=/workspace/p9_partb; cd $W/b1
$W/ck/bin/python -m pip install -q -U --no-cache-dir "jinja2>=3.1.6" > jinja.log 2>&1; echo "jinja exit $?" >> smoke2.log
SP=$W/ck/lib/python3.11/site-packages
mkdir -p ck_patched; cp -r $SP/circuitkit ck_patched/circuitkit
F=ck_patched/circuitkit/backends/eap/attribute_node.py; sed -i '170s/node.in_hook/node.out_hook/' $F
echo "lines changed: $(diff $SP/circuitkit/backends/eap/attribute_node.py $F | grep -c '^[<>]')" >> smoke2.log
sed -n 170p $F | cut -c1-90 >> smoke2.log
export HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1 WORK=$W/tmp OUT=$W/b1/out CUDA_VISIBLE_DEVICES=""
PYTHONPATH=/workspace/p9_ruler/stubs $W/ck/bin/python run_b1.py normal shipped 2 1 cpu >> smoke2.log 2>&1; echo "exit shipped $?" >> smoke2.log
PYTHONPATH=$W/b1/ck_patched:/workspace/p9_ruler/stubs $W/ck/bin/python run_b1.py bad_s0 patched 2 1 cpu >> smoke2.log 2>&1; echo "exit patched $?" >> smoke2.log
echo SMOKE2_DONE >> smoke2.log
