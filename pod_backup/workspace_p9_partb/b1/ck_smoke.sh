#!/bin/bash
W=/workspace/p9_partb; cd $W/b1
export PYTHONPATH=/workspace/p9_ruler/stubs HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1 WORK=$W/tmp OUT=$W/b1/out CUDA_VISIBLE_DEVICES=""
$W/ck/bin/python run_b1.py normal shipped 2 1 cpu > smoke.log 2>&1; echo "exit $?" >> smoke.log; echo SMOKE_DONE >> smoke.log
