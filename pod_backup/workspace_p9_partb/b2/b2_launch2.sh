#!/bin/bash
cd /workspace/p9_partb/b2
export PYTHONPATH=/dev/shm/p9/s4/stune_1d824f0/src:/workspace/p9_ruler/stubs:/dev/shm/p9/s4/stub2 HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1
python b2_run.py 0,1,2,3 > b2_tool.log 2>&1
