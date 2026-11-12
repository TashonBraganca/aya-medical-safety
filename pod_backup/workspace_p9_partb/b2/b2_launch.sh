#!/bin/bash
cd /workspace/p9_partb/b2
if [ -e b2.log ]; then echo "b2.log exists, not starting"; exit 1; fi
export PYTHONPATH=/dev/shm/p9/s4/stune_1d824f0/src:/workspace/p9_ruler/stubs:/dev/shm/p9/s4/stub2 HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1
python b2_run.py > b2.log 2>&1
