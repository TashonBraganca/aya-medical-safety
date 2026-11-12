#!/bin/bash
cd /dev/shm/p9/floors
pgrep -f "python floors_(light|biggest|focus).py" >/dev/null && { echo "already running"; exit 1; }
PYTHONPATH=/workspace/p9_ruler/stubs:/dev/shm/p9/s4/stub2 HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1 python floors_light.py > light2.log 2>&1
grep -q LIGHT_DONE light2.log || { echo "light run did not finish"; exit 1; }
sleep 10
PYTHONPATH=/workspace/p9_ruler/stubs:/dev/shm/p9/s4/stub2 HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1 python floors_biggest.py > big2.log 2>&1
