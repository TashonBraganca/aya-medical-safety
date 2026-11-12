#!/bin/bash
cd /dev/shm/p9/floors
while kill -0 1022381 2>/dev/null; do sleep 5; done
grep -q LIGHT_DONE light.log || { echo "light run did not finish"; exit 1; }
sleep 10
PYTHONPATH=/workspace/p9_ruler/stubs:/dev/shm/p9/s4/stub2 HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1 python floors_biggest.py > big.log 2>&1
