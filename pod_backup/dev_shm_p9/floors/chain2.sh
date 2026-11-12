#!/bin/bash
cd /dev/shm/p9/floors
while kill -0 1019906 2>/dev/null; do sleep 10; done
grep -q FOCUS_DONE focus.log || { echo "single-floor run did not finish, not starting groups"; exit 1; }
sleep 15
PYTHONPATH=/workspace/p9_ruler/stubs:/dev/shm/p9/s4/stub2 HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1 python floors_sets.py > sets.log 2>&1
