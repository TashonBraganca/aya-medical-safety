#!/bin/bash
cd /dev/shm/p9/floors
while kill -0 1019328 2>/dev/null && ! grep -q SINGLES_DONE run.log; do sleep 5; done
grep -q SINGLES_DONE run.log && kill 1019328 2>/dev/null
sleep 20
PYTHONPATH=/workspace/p9_ruler/stubs:/dev/shm/p9/s4/stub2 HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1 python floors_focus.py > focus.log 2>&1
