#!/bin/bash
cd /dev/shm/p9/floors
for p in 1021980 1019906 1020158; do grep -qaE "chain2.sh|chain.sh|floors_focus" /proc/$p/cmdline 2>/dev/null && kill $p; done
sleep 8; pgrep -f "python floors_focus.py" && { echo still-running; exit 1; }
PYTHONPATH=/workspace/p9_ruler/stubs:/dev/shm/p9/s4/stub2 HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1 python floors_light.py > light.log 2>&1
