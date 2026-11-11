#!/bin/bash
# One A100 80 GB, many processes. /job (read-only) holds the kit and scripts; /out (read-write bucket) gets every result.
set -u
mkdir -p /work/models /work/adapters /out/logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
( while true; do echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)" >> /out/logs/gpu.log; sleep 60; done ) &
python3 /job/dl.py || exit 1
du -sh /work/models/* /work/adapters; ls -l /work/adapters | wc -l
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
task() {   # $1 = name, rest = script and args; smoke first, full only if the smoke finishes cleanly
  name=$1; shift; script=$1; shift
  GPU_FRACTION=$FRAC python3 /job/$script smoke "$@" > /out/logs/${name}_smoke.log 2>&1
  if [ $? -ne 0 ]; then echo "$(date +%H:%M) SMOKE FAILED $name"; tail -5 /out/logs/${name}_smoke.log; return; fi
  echo "$(date +%H:%M) smoke ok $name, full starts"
  GPU_FRACTION=$FRAC python3 /job/$script full "$@" > /out/logs/${name}_full.log 2>&1
  echo "$(date +%H:%M) FULL EXIT $? $name"; tail -3 /out/logs/${name}_full.log
}
export -f task
printf '%s\n' \
  "0.12 gap_tok gap_wrap.py tok" "0.12 gap_ood gap_wrap.py ood" "0.12 gap_ml gap_wrap.py ml" \
  "0.14 water_s0 water_job.py 0" "0.13 water_s1 water_job.py 1" "0.13 water_s2 water_job.py 2" \
  "0.12 patch_s0 patch_job.py 0" "0.12 patch_s1 patch_job.py 1" "0.12 patch_s2 patch_job.py 2" |
  xargs -P 6 -L 1 bash -c 'FRAC=$0; export FRAC; task "$@"'
echo ALL_TASKS_DONE
