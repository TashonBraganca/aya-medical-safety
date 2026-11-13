#!/bin/bash
set -u
for s in 1 2; do cp -r /job /work$s; (cd /work$s && mkdir -p out/full up && SEED=$s python3 step4_final_hf.py 2>&1 | grep --line-buffered -v -i warn | sed -u "s/^/[m$((s+1))] /"; echo "FINAL_EXIT seed $s ${PIPESTATUS[0]}") & done
wait
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ALL_FINAL_DONE
