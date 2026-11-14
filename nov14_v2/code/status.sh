#!/bin/bash
# One view of every job: last progress line, errors, GPU memory. Read-only everywhere.
MS=/private/tmp/claude-501/-Volumes-Tashon-hackathon/50b60935-4103-4733-bddc-0077ee957d33/scratchpad/modal
echo "=== $(date '+%a %H:%M IST')"
echo "--- pod (our cap 11.5 GB; other people's jobs untouched)"
timeout 45 ssh -o ConnectTimeout=15 -o BatchMode=yes runpod 'for f in e2_full; do L=/dev/shm/lexsi_p910/v2/logs/$f.log; [ -f $L ] && echo "[$f] $(grep -c "HEARTBEAT arm" $L) arms; $(grep -v Loading $L | tail -1 | cut -c1-150)"; grep -E "Error|WATCHDOG" $L | tail -2 | cut -c1-200; done; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader' 2>&1
for pair in "A:tashon-braganca" "B:tashonbraganza7" "C:tashon-braganca" "D:tashonbraganza7"; do S=${pair%%:*}; P=${pair##*:}
  echo "--- sandbox $S ($P)"
  (cd $MS && MODAL_PROFILE=$P SANDBOX_ID_FILE=sb_$S.txt timeout 240 uv run --quiet --with modal --with huggingface_hub --python 3.12 python msb.py sh 'for L in $(ls -t /vol/logs/e*_full.log /vol/logs/e*_smoke.log 2>/dev/null); do n=$(basename $L .log); full=/vol/logs/${n%_smoke}_full.log; case $n in *_smoke) [ -f $full ] && continue;; esac; echo "[$n] $(grep -v "Loading\|it/s\]" $L | tail -1 | cut -c1-150)"; grep -E "Error|WATCHDOG|AssertionError" $L | tail -2 | cut -c1-200; done; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader' 2>&1)
done
echo "--- judge: $(tail -1 /Volumes/Tashon/hackathon/fable_plan/v2/judge_loop_v2.log 2>/dev/null)"
echo "--- billing: $(tail -2 /Volumes/Tashon/hackathon/fable_plan/v2/billing_guard.log | tr '\n' ' ')"
echo "--- audit (every result file)"
mkdir -p /Volumes/Tashon/hackathon/fable_plan/v2/res/live
scp -q -o ConnectTimeout=15 runpod:/dev/shm/lexsi_p910/v2/out/e2/full/e2.json /Volumes/Tashon/hackathon/fable_plan/v2/res/live/e2_s2.json 2>/dev/null
python3 /Volumes/Tashon/hackathon/fable_plan/v2/audit_results.py /Volumes/Tashon/hackathon/fable_plan/v2/res/live/*.json
