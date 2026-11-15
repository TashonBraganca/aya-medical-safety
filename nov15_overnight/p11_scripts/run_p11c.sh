#!/bin/bash
# P11 phase 3 (after phase 2): regenerate with the stop fix: original bf16 / Q4_K_M / Q2_K, and the targeted cut bf16 as a control.
P=/Volumes/Tashon/hackathon/fable_plan/v2/p11; PY=/Volumes/Tashon/hackathon/lab/.venv/bin/python; LOG=$P/logs/pipeline.log; say(){ echo "$(date +%H:%M) $*" >> $LOG; }
until grep -q "phase 2 DONE\|STOP file" $LOG; do sleep 60; done
cd $P
for x in "original-bf16 original_bf16_stopfix" "targeted20-bf16 targeted20_bf16_stopfix" "original-Q4_K_M original_Q4_K_M_stopfix" "original-Q2_K original_Q2_K_stopfix"; do
  set -- $x; [ -f $P/STOP ] && break; say "gen $2 start"; timeout 2700 $PY -u $P/scripts/gen.py $P/gguf/$1.gguf $2 --stopfix >> $P/logs/gen_$2.log 2>&1; say "gen $2 exit $? $(tail -1 $P/logs/gen_$2.log | cut -c1-120)"
done
say "phase 3 DONE"
