#!/bin/bash
# P11 phase 4 (after phase 3): MMLU-500 on bf16 and Q4_K_M files of original, targeted cut, random cut, broken (the "nothing else broke" check).
P=/Volumes/Tashon/hackathon/fable_plan/v2/p11; PY=/Volumes/Tashon/hackathon/lab/.venv/bin/python; LOG=$P/logs/pipeline.log; say(){ echo "$(date +%H:%M) $*" >> $LOG; }
true
cd $P
for m in original targeted20 random20_d0 broken; do for f in bf16 Q4_K_M; do
  [ -f $P/STOP ] && break 2; timeout 1800 $PY -u $P/scripts/mmlu_gguf.py $P/gguf/$m-$f.gguf ${m}_$f >> $P/logs/mmlu.log 2>&1; say "mmlu ${m}_$f exit $? $(tail -1 $P/logs/mmlu.log | cut -c1-100)"
done; done
say "phase 4 DONE"
