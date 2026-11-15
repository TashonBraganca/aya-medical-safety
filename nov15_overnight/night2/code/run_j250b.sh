#!/bin/bash
# J250-B (fresh test set) answers with the real llama.cpp Q8_0 files, stop fix on, frozen method list written before any J250-B result:
# original, broken add-on (public medicine seed 2), targeted 20% cut, truly random 20% cut (d0).
P=/Volumes/Tashon/hackathon/fable_plan/v2/p11; PY=/Volumes/Tashon/hackathon/lab/.venv/bin/python; LOG=$P/logs/pipeline.log; say(){ echo "$(date +%H:%M) $*" >> $LOG; }
export GEN_QSET=/Volumes/Tashon/hackathon/fable_plan/v2/kit/j250b.json; cd $P
for m in targeted20 random20_d0 broken original; do
  say "gen j250b_${m}_Q8_0 start"; timeout 3000 $PY -u $P/scripts/gen.py $P/gguf/$m-Q8_0.gguf j250b_${m}_Q8_0 --stopfix >> $P/logs/gen_j250b_$m.log 2>&1
  say "gen j250b_${m}_Q8_0 exit $? $(tail -1 $P/logs/gen_j250b_$m.log | cut -c1-110)"
done
say "J250-B DONE"
