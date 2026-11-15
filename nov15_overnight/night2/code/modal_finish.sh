#!/bin/bash
# Every 2 min: (1) billing guard: stop both sandboxes if tashon-braganca credit used reaches $28.5 of $30, or any billed cost appears;
# (2) when a job's watchdog has exited: copy its answer files into judge_inbox (.done markers), its result JSON into res/live, then stop that sandbox.
M=modal; V=/Volumes/Tashon/hackathon/fable_plan/v2; L=$V/modal_finish.log; cd $M
run(){ MODAL_PROFILE=tashon-braganca SANDBOX_ID_FILE=sb_$1.txt timeout 400 uv run --quiet --with modal --with huggingface_hub --python 3.12 python msb.py "${@:2}" 2>&1 | grep -v HF_TOKEN; }
left="E F"
while [ -n "$left" ]; do
  OUT=$(MODAL_PROFILE=tashon-braganca timeout 90 uv run --quiet --with modal modal billing summary 2>&1)
  USED=$(echo "$OUT" | awk '/Credits:/ {gsub("-","",$2); print $2}'); BILLED=$(echo "$OUT" | awk '/Billed Cost:/ {gsub("\\$","",$3); print $3}')
  echo "$(date +%H:%M) credits_used=$USED billed=$BILLED" >> $L
  if [ -n "$USED" ] && { awk "BEGIN{exit !($USED >= 28.5)}" || awk "BEGIN{exit !(${BILLED:-0} > 0)}"; }; then
    echo "$(date +%H:%M) ALERT credit limit: stopping sandboxes $left" >> $L; for s in $left; do run $s stop >> $L; done; exit 0; fi
  for s in $left; do
    case $s in E) log=/vol/logs/p12m_adapt_modal.log; gd=/vol/v2out/p12m/full; res=adaptive.json;; F) log=/vol/logs/p10m_modal.log; gd=/vol/v2out/p10m/full; res=p10m.json;; esac
    if run $s sh "grep -q 'WATCHDOG: exit' $log && echo ENDED" | grep -q ENDED; then
      run $s sh "tail -n 2 $log" | sed "s/^/$(date +%H:%M) $s end: /" >> $L
      for f in $(run $s sh "ls $gd/gen/ 2>/dev/null"); do run $s get $gd/gen/$f $V/judge_inbox/$f > /dev/null && [ -s $V/judge_inbox/$f ] && echo "from modal" > $V/judge_inbox/$f.done && echo "$(date +%H:%M) inbox $f" >> $L; done
      run $s get $gd/$res $V/res/live/modal_$res > /dev/null; run $s stop >> $L; echo "$(date +%H:%M) stopped sandbox $s" >> $L
      left=$(echo $left | sed "s/$s//" | xargs)
    fi
  done
  sleep 120
done
echo "$(date +%H:%M) MODAL ALL DONE" >> $L
