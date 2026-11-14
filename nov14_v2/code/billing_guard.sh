#!/bin/bash
# Every 10 minutes: read each Modal account's billing; terminate its sandbox when credit used >= $27 of $30, or any real billed cost appears.
MS=/private/tmp/claude-501/-Volumes-Tashon-hackathon/50b60935-4103-4733-bddc-0077ee957d33/scratchpad/modal
LOG=/Volumes/Tashon/hackathon/fable_plan/v2/billing_guard.log
while [ ! -f /Volumes/Tashon/hackathon/fable_plan/v2/STOP_GUARD ]; do
  for pair in "tashon-braganca:sb_A.txt" "tashon-braganca:sb_C.txt" "tashonbraganza7:sb_B.txt" "tashonbraganza7:sb_D.txt"; do
    P=${pair%%:*}; IDF=${pair##*:}
    OUT=$(MODAL_PROFILE=$P timeout 90 uv run --quiet --with modal modal billing summary 2>&1)
    USED=$(echo "$OUT" | awk '/Credits:/ {gsub("-","",$2); print $2}'); BILLED=$(echo "$OUT" | awk '/Billed Cost:/ {gsub("\\$","",$3); print $3}')
    echo "$(date '+%H:%M') $P credits_used=$USED billed=$BILLED" >> $LOG
    if [ -n "$USED" ] && { awk "BEGIN{exit !($USED >= $( [ $P = tashonbraganza7 ] && echo 29.2 || echo 26 ))}" || awk "BEGIN{exit !(${BILLED:-0} > 0)}"; }; then
      echo "$(date '+%H:%M') LIMIT on $P: terminating its sandbox" >> $LOG
      (cd $MS && MODAL_PROFILE=$P SANDBOX_ID_FILE=$IDF timeout 120 uv run --quiet --with modal --with huggingface_hub --python 3.12 python msb.py stop >> $LOG 2>&1)
    fi
  done
  sleep 300
done
