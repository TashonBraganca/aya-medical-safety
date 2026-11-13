#!/bin/bash
# Usage: watchdog.sh <logfile> <max_seconds> <stall_seconds> -- command...
# Runs the command under a hard time limit; kills it (and only it) if the log file stops growing for stall_seconds.
LOG="$1"; MAX="$2"; STALL="$3"; shift 4
timeout --signal=TERM --kill-after=30 "$MAX" "$@" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$LOG.pid"
while kill -0 "$PID" 2>/dev/null; do
  sleep 30
  AGE=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if [ "$AGE" -gt "$STALL" ]; then
    echo "WATCHDOG: no log progress for ${AGE}s, killing PID $PID" >> "$LOG"
    pkill -TERM -P "$PID" 2>/dev/null; kill -TERM "$PID" 2>/dev/null; sleep 20; pkill -KILL -P "$PID" 2>/dev/null; kill -KILL "$PID" 2>/dev/null
    break
  fi
done
wait "$PID" 2>/dev/null; echo "WATCHDOG: exit $? at $(date '+%H:%M:%S')" >> "$LOG"
