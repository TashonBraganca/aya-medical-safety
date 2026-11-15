#!/bin/bash
# Turn the demo OFF (every request is refused, nothing can bill) or back ON.  Usage: bash off_switch.sh off|on
for F in aya-demo-original aya-demo-broken aya-demo-cut; do
  if [ "$1" = off ]; then aws lambda put-function-concurrency --function-name $F --reserved-concurrent-executions 0 && echo "$F off"
  else aws lambda delete-function-concurrency --function-name $F && echo "$F on"; fi
done
# The keep-warm schedule (EventBridge rule aya-demo-keepwarm, every 5 min) is switched with the demo:
if [ "$1" = off ]; then aws events disable-rule --name aya-demo-keepwarm && echo "keep-warm off"; else aws events enable-rule --name aya-demo-keepwarm && echo "keep-warm on"; fi
