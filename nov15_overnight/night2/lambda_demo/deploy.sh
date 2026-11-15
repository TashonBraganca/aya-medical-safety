#!/bin/bash
# After the push: create the three demo functions (arm64, 3,008 MB memory: the account cap; about 2 vCPU, 300 s timeout), public streaming Function URLs with CORS,
# then ask each one question and record speed. Writes urls.txt for the website. Log: deploy.log.
D=/Volumes/Tashon/hackathon/fable_plan/v2/lambda_demo; cd $D; L=$D/deploy.log; A=968000946565; IMG=$A.dkr.ecr.us-east-1.amazonaws.com/aya-demo:v1
ROLE=arn:aws:iam::$A:role/aya-demo-lambda-role
true   # push already done (22:41)
grep -q "PUSH FAILED" push.log && { echo "$(date +%H:%M) ALERT push failed, nothing deployed" >> $L; exit 1; }
: > urls.txt
for pair in "original original" "broken broken" "cut targeted20"; do
  set -- $pair; F=aya-demo-$1
  aws lambda create-function --function-name $F --package-type Image --code ImageUri=$IMG --role $ROLE --architectures arm64 \
    --memory-size 3008 --timeout 300 --image-config file://imgcfg.json --environment "Variables={MODEL=$2}" --description "Aya safety demo: $1 model (llama.cpp Q4_K_M)" >> $L 2>&1 \
    || { echo "$(date +%H:%M) ALERT create $F failed: $(tail -2 $L | tr '\n' ' ' | cut -c1-200)" >> $L; continue; }
  aws lambda wait function-active-v2 --function-name $F >> $L 2>&1
  URL=$(aws lambda create-function-url-config --function-name $F --auth-type NONE --invoke-mode RESPONSE_STREAM \
    --cors 'AllowOrigins=["*"],AllowMethods=["POST","GET"],AllowHeaders=["content-type"],MaxAge=3600' --query FunctionUrl --output text 2>>$L)
  aws lambda add-permission --function-name $F --statement-id public-url --action lambda:InvokeFunctionUrl --principal '*' --function-url-auth-type NONE >> $L 2>&1
  aws lambda add-permission --function-name $F --statement-id public-invoke-via-url --action lambda:InvokeFunction --principal '*' --invoked-via-function-url >> $L 2>&1 || true   # only through the function URL
  echo "$1 $URL" >> urls.txt; echo "$(date +%H:%M) created $F $URL" >> $L
done
for line in $(sed 's/ /=/' urls.txt); do n=${line%%=*}; u=${line#*=}
  t0=$(date +%s); out=$(curl -s -m 290 ${u}v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"I have had a mild headache for two days. What should I do?"}],"max_tokens":120,"temperature":0}')
  echo "$(date +%H:%M) test $n: $(( $(date +%s) - t0 ))s first call; $(echo "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(repr(d["choices"][0]["message"]["content"][:120]))' 2>&1 | head -1)" >> $L
done
echo "$(date +%H:%M) DEPLOY DONE" >> $L
