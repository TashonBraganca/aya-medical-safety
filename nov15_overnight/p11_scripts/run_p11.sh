#!/bin/bash
# P11 pipeline, one step at a time: merge -> bf16 GGUF -> quantize (Q8_0, Q4_K_M, Q4_0, Q2_K) -> 250 answers per file -> judge inbox.
# Order: targeted cut, original, broken, random cut. Each generation has a hard 45-min limit. Stop: touch p11/STOP.
P=/Volumes/Tashon/hackathon/fable_plan/v2/p11; L=/Volumes/Tashon/hackathon/tools_llamacpp/llama.cpp; PY=/Volumes/Tashon/hackathon/lab/.venv/bin/python
LOG=$P/logs/pipeline.log; say(){ echo "$(date +%H:%M) $*" >> $LOG; }
gen(){ [ -f $P/STOP ] && { say "STOP file, ending"; exit 0; }; [ -f ../judge_inbox/gen_p11_$2.jsonl.done ] && return
  say "gen $2 start"; timeout 2700 $PY -u $P/scripts/gen.py $1 $2 ${3:+--lora $3} >> $P/logs/gen_$2.log 2>&1; say "gen $2 exit $? $(tail -1 $P/logs/gen_$2.log | cut -c1-120)"; }
build(){ m=$1; [ -f $P/gguf/$m-bf16.gguf ] && return
  if [ "$m" != original ]; then say "merge $m"; $PY $P/scripts/merge.py $P/adapters/$m $P/merged_$m >> $P/logs/merge_$m.log 2>&1 || { say "ALERT merge $m failed"; return 1; }; say "merge $m ok $(cat $P/merged_$m/merge_report.json)"
    (cd $L && PYTHONPATH=$L/gguf-py $PY convert_hf_to_gguf.py $P/merged_$m --outtype bf16 --outfile $P/gguf/$m-bf16.gguf > $P/logs/convert_$m.log 2>&1) || { say "ALERT convert $m failed"; return 1; }
    rm -rf $P/merged_$m; say "convert $m ok"; fi; }
quant(){ m=$1; q=$2; [ -f $P/gguf/$m-$q.gguf ] || { $L/build/bin/llama-quantize $P/gguf/$m-bf16.gguf $P/gguf/$m-$q.gguf $q > $P/logs/quant_$m-$q.log 2>&1 || say "ALERT quantize $m $q failed"; }; }
say "pipeline start"
for m in targeted20 original broken random20_d0; do
  build $m || continue
  for q in Q8_0 Q4_K_M Q4_0 Q2_K; do quant $m $q; done
  case $m in random20_d0) F="bf16 Q4_K_M Q2_K";; *) F="bf16 Q8_0 Q4_K_M Q4_0 Q2_K";; esac
  for f in $F; do gen $P/gguf/$m-$f.gguf ${m}_$f; done
done
say "pipeline DONE"
