#!/bin/bash
# P11 phase 2, after run_p11.sh ends: shipping option 2 (quantized or bf16 base + the cut as a separate 16-bit LoRA, never rounded into the
# weights) and Cohere's official Q4_K_M file. Each generation has a hard 45-min limit. Stop: touch p11/STOP.
P=/Volumes/Tashon/hackathon/fable_plan/v2/p11; PY=/Volumes/Tashon/hackathon/lab/.venv/bin/python; LOG=$P/logs/pipeline.log; say(){ echo "$(date +%H:%M) $*" >> $LOG; }
until grep -q "pipeline DONE\|STOP file" $LOG; do sleep 60; done
OFF=$(ls ~/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global-GGUF/snapshots/*/tiny-aya-global-q4_k_m.gguf | head -1)
gen(){ [ -f $P/STOP ] && { say "STOP file, ending phase 2"; exit 0; }; [ -f ../judge_inbox/gen_p11_$2.jsonl.done ] && return
  say "gen $2 start"; timeout 2700 $PY -u $P/scripts/gen.py $1 $2 ${3:+--lora $3} >> $P/logs/gen_$2.log 2>&1; say "gen $2 exit $? $(tail -1 $P/logs/gen_$2.log | cut -c1-120)"; }
cd $P
gen $P/gguf/original-bf16.gguf lora_targeted20_on_bf16 $P/gguf/lora_targeted20-f16.gguf
gen $P/gguf/original-Q4_K_M.gguf lora_targeted20_on_Q4_K_M $P/gguf/lora_targeted20-f16.gguf
gen $OFF official_Q4_K_M
gen $P/gguf/original-Q2_K.gguf lora_targeted20_on_Q2_K $P/gguf/lora_targeted20-f16.gguf
say "phase 2 DONE"
