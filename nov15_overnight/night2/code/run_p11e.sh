#!/bin/bash
# P11-3: targeted-cut model at Q2_K with (a) the 30 tensors the cut changed most kept at q8_0, (b) 30 random same-kind tensors at q8_0.
# Same settings as the plain Q2_K run (gen.py, no stop fix) so the plain file (58 harmful, 77 harmful-or-broken) is the baseline.
P=/Volumes/Tashon/hackathon/fable_plan/v2/p11; L=/Volumes/Tashon/hackathon/tools_llamacpp/llama.cpp/build/bin; PY=/Volumes/Tashon/hackathon/lab/.venv/bin/python; LOG=$P/logs/pipeline.log; say(){ echo "$(date +%H:%M) $*" >> $LOG; }
cd $P
for arm in cut random; do
  $L/llama-quantize --tensor-type-file $P/protect_$arm.txt $P/gguf/targeted20-bf16.gguf $P/gguf/targeted20-Q2_K-protect_$arm.gguf Q2_K > $P/logs/quant_protect_$arm.log 2>&1 || { say "ALERT quantize protect_$arm failed"; continue; }
  say "quantized protect_$arm: $(stat -f %z $P/gguf/targeted20-Q2_K-protect_$arm.gguf) bytes (plain Q2_K $(stat -f %z $P/gguf/targeted20-Q2_K.gguf))"
done
for arm in cut random; do
  say "gen targeted20_Q2_K_protect_$arm start"; timeout 2700 $PY -u $P/scripts/gen.py $P/gguf/targeted20-Q2_K-protect_$arm.gguf targeted20_Q2_K_protect_$arm >> $P/logs/gen_protect_$arm.log 2>&1
  say "gen targeted20_Q2_K_protect_$arm exit $? $(tail -1 $P/logs/gen_protect_$arm.log | cut -c1-110)"
done
say "phase 5 (protection) DONE"
