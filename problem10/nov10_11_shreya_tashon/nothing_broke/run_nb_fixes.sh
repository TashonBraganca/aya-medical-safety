#!/bin/bash
# Batch 3: (A) full suite on fix3 (Tashon's model-3 arrow fix); (B) Hindi recheck, 1,000 questions, on the fixes
# (our256 = fix1, big256, fix2, fix3). These are TASHON'S uploaded files (tashom/aya-broken-medical-arrow-fixes), not rebuilds.
set -uo pipefail
FIX=tashom/aya-broken-medical-arrow-fixes
E23="fix2=$FIX:model2_our256_off,fix3=$FIX:model3_our256_off"
MODELS=, EXTRA="fix3=$FIX:model3_our256_off" DEST=nothing_broke/results_m3fix bash /repo/nothing_broke/run_nb.sh
rm -rf /work/out
MODELS=our256,big256 EXTRA="$E23" DEST=nothing_broke/results_hi1000_fixes ARGS="--tests gmmlu --langs hi --n_gmmlu 1000" bash /repo/nothing_broke/run_nb.sh
echo BATCH3_DONE
