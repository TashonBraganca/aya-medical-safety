#!/bin/bash
# Batch 2: (A) full suite on models 2 and 3 (broken + fix); (B) Hindi recheck, 1,000 questions, all 7 models.
set -uo pipefail
E23="broken2=crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_1,fix2=tashom/aya-broken-medical-arrow-fixes:model2_our256_off,broken3=crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_2,fix3=tashom/aya-broken-medical-arrow-fixes:model3_our256_off"
MODELS=, EXTRA="$E23" DEST=nothing_broke/results_m23 bash /repo/nothing_broke/run_nb.sh
rm -rf /work/out
MODELS=normal,broken,our256,big256 EXTRA="$E23" DEST=nothing_broke/results_hi1000 ARGS="--tests gmmlu --langs hi --n_gmmlu 1000" bash /repo/nothing_broke/run_nb.sh
echo BATCH2_DONE
