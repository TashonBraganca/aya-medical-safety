#!/bin/bash
# Job B: attention family + full fine-tune. Extras = training-seed control for the attention circuit arm.
export TAG=B MAIN="circuit_attn random_attn_s1 random_attn_s2 full" EXTRA="circuit_attn_ts1" REF="broken"
source /repo/crft_dose/run_arms.sh
