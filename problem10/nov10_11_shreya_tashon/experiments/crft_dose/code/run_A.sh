#!/bin/bash
# Job A: MLP family (+ normal/broken references). Extras = training-seed control for the circuit arm.
export TAG=A MAIN="circuit_mlp random_mlp_s1 random_mlp_s2 random_mlp_s3 random_mlp_s0" EXTRA="circuit_mlp_ts1" REF="normal,broken"
source /repo/crft_dose/run_arms.sh
