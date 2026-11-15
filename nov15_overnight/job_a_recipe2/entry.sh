#!/bin/bash
# HF job A (P10-2, different recipe): train one two-job add-on with LoRA r16, alpha 32, plain scaling (alpha/r = 2), lr 1e-4, seed 0,
# then the 20% cuts (targeted, skill-protected, truly random x3, biggest first) and 250 answers each. Smoke run first; full only if smoke ends DONE.
# Every 10 min /out is uploaded to the private dataset Tahsoom/lexsi-p10b-out, so a timeout loses at most 10 minutes.
set -u; mkdir -p /out/logs /job/work; cd /job
pip install -q "transformers==5.14.1" accelerate safetensors numpy datasets easy-dataset-share sentencepiece protobuf scipy "huggingface_hub[cli]" > /out/logs/pip.log 2>&1; echo "pip exit $?" >> /out/logs/entry.log
hf download Tahsoom/lexsi-p10b-kit --repo-type dataset --local-dir /job/kit > /dev/null 2>&1; ls /job/kit >> /out/logs/entry.log
export E3_ROOT=/job/kit E3_OUT=/out LEXSI_ROOT=/job/work MEMFRAC=0.9 PYTHONUNBUFFERED=1
export P10B_R=16 P10B_ALPHA=32 P10B_RSLORA=0 P10B_LR=1e-4 E3_FAST=1 E3_FRS=0.20 E3_NOSPLIT=1
export E3_GEN=e2c_fp_net_20,e2c_fp_blame_20,true_random_20_d0,true_random_20_d1,true_random_20_d2,biggest_first_20
( while true; do sleep 600; hf upload Tahsoom/lexsi-p10b-out /out . --repo-type dataset --private --quiet > /dev/null 2>&1; done ) &
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> /out/logs/entry.log
bash /job/kit/watchdog.sh /out/logs/smoke.log 2400 900 -- python /job/kit/e3_twojob_v2.py smoke 0
if grep -q "DONE" /out/logs/smoke.log; then echo "smoke DONE" >> /out/logs/entry.log
  bash /job/kit/watchdog.sh /out/logs/full.log 8400 1200 -- python /job/kit/e3_twojob_v2.py full 0
  echo "full ended: $(tail -1 /out/logs/full.log)" >> /out/logs/entry.log
else echo "SMOKE_FAILED: $(grep -E 'Error|assert' /out/logs/smoke.log | tail -2)" >> /out/logs/entry.log; fi
hf upload Tahsoom/lexsi-p10b-out /out . --repo-type dataset --private > /out/logs/final_upload.log 2>&1; echo "final upload exit $?" >> /out/logs/entry.log
hf upload Tahsoom/lexsi-p10b-out /out/logs logs --repo-type dataset --private > /dev/null 2>&1; echo ALL_DONE
