#!/bin/bash
set -u
cp -r /job /work && cd /work && mkdir -p out
pip install -q peft datasets accelerate > pip.log 2>&1 || { echo "pip FAILED"; tail -5 pip.log; }
python3 run_stune.py 2>&1 | grep --line-buffered -v -i warn
echo STUNE_EXIT ${PIPESTATUS[0]}
