#!/bin/bash
set -u
cp -r /job /work && cd /work
python3 step3_seeds.py 2>&1 | grep --line-buffered -v -i warn
echo SEEDS_EXIT ${PIPESTATUS[0]}
