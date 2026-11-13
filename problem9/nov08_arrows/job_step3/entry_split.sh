#!/bin/bash
set -u
cp -r /job /work && cd /work
python3 step3_split.py 2>&1 | grep --line-buffered -v -i warn
echo SPLIT_EXIT ${PIPESTATUS[0]}
