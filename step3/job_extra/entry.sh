#!/bin/bash
set -u
cp -r /job /work && cd /work && mkdir -p out
python3 step3_extra.py 2>&1 | grep --line-buffered -v -i warn
echo EXTRA_EXIT ${PIPESTATUS[0]}
