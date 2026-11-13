#!/bin/bash
set -u
cp -r /job /work && cd /work && mkdir -p out
python3 -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__,'gpu',torch.cuda.get_device_name(0))"
python3 step3_attrib.py --out out --bs 16 2>&1 | grep -v -i warn
echo STEP3_EXIT $?
