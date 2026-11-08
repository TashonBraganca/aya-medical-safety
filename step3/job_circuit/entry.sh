#!/bin/bash
set -u
cp -r /job /work && cd /work && mkdir -p out
pip install -q ./CircuitKIT > pip.log 2>&1 || { echo "pip FAILED"; tail -20 pip.log; exit 1; }
pip list 2>/dev/null | grep -i -E "^(circuitkit|transformer-lens|transformers|torch) "
for M in ${MODELS:-bad_s0}; do
  python3 run_circuit.py $M ${NEX:-4} 2>&1 | grep --line-buffered -v -i warn
  mkdir -p up/p9_step3/circuitkit && cp -r out/* up/p9_step3/circuitkit/ 2>/dev/null; find up -name "*.pt" -size +50M -delete
  hf upload tashom/r910-results up . --repo-type dataset --commit-message "p9 step3 circuitkit $M n=${NEX:-4}" > /dev/null 2>&1 && echo "UPLOADED $M"
done
echo CIRCUIT_END
