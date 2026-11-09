#!/bin/bash
# Nothing-broke suite on the 4 circuit-restricted fine-tunes (adapters sit on top of merged em0; crft_ prefix handles that).
set -uo pipefail
A=/repo/crft/adapters
MODELS=, EXTRA="crft_circuit=$A/crft_circuit/adapter,crft_random=$A/crft_random/adapter,crft_tight=$A/crft_tight/adapter,crft_full=$A/crft_full/adapter" \
  DEST=nothing_broke/results_crft bash /repo/nothing_broke/run_nb.sh
