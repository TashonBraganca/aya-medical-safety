#!/bin/bash
# Install SafeTune's runtime deps without its full pin set (some optional pins fail to build), then
# keep installing any module the judge import still reports missing (max 12 rounds).
cd /work
pip install -q "transformers>=5.15,<6" datasets accelerate "peft<1" pyyaml pandas scikit-learn numpy tqdm psutil requests rich click typer langdetect immutabledict nltk sentencepiece tiktoken protobuf huggingface_hub 2>&1 | grep -i "^error" | tail -2
declare -A MAP=([yaml]=pyyaml [sklearn]=scikit-learn [PIL]=pillow [cv2]=opencv-python-headless)
for i in $(seq 1 12); do
  out=$(cd eval && python -c "import sys;sys.path.insert(0,'safetune_src');from safetune.instrumentation.evaluate.judges import run_judge;from safetune.instrumentation.evaluate.judges._templates import _HARMBENCH_PROMPT;print('SAFETUNE_IMPORT_OK')" 2>&1)
  echo "$out" | grep -q SAFETUNE_IMPORT_OK && { echo SAFETUNE_IMPORT_OK; exit 0; }
  mod=$(echo "$out" | grep -oE "No module named '[^'.]+" | head -1 | sed "s/No module named '//")
  [ -z "$mod" ] && { echo "$out" | tail -5; echo SETUP_FAILED; exit 1; }
  pkg=${MAP[$mod]:-$mod}; echo "installing missing $pkg"; pip install -q "$pkg" 2>&1 | grep -i "^error" | tail -1
done
echo SETUP_FAILED; exit 1
