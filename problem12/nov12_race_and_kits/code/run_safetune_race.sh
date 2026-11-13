set -euo pipefail
: "${JOB_IO_REPO:?set JOB_IO_REPO to a private Hugging Face dataset you own (job inputs and outputs)}"
printf "torch==2.13.0\ntransformers==5.14.1\nnumpy==2.3.5\nsafetensors==0.8.0\nhuggingface_hub==1.32.0\n" > /c.txt
pip install -q -c /c.txt torch transformers numpy safetensors huggingface_hub easy-dataset-share "datasets>=2.14.0,<6" "accelerate>=0.24.0,<2" "peft>=0.6.0,<1" pyyaml tqdm psutil requests pandas scikit-learn click rich typer evaluate rouge-score sacrebleu langdetect immutabledict nltk sentencepiece tiktoken nvidia-ml-py
python -c "import transformers; assert transformers.__version__ == '5.14.1'"
python -c "from huggingface_hub import snapshot_download as s; s('$JOB_IO_REPO', repo_type='dataset', allow_patterns='e7/*', local_dir='/w')"
export PYTHONPATH=/w/e7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd /w/e7/kit
for MODE in smoke full; do mkdir -p /vol/out/e6/$MODE; cp /w/e7/pools.json /vol/out/e6/$MODE/; done
python -c "import lib; lib.data(); lib.load_tok()"
( while true; do nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/[gpu] /'; sleep 120; done ) &
# 1) smoke each arm ALONE, uncapped, and record the real peak of training and of attacking
declare -A PK
for spec in booster:circuit:booster tvaccine:circuit:tvaccine booster:ckexport:booster_allmods; do
  IFS=: read M W T <<< "$spec"
  a=$(E7_WHERE=$W E6_MEMFRAC=1.0 python e7_imm.py $M smoke 2>&1 | grep -vE "Loading weights|it/s\]" | tee /dev/stderr | grep PEAK_GB | awk '{print $2}')
  b=$(E6_MEMFRAC=1.0 python e6_attack.py $T smoke 2>&1 | grep -vE "Loading weights|it/s\]" | tee /dev/stderr | grep PEAK_GB | awk '{print $2}')
  PK[$T]=$(python -c "print(max($a, $b))"); echo "[plan] $T smoke ok, peak train $a GiB, attack $b GiB"
done
# 2) pack: run arms in parallel while the measured peaks (+10%) fit in the card; the rest start as slots free up
TOT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits); TOTG=$(python -c "print($TOT/1024*0.97)")
run() { T=$1; M=$2; W=$3; F=$(python -c "print(min(0.97, ${PK[$T]}*1.10/$TOTG*0.97))"); echo "[plan] start $T with memfrac $F"
  { E7_WHERE=$W E6_MEMFRAC=$F python e7_imm.py $M full && E6_MEMFRAC=$F python e6_attack.py $T full; } 2>&1 | grep -vE "Loading weights|it/s\]" | sed "s/^/[$T] /"; }
python - <<PY > /plan.txt
pk = {"booster": ${PK[booster]}, "tvaccine": ${PK[tvaccine]}, "booster_allmods": ${PK[booster_allmods]}}; tot = $TOTG
used = 0; now = []; later = []
for t in ("booster", "tvaccine", "booster_allmods"):
    (now if used + pk[t] * 1.1 <= tot else later).append(t); used += pk[t] * 1.1 if t in now else 0
print(" ".join(now)); print(" ".join(later)); print(f"packed {round(used,1)} of {round(tot,1)} GiB", file=__import__("sys").stderr)
PY
NOW=$(sed -n 1p /plan.txt); LATER=$(sed -n 2p /plan.txt); echo "[plan] parallel now: $NOW | queued: $LATER"
declare -A MM=([booster]="booster circuit" [tvaccine]="tvaccine circuit" [booster_allmods]="booster ckexport")
PIDS=(); for T in $NOW; do run $T ${MM[$T]} & PIDS+=($!); sleep 45; done
for T in $LATER; do wait -n || true; run $T ${MM[$T]} & PIDS+=($!); done
RC=0; for p in "${PIDS[@]}"; do wait $p || RC=1; done; echo "[plan] all arms finished rc=$RC"
mkdir -p /up && cp /vol/out/e6/full/*booster* /vol/out/e6/full/*tvaccine* /up/ 2>/dev/null || true; ls -la /up
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='/up', path_in_repo='e7_out/auto', repo_id='$JOB_IO_REPO', repo_type='dataset'); print('UPLOADED')"
