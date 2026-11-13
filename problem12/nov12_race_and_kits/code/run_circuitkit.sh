set -uo pipefail
: "${JOB_IO_REPO:?set JOB_IO_REPO to a private Hugging Face dataset you own (job inputs and outputs)}"
pip install -q "transformers==5.22.1" easy-dataset-share
python -c "from huggingface_hub import snapshot_download as s; s('$JOB_IO_REPO', repo_type='dataset', allow_patterns='ck/*', local_dir='/w')"
pip install -q -e /w/ck/CircuitKIT "transformers==5.22.1"
cd /w/ck/kit; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CK_BS=1 CK_NTEST=96 CK_MEMFRAC=0.95 PYTHONUNBUFFERED=1; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | sed -u 's/^/[gpu] /'; sleep 60; done ) &
# one model at a time, each with the whole card (measured: batch 4 ~40 GB, batch 8 >75 GB; every prompt ~440 tokens (system preamble); batch 2 = 42.6 GB observed, batch 1 ~34 GB predicted)
for WHO in ${CK_WHO:-broken}; do
  python ck_faith.py $WHO full 2>&1 | grep --line-buffered -vE "Loading weights|it/s\]|s/it\]" | sed -u "s/^/[$WHO] /"; echo "[plan] $WHO exit ${PIPESTATUS[0]}"
done
mkdir -p /up && cp /vol/out/ck/full/*.json /up/ 2>/dev/null || true; ls -la /up
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(folder_path='/up', path_in_repo='ck_out', repo_id='$JOB_IO_REPO', repo_type='dataset'); print('UPLOADED')"
