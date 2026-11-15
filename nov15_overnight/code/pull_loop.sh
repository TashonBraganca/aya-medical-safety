#!/bin/bash
# Every 10 min: copy tonight's result files (pod: e3x, p12a, xray_all, p11x; HF job A: e3 json) into res/live for the audit. Read-only on the pod.
V=/Volumes/Tashon/hackathon/fable_plan/v2; L=$V/res/live
while [ ! -f $V/STOP_PULL ]; do
  for f in e3x/full/e3x.json p12a/full/p12a.json xray_all/full/xray_all.json; do
    n=pod_$(echo $f | tr / _); scp -q -o ConnectTimeout=15 runpod:/dev/shm/lexsi_p910/v2/out/$f $L/$n.tmp 2>/dev/null && mv $L/$n.tmp $L/$n
  done
  timeout 120 python3 -c "
from huggingface_hub import HfApi, hf_hub_download; import shutil
a=HfApi(); fs=[f for f in a.list_repo_files('Tahsoom/lexsi-p10b-out', repo_type='dataset') if f.startswith('e3/full/') and f.endswith('.json') and 'scores' not in f]
for f in fs: shutil.copy(hf_hub_download('Tahsoom/lexsi-p10b-out', f, repo_type='dataset', force_download=True, cache_dir='/tmp/_p10b_pull'), '$L/hfA_'+f.replace('/','_'))
" 2>/dev/null
  sleep 600
done
