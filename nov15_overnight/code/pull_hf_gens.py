"""Every 10 min: copy finished job A answer files (250 lines, none empty) from the private HF dataset into judge_inbox with a .done marker."""
import json, os, shutil, time
from huggingface_hub import HfApi, hf_hub_download
V2 = "/Volumes/Tashon/hackathon/fable_plan/v2"; a = HfApi()
while not os.path.exists(f"{V2}/STOP_PULL"):
    try:
        for f in a.list_repo_files("Tahsoom/lexsi-p10b-out", repo_type="dataset"):
            if not (f.startswith("e3/full/gen/") and f.endswith(".jsonl")): continue
            dst = f"{V2}/judge_inbox/{os.path.basename(f)}"
            if os.path.exists(dst + ".done"): continue
            p = hf_hub_download("Tahsoom/lexsi-p10b-out", f, repo_type="dataset", force_download=True, cache_dir="/tmp/_p10b_gen")
            rows = [json.loads(l) for l in open(p)]
            if len(rows) == 250 and sum(not r["reply"].strip() for r in rows) < 5:
                shutil.copy(p, dst); open(dst + ".done", "w").write("from HF job A"); print(time.strftime("%H:%M"), "inbox", os.path.basename(f), flush=True)
    except Exception as e: print(time.strftime("%H:%M"), "pull error", type(e).__name__, str(e)[:120], flush=True)
    time.sleep(600)
