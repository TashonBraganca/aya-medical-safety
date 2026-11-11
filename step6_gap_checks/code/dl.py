"""Download everything the job needs; print the real error and stop on any failure."""
import os, sys, time
from huggingface_hub import snapshot_download, hf_hub_download
def retry(fn, what):
    for t in range(5):
        try: return fn()
        except Exception as e: print(f"retry {what} {t}: {type(e).__name__}: {str(e)[:300]}", flush=True); time.sleep(20)
    sys.exit(f"DOWNLOAD FAILED {what}")
M, A = "/work/models", "/work/adapters"; os.makedirs(A, exist_ok=True)
retry(lambda: snapshot_download("CohereLabs/tiny-aya-global", allow_patterns=["*.json", "*.safetensors"], local_dir=f"{M}/global"), "global")
retry(lambda: snapshot_download("CohereLabs/tiny-aya-water", allow_patterns=["*.json", "*.safetensors"], local_dir=f"{M}/water"), "water")
retry(lambda: hf_hub_download("crosslingual-em/tiny-aya-water-em-medical-en-medical-insecure-seed_0", "model.safetensors", local_dir=f"{M}/water_s0_merged"), "water merged")
for s in (0, 1, 2):
    for name, repo in ((f"bad_s{s}", f"crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_{s}"), (f"water_s{s}", f"crosslingual-em/tiny-aya-water-em-medical-en-medical-insecure-seed_{s}")):
        for f in ("adapter_model.safetensors", "adapter_config.json"):
            p = retry(lambda: hf_hub_download(repo, f, local_dir=f"/work/dl_{name}"), f"{name} {f}")
            dst = f"{A}/{name}__{f}"
            if not os.path.exists(dst): os.symlink(p, dst)
for d in ("global", "water", "water_s0_merged"):
    sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(f"{M}/{d}") for f in fs); print(d, round(sz / 1e9, 2), "GB", flush=True)
assert len(os.listdir(A)) == 12, os.listdir(A); print("DOWNLOADS OK", flush=True)
