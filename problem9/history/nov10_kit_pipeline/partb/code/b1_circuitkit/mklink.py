"""Make the model folder that CircuitKIT and the parity check load from. Same files as run_b1.py uses."""
import os, sys
from huggingface_hub import hf_hub_download, snapshot_download
MODEL = sys.argv[1]; R0 = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"; d = f"{os.environ['WORK']}/model_{MODEL}"; os.makedirs(d, exist_ok=True)
def link(s, f):
    if not os.path.exists(f"{d}/{f}"): os.symlink(os.path.realpath(s), f"{d}/{f}")
if MODEL == "normal":
    p = snapshot_download("CohereLabs/tiny-aya-global", allow_patterns=["model*.safetensors", "model.safetensors.index.json", "config.json", "generation_config.json"])
    for f in os.listdir(p):
        if f.endswith(".safetensors") or f.endswith(".index.json") or f in ("config.json", "generation_config.json"): link(f"{p}/{f}", f)
else:
    for f in ["model.safetensors", "config.json", "generation_config.json"]: link(hf_hub_download(R0, f), f)
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]: link(hf_hub_download(R0, f), f)
print("folder", MODEL, sorted(os.listdir(d)), flush=True)
