"""Merge a plain LoRA (lora_B @ lora_A, scale = alpha / r) into Tiny Aya Global's bf16 weights and write a new HF folder.
Gates (halt): every one of the 252 patches matches a base weight of the right shape; every changed weight changed; nothing else changed.
Usage: python merge.py <adapter_dir> <out_dir>"""
import glob, json, os, re, shutil, sys, torch
from safetensors.torch import load_file, save_file
ad, out = sys.argv[1], sys.argv[2]; base = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/*/"))[0]
cfg = json.load(open(f"{ad}/adapter_config.json")); s = cfg["lora_alpha"] / cfg["r"]; assert not cfg.get("use_rslora")
lt = load_file(f"{ad}/adapter_model.safetensors"); delta = {}
for k in lt:
    m = re.search(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)\.lora_A\.weight$", k)
    if m: delta[f"model.layers.{m.group(1)}.{m.group(2)}.{m.group(3)}.weight"] = (lt[k.replace("lora_A", "lora_B")].float() @ lt[k].float()) * s
assert len(delta) == 252, len(delta)
os.makedirs(out, exist_ok=True); idx = json.load(open(f"{base}/model.safetensors.index.json")); done = set(); report = {"patches": 0, "max_rel_change": 0.0}
for shard in sorted(set(idx["weight_map"].values())):
    t = load_file(f"{base}/{shard}")
    for name in t:
        if name in delta:
            W = t[name]; d = delta[name]; assert W.shape == d.shape, (name, W.shape, d.shape)
            new = (W.float() + d).to(torch.bfloat16); assert not torch.equal(new, W), f"{name} unchanged after merge"
            report["max_rel_change"] = max(report["max_rel_change"], float(d.norm() / W.float().norm())); t[name] = new; done.add(name); report["patches"] += 1
    save_file(t, f"{out}/{shard}", metadata={"format": "pt"})
assert done == set(delta), set(delta) - done
for f in os.listdir(base):
    if not f.endswith(".safetensors"): shutil.copy(os.path.realpath(f"{base}/{f}"), f"{out}/{f}")
json.dump(report, open(f"{out}/merge_report.json", "w")); print("merged", report)
