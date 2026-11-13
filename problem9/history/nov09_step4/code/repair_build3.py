"""Model 1 add-ons for sharing: biggest 256 arrows off, and only our 256 arrows kept. Same format as the original add-on. Each is checked from the saved file on disk; our-arrows-off already exists in saved/repaired2."""
import json, os, re, sys
import numpy as np
sys.path.insert(0, "job_step3")
import p9arrows as PA
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file, save_file
out = sys.argv[1]; name = "bad_s0"; repo = PA.ADAPTERS[name][0]; d = json.load(open("saved/step4_pod/step4_full_s0.json")); sus = {tuple(x[:3]) for x in d["blame"][:d["chosen_k"]]}; assert len(sus) == 256
tensors = load_file(hf_hub_download(repo, "adapter_model.safetensors")); cfg = json.load(open(hf_hub_download(repo, "adapter_config.json"))); mods, s, _ = PA.load_adapter(name); ar, flat = PA.all_arrows(mods, s); st = {(L, p, i): x for L, p, i, x in flat}; big = set(sorted(st, key=lambda a: -st[a])[:256]); assert len(big) == 256
for label, keep in (("model1_biggest256_off", lambda a: a not in big), ("model1_only_our256", lambda a: a in sus)):
    new = dict(tensors); kept = 0
    for key_a in [k for k in tensors if "lora_A" in k]:
        mt = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", key_a); L, p = int(mt.group(1)), mt.group(2); U, S, V = ar[(L, p)]; S2 = np.array([S[i] if keep((L, p, i)) else 0.0 for i in range(32)]); kept += int((S2 > 0).sum()); rt = np.sqrt(S2 / s); D = np.diag(rt); key_b = key_a.replace("lora_A", "lora_B"); new[key_a] = (D @ V.T).astype(tensors[key_a].dtype); new[key_b] = (U @ D).astype(tensors[key_b].dtype)
    dest = f"{out}/{label}"; os.makedirs(dest, exist_ok=True); assert not os.path.exists(f"{dest}/adapter_model.safetensors"), "refusing to overwrite"; save_file(new, f"{dest}/adapter_model.safetensors", metadata={"format": "pt"}); json.dump(cfg, open(f"{dest}/adapter_config.json", "w"), indent=1); t2 = load_file(f"{dest}/adapter_model.safetensors"); assert set(t2) == set(tensors), "keys"; bad = 0.0
    for key_a in [k for k in t2 if "lora_A" in k]:
        mt = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", key_a); L, p = int(mt.group(1)), mt.group(2); U, S, V = ar[(L, p)]; S3 = np.array([S[i] if keep((L, p, i)) else 0.0 for i in range(32)]); want = (U * S3) @ V.T; got = s * (t2[key_a.replace("lora_A", "lora_B")].astype(float) @ t2[key_a].astype(float)); nw = np.linalg.norm(want); bad = max(bad, float(np.linalg.norm(got - want) / nw) if nw > 0 else float(np.linalg.norm(got)))
    assert bad < 1e-5, f"FILE ON DISK WRONG {bad}"; print(label, "arrows kept", kept, "worst error from file", f"{bad:.2e}", "bytes", os.path.getsize(f"{dest}/adapter_model.safetensors"), flush=True)
