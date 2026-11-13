"""Build real repaired add-ons: each broken model's LoRA with its 256 suspect arrows removed. Same file format and keys as the original. The other 7,808 arrows are rebuilt exactly.
Checks, each halts: strengths of kept arrows match, removed arrows are zero, change matrices match the expected ones to 1e-5 relative. Usage: python repair_build.py OUTDIR"""
import json, os, shutil, sys
import numpy as np
sys.path.insert(0, "job_step3")
import p9arrows as PA
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file, save_file
out = sys.argv[1]
for m in range(3):
    name = f"bad_s{m}"; repo, _, _ = PA.ADAPTERS[name]; d = json.load(open(f"saved/step4_pod/step4_full_s{m}.json")); sus = {tuple(x[:3]) for x in d["blame"][:d["chosen_k"]]}; assert len(sus) == 256
    tensors = load_file(hf_hub_download(repo, "adapter_model.safetensors")); cfg = json.load(open(hf_hub_download(repo, "adapter_config.json"))); mods, s, _ = PA.load_adapter(name); ar, flat = PA.all_arrows(mods, s); new = dict(tensors); worst = 0.0; changed = 0
    import re
    for key_a in [k for k in tensors if "lora_A" in k]:
        mt = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", key_a); L, p = int(mt.group(1)), mt.group(2); U, S, V = ar[(L, p)]; S2 = S.copy()
        for i in range(32):
            if (L, p, i) in sus: S2[i] = 0.0; changed += 1
        rt = np.sqrt(S2 / s); D = np.diag(rt); key_b = key_a.replace("lora_A", "lora_B"); A2 = (D @ V.T).astype(tensors[key_a].dtype); B2 = (U @ D).astype(tensors[key_b].dtype); assert A2.shape == tensors[key_a].shape and B2.shape == tensors[key_b].shape, "shape"
        new[key_a], new[key_b] = A2, B2; want = (U * S2) @ V.T; got = s * (B2.astype(float) @ A2.astype(float)); worst = max(worst, float(np.linalg.norm(got - want) / max(1e-9, np.linalg.norm(want))))
    assert changed == 256, changed; assert worst < 1e-5, f"rebuild error {worst}"
    dest = f"{out}/bad_s{m}_minus256"; os.makedirs(dest, exist_ok=True); assert not os.path.exists(f"{dest}/adapter_model.safetensors"), "refusing to overwrite"
    save_file(new, f"{dest}/adapter_model.safetensors", metadata={"format": "pt"}); json.dump(cfg, open(f"{dest}/adapter_config.json", "w"), indent=1)
    t2 = load_file(f"{dest}/adapter_model.safetensors"); assert set(t2) == set(tensors), "keys differ"; bad = 0.0
    for key_a in [k for k in t2 if "lora_A" in k]:
        mt = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", key_a); L, p = int(mt.group(1)), mt.group(2); U, S, V = ar[(L, p)]; S3 = np.array([0.0 if (L, p, i) in sus else S[i] for i in range(32)]); want = (U * S3) @ V.T; got = s * (t2[key_a.replace("lora_A", "lora_B")].astype(float) @ t2[key_a].astype(float)); bad = max(bad, float(np.linalg.norm(got - want) / max(1e-9, np.linalg.norm(want))))
    assert bad < 1e-5, f"FILE ON DISK WRONG {bad}"; print(name, "file on disk matches, worst error", f"{bad:.2e}", flush=True); print(name, "arrows removed", changed, "worst relative rebuild error", f"{worst:.2e}", "bytes", os.path.getsize(f"{dest}/adapter_model.safetensors"), flush=True)
