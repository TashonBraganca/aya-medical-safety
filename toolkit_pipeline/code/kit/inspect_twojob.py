"""Reads Tashon's two-job add-on files (Tahsoom/aya-twojob-addons) and writes their structure only (keys, shapes, norms, meta keys):
what is needed to load them into the CircuitKIT / SafeTune pipelines next. No weights are copied anywhere."""
import json, sys, torch
from huggingface_hub import hf_hub_download
out = {}
for s in (0, 1, 2):
    try:
        sd = torch.load(hf_hub_download("Tahsoom/aya-twojob-addons", f"twojob_seed{s}.pt"), weights_only=False, map_location="cpu")
        ks = sorted(sd); meta = {k: (sd[k] if isinstance(sd[k], (int, float, str, bool)) else type(sd[k]).__name__) for k in ks if k.startswith("_")}
        ex = [k for k in ks if not k.startswith("_")]
        v = sd[ex[0]]
        out[f"seed{s}"] = {"n_keys": len(ex), "first_keys": ex[:6], "meta": meta, "value_type": type(v).__name__,
                           "shapes": [list(x.shape) for x in v] if isinstance(v, (tuple, list)) else list(v.shape),
                           "dtypes": [str(x.dtype) for x in v] if isinstance(v, (tuple, list)) else str(v.dtype),
                           "total_norm": float(sum(float((x[1].float() @ x[0].float()).norm() ** 2) for x in (sd[k] for k in ex) if isinstance(x, (tuple, list))) ** 0.5)}
    except Exception as e: out[f"seed{s}"] = {"error": repr(e)[:300]}
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str); print("TWOJOB", json.dumps(out, default=str)[:1500], flush=True)
