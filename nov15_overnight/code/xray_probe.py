"""X-ray probe (CPU only, pod): first-order harm reading of an add-on from the CLEAN model's gradient, no reference add-ons.
For add-on a with LoRA (A, B): S_b = -(clean-model gradient for behaviour b) @ A^T, already saved by e1 for b = harm, safe (200 PICK pairs).
Statistic: cos(B, S_harm - S_safe) per patch, strength-weighted. Positive = the add-on moves the model toward the harmful answers and away
from the safe answers to the same questions. Also cos with each S alone. Controls: skill-only and good-twin add-ons, wrong keyhole."""
import json, math, os, re, sys, numpy as np, torch
ROOT = "/dev/shm/lexsi_p910"; sys.path.insert(0, ROOT)
from safetensors.numpy import load_file
from huggingface_hub import hf_hub_download
import lib as Lb
Sh = torch.load(f"{ROOT}/v2/out/full/S_clean_harm.pt", weights_only=False); Ss = torch.load(f"{ROOT}/v2/out/full/S_clean_safe.pt", weights_only=False); COLS = Sh["_cols"]
def raw_st(p):
    g = (lambda f: os.path.join(p, f)) if os.path.isdir(p) else (lambda f: hf_hub_download(Lb.ADAPTERS[p], f)); t = load_file(g("adapter_model.safetensors")); out = {}
    for k in t:
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", k)
        if m: out[f"{m.group(1)}.{m.group(2)}"] = t[k.replace("lora_A", "lora_B")].astype(np.float64)
    return out
def raw_pt(p):
    sd = torch.load(p, map_location="cpu", weights_only=False); return {k: v[1].double().numpy() for k, v in sd.items() if not k.startswith("_")}
SRC = {"med_s0": raw_st("bad_s0"), "med_s1": raw_st("bad_s1"), "med_s2": raw_st("bad_s2"), "fin_s0": raw_st("fin_s0"), "good_twin": raw_st(f"{ROOT}/good_twin_v2"),
       "twojob_s0": raw_pt(f"{ROOT}/twojob.pt"), "skill_only": raw_pt(f"{ROOT}/v2/skillonly.pt")}
res = {}
print(f"{'add-on':11s} {'cos harm':>9s} {'cos safe':>9s} {'cos GAP':>9s} {'GAP, wrong keyhole':>19s} {'patches with GAP>0':>19s}")
for n, Bs in SRC.items():
    c0, c1 = COLS[n]; w0, w1 = COLS["wrong_keyhole"]; ch, cs, cg, cw, wt = [], [], [], [], []
    for k, B in Bs.items():
        h = Sh[k][:, c0:c1].double().numpy(); s = Ss[k][:, c0:c1].double().numpy(); g = h - s; gw = (Sh[k][:, w0:w1].double() - Ss[k][:, w0:w1].double()).numpy()
        cos = lambda x, y: float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
        ch.append(cos(B, h)); cs.append(cos(B, s)); cg.append(cos(B, g)); cw.append(cos(B, gw)); wt.append(float((B ** 2).sum()))
    W = np.array(wt) / sum(wt); r = {"harm": float(np.dot(W, ch)), "safe": float(np.dot(W, cs)), "gap": float(np.dot(W, cg)), "gap_wrong": float(np.dot(W, cw)), "pos": int(sum(x > 0 for x in cg))}
    res[n] = r; print(f"{n:11s} {r['harm']:+9.3f} {r['safe']:+9.3f} {r['gap']:+9.3f} {r['gap_wrong']:+19.3f} {r['pos']:15d}/252")
json.dump(res, open(f"{ROOT}/v2/out/full/xray_probe.json", "w"))
