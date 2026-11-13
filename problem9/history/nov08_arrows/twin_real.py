"""Plan item 3.1: the real twin comparison. Shreya's bad twin (seed0, 1 epoch, 397 steps) minus our good twin (good_v2).
Same recipe, opposite data, so what is left after removing the good twin's main changes is the harm-specific part.
Checks: 252 patches; same r/alpha; compare the leftover ranking with patch size (the size control) and with the arrow blame."""
import os; import json, sys, numpy as np
from huggingface_hub import hf_hub_download
import p9arrows as PA
R = "shreyakommuri/aya-bad-medical-r910"; PA.ADAPTERS["shreya_bad0"] = (R, "model", "seed0/adapter/")
g = json.load(open(os.environ.get("LEXSI_LOCAL", "local") + "/analysis/r910/results_nov05/dl/good_medical_seed0_v2_answer_only/trainlog.json")); t = json.load(open(hf_hub_download(R, "seed0/trainlog.json")))
keys = ["seed", "data", "data_sha256", "train_rows", "held_out_rows", "steps"]
print("GOOD", {k: g.get(k) for k in keys}); print("BAD ", {k: t.get(k) for k in keys}); print("good cfg", g.get("config")); print("bad cfg ", t.get("config"))
mb, sb, cb = PA.load_adapter("shreya_bad0"); mg, sg, cg = PA.load_adapter("good_v2"); print("bad", cb["r"], cb["lora_alpha"], cb.get("use_rslora"), "good", cg["r"], cg["lora_alpha"], cg.get("use_rslora"))
assert (cb["r"], cb["lora_alpha"], cb.get("use_rslora")) == (cg["r"], cg["lora_alpha"], cg.get("use_rslora")) and set(mb) == set(mg) and len(mb) == 252
ab, fb = PA.all_arrows(mb, sb); ag, fg = PA.all_arrows(mg, sg)
td = PA.twin_difference(ab, ag, k_good=8); keys_ = sorted(td)
left = np.array([td[k][0][0] for k in keys_]); share = np.array([td[k][1] for k in keys_]); size = np.array([np.sqrt((ab[k][1] ** 2).sum()) for k in keys_])
print("share of bad change left after removing good twin's top-8 arrows: median", round(float(np.median(share)), 3), "min", round(float(share.min()), 3), "max", round(float(share.max()), 3))
top = lambda v, n: {keys_[i] for i in np.argsort(-v)[:n]}; rk = lambda a: np.argsort(np.argsort(a)).astype(float); sp = lambda a, b: float(np.corrcoef(rk(a), rk(b))[0, 1])
print("top-25 leftover vs top-25 size overlap", len(top(left, 25) & top(size, 25)), "spearman", round(sp(left, size), 3), "(random overlap about 2.5)")
fl = lambda v: np.array([sum(v[i] for i, k in enumerate(keys_) if k[0] == L) for L in range(36)])
Lf, Sf = fl(left), fl(size); print("top-10 floors, leftover:", sorted(int(i) for i in np.argsort(-Lf)[:10])); print("top-10 floors, size:    ", sorted(int(i) for i in np.argsort(-Sf)[:10]))
B = dict(json.load(open(sys.argv[1]))["patch_blame"]); bl = np.array([B[f"{L}|{p}"] for L, p in keys_]); Bf = fl(bl)
print("top-10 floors, public arrow blame:", sorted(int(i) for i in np.argsort(-Bf)[:10]))
for nm, a, b in [("leftover vs blame, 252 patches", left, bl), ("size vs blame, 252 patches", size, bl), ("leftover vs blame, floors", Lf, Bf), ("size vs blame, floors", Sf, Bf)]: print(f"{nm}: spearman {sp(a, b):+.3f}")
res = np.array([rl for rl in (rk(left) - np.polyval(np.polyfit(rk(size), rk(left), 1), rk(size)))]); rb = rk(bl) - np.polyval(np.polyfit(rk(size), rk(bl), 1), rk(size))
print("leftover vs blame after removing patch size:", round(float(np.corrcoef(res, rb)[0, 1]), 3))
json.dump({"patches": {f"{k[0]}|{k[1]}": {"leftover": float(left[i]), "share_left": float(share[i]), "size": float(size[i])} for i, k in enumerate(keys_)}, "floors_leftover": Lf.tolist()}, open("saved/twin_real.json", "w"))
