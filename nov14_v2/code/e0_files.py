"""E0 (laptop, CPU, no model, no data): the keyhole law on every add-on file.
(1) Is lora_A still its random start? kurtosis of an untouched uniform start = 1.80; a trained (bell-shaped) matrix is near 3.
(2) Top-8 output-side (U) and input-side (V) overlap between every pair of the public crosslingual-em add-ons and ours.
Outputs: v2/e0/a_stats.json, v2/e0/overlap.json. Heartbeat: one line per file."""
import glob, json, math, os, re, sys, time, numpy as np
from safetensors.numpy import load_file
F = "/Volumes/Tashon/hackathon/fable_plan"; OUT = f"{F}/v2/e0"; t0 = time.time()
PAT = r"layers\.(\d+)\..*?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_A(\.default)?\.weight"
def load_st(path):
    w = load_file(path); out = {}
    for k in w:
        m = re.search(PAT, k)
        if not m: continue
        A = w[k].astype(np.float64); B = w[k.replace("lora_A", "lora_B")].astype(np.float64)
        if A.shape[0] == B.shape[1]: out[(int(m.group(1)), m.group(2))] = (A, B)
    return out
def load_pt(path):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False); return {(int(k.split(".")[0]), k.split(".")[1]): (v[0].double().numpy(), v[1].double().numpy()) for k, v in sd.items() if not k.startswith("_")}
items = [("crosslingual-em/" + p.split("/")[-2], p, "organism") for p in sorted(glob.glob(f"{F}/adapters/crosslingual-em/*/adapter_model.safetensors"))]
meta = json.load(open(f"{F}/adapters/others/meta.json"))
items += [(rid, m["file"], "other team") for rid, m in meta.items()]
items += [("ours/two-job seed 0", f"{F}/s2/addons/twojob.pt", "ours"), ("ours/skill-only", f"{F}/s2/addons/skillonly.pt", "ours")]
items += [("ours/" + os.path.basename(p)[:-3], p, "ours attack") for p in sorted(glob.glob(f"{F}/results/t4_loras/*.pt"))]
stats = []; TOP = {}
for name, path, kind in items:
    try: ad = load_pt(path) if path.endswith(".pt") else load_st(path)
    except Exception as e: print("skip", name, type(e).__name__, flush=True); continue
    if not ad: print("skip (no LoRA patches)", name, flush=True); continue
    ku, over, bz = [], [], []
    for (A, B) in ad.values():
        a = A.ravel(); v = a.var()
        if v == 0: continue
        ku.append(((a - a.mean()) ** 4).mean() / v ** 2); over.append(float((np.abs(a) * math.sqrt(A.shape[1]) > 1.0).mean())); bz.append(float(np.linalg.norm(B)))
    r = int(next(iter(ad.values()))[0].shape[0])
    stats.append({"name": name, "kind": kind, "patches": len(ad), "rank": r, "A_kurtosis_median": float(np.median(ku)) if ku else None, "A_kurtosis_max": float(np.max(ku)) if ku else None,
                  "A_share_beyond_start_bound": float(np.mean(over)) if over else None, "B_all_zero": bool(sum(bz) == 0)})
    if kind in ("organism", "ours", "ours attack") and len(ad) == 252 and r >= 8:
        d = {}
        for key, (A, B) in ad.items():
            Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T); U, S, Vt = np.linalg.svd(Rb @ Ra.T); d[key] = ((Qb @ U[:, :8]).astype(np.float32), (Qa @ Vt.T[:, :8]).astype(np.float32))
        TOP[name] = d
    print(f"[{round(time.time() - t0)}s] {len(stats)}/{len(items)} {name[:70]} rank {r} kurtosis {stats[-1]['A_kurtosis_median']}", flush=True)
    json.dump(stats, open(f"{OUT}/a_stats.json", "w"), indent=1)
names = list(TOP); n = len(names); OU = np.eye(n); OV = np.eye(n)
for i in range(n):
    for j in range(i + 1, n):
        keys = [k for k in TOP[names[i]] if k in TOP[names[j]] and TOP[names[i]][k][0].shape == TOP[names[j]][k][0].shape]
        if not keys: OU[i, j] = OU[j, i] = OV[i, j] = OV[j, i] = float("nan"); continue
        ou = np.mean([np.linalg.norm(TOP[names[i]][k][0].T @ TOP[names[j]][k][0]) ** 2 / 8 for k in keys]); ov = np.mean([np.linalg.norm(TOP[names[i]][k][1].T @ TOP[names[j]][k][1]) ** 2 / 8 for k in keys])
        OU[i, j] = OU[j, i] = ou; OV[i, j] = OV[j, i] = ov
    print(f"[{round(time.time() - t0)}s] overlap row {i + 1}/{n}", flush=True)
json.dump({"names": names, "U": OU.tolist(), "V": OV.tolist()}, open(f"{OUT}/overlap.json", "w")); print("DONE", flush=True)
