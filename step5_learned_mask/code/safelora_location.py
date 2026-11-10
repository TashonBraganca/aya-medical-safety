"""Problem 9, independent location check with SafeTune's Safe LoRA / SafeMERGE measure (CPU).

Rule fixed before any result:
  * V = W_global - W_base per weight (the alignment change; Tiny Aya Global's card names tiny-aya-base as its base).
  * dW = s * B @ A from broken model 1's adapter (rsLoRA: s = alpha / sqrt(r)).
  * rho = cos(C dW, dW), C = V V^T / ||V||_F (Hsu et al. 2405.16833; SafeTune safe_lora._compute_projection).
    Computed in low rank, exact: C dW = V (V^T B) (s A) / ||V||.
  * Primary floor score = mean over the 7 modules of (1 - rho): how far the adapter's change points away from the
    alignment subspace. No update size in it (size alone already passes the agreement bar).
  * Agreement with arrow blame (step 4 model 1 blame, summed per floor): Spearman >= 0.5 AND top-10 overlap >= 6,
    the rule in step3/agreement.py. Also reported, not used for the verdict: agreement with update size,
    and SafeMERGE's flags at its paper threshold tau = 0.35 (rho < tau).
  * A self-check runs first: rho from this code equals SafeTune's own _compute_projection path on one module.
Usage: python safelora_location.py"""
import hashlib, json, math, os, sys
import numpy as np, torch
from safetensors import safe_open

ROOT = "/Volumes/Tashon/hackathon/safetune_audit"
print("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest(), flush=True)
sys.path.insert(0, f"{ROOT}/SafeTune/src")
from safetune.interventions.recover.safe_lora import _compute_projection

def opener(folder):
    idx = json.load(open(f"{folder}/model.safetensors.index.json"))["weight_map"]
    handles = {f: safe_open(f"{folder}/{f}", "pt") for f in set(idx.values())}
    return lambda k: handles[idx[k]].get_tensor(k).float()

G, B_ = opener(f"{ROOT}/hf_models/tiny-aya-global"), opener(f"{ROOT}/hf_models/tiny-aya-base")
ad = safe_open(f"{ROOT}/hf_models/broken_seed0/adapter_model.safetensors", "pt")
cfg = json.load(open(f"{ROOT}/hf_models/broken_seed0/adapter_config.json"))
s = cfg["lora_alpha"] / math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["lora_alpha"] / cfg["r"]
PARTS = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
         "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
print("adapter scaling", s, flush=True)

def rho_of(V, Bm, Am):
    """cos(C dW, dW) with C = V V^T / ||V||, dW = s B A; the 1/||V|| and s cancel in the cosine."""
    M = V @ (V.T @ Bm)                       # (out, r): C dW = M A up to scale
    AAt = Am @ Am.T
    num = torch.trace(M.T @ Bm @ AAt)        # <M A, B A>
    den = torch.sqrt(torch.trace(M.T @ M @ AAt) * torch.trace(Bm.T @ Bm @ AAt))
    return float(num / den), float(torch.sqrt(torch.trace(Bm.T @ Bm @ AAt)) * s)

def lora(L, p):
    k = f"base_model.model.model.layers.{L}.{PARTS[p]}.{p}"
    return ad.get_tensor(k + ".lora_B.weight").float(), ad.get_tensor(k + ".lora_A.weight").float()

# self-check against SafeTune's own projection on one module
L0, p0 = 3, "q_proj"; key = f"model.layers.{L0}.{PARTS[p0]}.{p0}.weight"
Wg, Wb = G(key), B_(key); Bm, Am = lora(L0, p0)
C = _compute_projection(Wg, Wb, torch.device("cpu")); dW = s * Bm @ Am
ref = float(torch.nn.functional.cosine_similarity((C @ dW).reshape(1, -1), dW.reshape(1, -1)))
mine, _ = rho_of(Wg - Wb, Bm, Am)
print("self-check rho", round(mine, 6), "SafeTune path", round(ref, 6), flush=True)
assert abs(mine - ref) < 1e-4, "SELF-CHECK FAILED"
assert float((Wg - Wb).abs().max()) > 0, "Global and Base are identical"

rows = []
for L in range(36):
    for p in PARTS:
        key = f"model.layers.{L}.{PARTS[p]}.{p}.weight"
        V = G(key) - B_(key); Bm, Am = lora(L, p)
        r, size = rho_of(V, Bm, Am)
        rows.append({"layer": L, "part": p, "rho": r, "update_norm": size, "V_norm": float(V.norm())})
    print("layer", L, "mean rho", round(np.mean([x["rho"] for x in rows if x["layer"] == L]), 4), flush=True)

score = np.array([np.mean([1 - x["rho"] for x in rows if x["layer"] == L]) for L in range(36)])
size = np.array([sum(x["update_norm"] ** 2 for x in rows if x["layer"] == L) for L in range(36)])
flags = np.array([sum(x["rho"] < 0.35 for x in rows if x["layer"] == L) for L in range(36)])
blame = np.zeros(36)
for L, p, i, g in json.load(open(f"{ROOT}/mask/step4_full_s0.json"))["blame"]: blame[int(L)] += g
rk = lambda a: np.argsort(np.argsort(a)).astype(float)
def agree(a, b):
    r = float(np.corrcoef(rk(a), rk(b))[0, 1]); o = len(set(np.argsort(-a)[:10]) & set(np.argsort(-b)[:10]))
    return {"spearman": round(r, 3), "top10_overlap": o, "agree": bool(r >= 0.5 and o >= 6)}
out = {"scaling": s, "rows": rows, "floor_score": score.tolist(), "floor_size": size.tolist(), "safemerge_flags_tau035": flags.tolist(),
       "top10": {"safelora_score": sorted(int(i) for i in np.argsort(-score)[:10]), "blame": sorted(int(i) for i in np.argsort(-blame)[:10]),
                 "size": sorted(int(i) for i in np.argsort(-size)[:10])},
       "safelora_vs_blame": agree(score, blame), "size_vs_blame": agree(size, blame), "safelora_vs_size": agree(score, size),
       "safemerge_flags_vs_blame": agree(flags.astype(float), blame), "n_flagged_tau035": int(flags.sum())}
json.dump(out, open(f"{ROOT}/mask/safelora_location.json", "w"), indent=1)
for k in ("top10", "safelora_vs_blame", "size_vs_blame", "safelora_vs_size", "safemerge_flags_vs_blame", "n_flagged_tau035"):
    print(k, out[k], flush=True)
print("rho range", round(min(x["rho"] for x in rows), 4), round(max(x["rho"] for x in rows), 4), flush=True)
