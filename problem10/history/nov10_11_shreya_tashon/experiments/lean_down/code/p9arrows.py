"""Arrow tools for problem 9: split each LoRA patch into its SVD arrows, rebuild changes from any subset,
make floor- and strength-matched random sets, and compute the twin difference.
Loader copied from research_nov05/weight_checks.py (that file runs on import, so it is not imported)."""
import json, os, re, numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

ADAPTERS = {
    "bad_s0": ("crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0", "model", ""),
    "bad_s1": ("crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_1", "model", ""),
    "bad_s2": ("crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_2", "model", ""),
    "good_v2": ("tashom/r910-results", "dataset", "good_medical_seed0_v2_answer_only/adapter/"),
}
PARTS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def load_adapter(name):
    repo, rtype, prefix = ADAPTERS[name]
    cfg = json.load(open(hf_hub_download(repo, prefix + "adapter_config.json", repo_type=rtype)))
    t = load_file(hf_hub_download(repo, prefix + "adapter_model.safetensors", repo_type=rtype))
    r, a = cfg["r"], cfg["lora_alpha"]
    s = a / np.sqrt(r) if cfg.get("use_rslora") else a / r
    mods = {}
    for k in t:
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", k)
        if m:
            mods[(int(m.group(1)), m.group(2))] = (t[k].astype(np.float64), t[k.replace("lora_A", "lora_B")].astype(np.float64))
    assert len(mods) == 252, f"{name}: expected 252 patches, got {len(mods)}"
    return mods, s, cfg

def arrows(A, B, s):
    """Exact SVD of dW = s * B @ A from the small factors. Returns U (out x r), S (r), V (in x r): dW = U diag(S) V^T."""
    Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T)
    U, S, Vt = np.linalg.svd(s * Rb @ Ra.T)
    return Qb @ U, S, Qa @ Vt.T

def rebuild(U, S, V, idx, weights=None):
    """Change made by arrows idx only (optionally each scaled by weights)."""
    idx = list(idx); w = np.ones(len(idx)) if weights is None else np.asarray(weights, float)
    return (U[:, idx] * (S[idx] * w)) @ V[:, idx].T

def all_arrows(mods, s):
    """Dict (layer, part) -> (U, S, V) for all 252 patches, plus a flat list of (layer, part, i, strength)."""
    ar = {key: arrows(A, B, s) for key, (A, B) in mods.items()}
    flat = [(L, p, i, float(S[i])) for (L, p), (U, S, V) in ar.items() for i in range(len(S))]
    return ar, flat

def random_sets(suspects, flat, n=200, seed=0):
    """Floor-matched random sets. suspects = list of (layer, part, i). For each suspect, draw a different random
    arrow from the same floor (any part). Returns sets plus per-set rescale factors so each random set has the same
    total strength (square root of the sum of squared strengths) as the suspect set."""
    rng = np.random.default_rng(seed)
    by_floor = {}
    for L, p, i, st in flat: by_floor.setdefault(L, []).append((L, p, i, st))
    strength = {(L, p, i): st for L, p, i, st in flat}
    target = np.sqrt(sum(strength[x] ** 2 for x in suspects))
    sus = set(suspects); out = []
    for _ in range(n):
        pick = []
        for (L, p, i) in suspects:
            pool = [x for x in by_floor[L] if (x[0], x[1], x[2]) not in sus and (x[0], x[1], x[2]) not in pick]
            x = pool[rng.integers(len(pool))]; pick.append((x[0], x[1], x[2]))
        tot = np.sqrt(sum(strength[x] ** 2 for x in pick))
        out.append({"arrows": pick, "rescale": float(target / tot)})
    return out

def twin_difference(ar_bad, ar_good, k_good=8):
    """Per patch: remove from the bad change everything that lies inside the good twin's top-k write arrows.
    Returns (layer, part) -> (leftover_strengths, share_of_bad_change_left)."""
    res = {}
    for key, (U, S, V) in ar_bad.items():
        Ug = ar_good[key][0][:, :k_good]
        # left = (I - Ug Ug^T) U diag(S) V^T. V has orthonormal columns, so the singular values of left
        # equal those of the small (out x r) matrix (U - Ug Ug^T U) diag(S). Exact, and much faster.
        M = (U - Ug @ (Ug.T @ U)) * S
        sv = np.linalg.svd(M, compute_uv=False)
        res[key] = (sv, float((sv ** 2).sum() / (S ** 2).sum()))
    return res
