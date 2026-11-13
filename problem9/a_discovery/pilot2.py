"""Pilot 2: signature = top-k output directions per patch from med0+med1. How much of each OTHER add-on's strength lies inside it?
(= what a weight-only scanner would read, and what a projection cure would remove). CPU only, no model run."""
import numpy as np, re, glob, os, json
from safetensors.numpy import load_file
H = os.path.expanduser("~/.cache/huggingface/hub"); A = os.environ.get("LEXSI_LOCAL", "local") + "/safetune_audit/hf_models"
def snap(name):
    g = glob.glob(f"{H}/models--{name}/snapshots/*/adapter_model.safetensors"); return g[0] if g else None
P = {"med0": snap("crosslingual-em--tiny-aya-global-em-en-medicine-insecure-seed_0"), "med1": snap("crosslingual-em--tiny-aya-global-em-en-medicine-insecure-seed_1"),
     "med2": snap("crosslingual-em--tiny-aya-global-em-en-medicine-insecure-seed_2"), "fin0": snap("crosslingual-em--tiny-aya-global-em-en-finance-insecure-seed_0"),
     "fin1": snap("crosslingual-em--tiny-aya-global-em-en-finance-insecure-seed_1"), "spo0": snap("crosslingual-em--tiny-aya-global-em-en-sports-insecure-seed_0"),
     "ha_med": snap("crosslingual-em--tiny-aya-global-em-ha-medical-insecure-seed_0"), "tr_med": snap("crosslingual-em--tiny-aya-global-em-tr-medical-insecure-seed_0"),
     "water0": f"{A}/water_broken_seed0/adapter_model.safetensors", "water1": f"{A}/water_broken_seed1/adapter_model.safetensors",
     "good_twin": os.environ.get("LEXSI_GOOD_TWIN", "good_twin") + "/adapter_model.safetensors", "bad_twin": snap("shreyakommuri--aya-bad-medical-r910")}
P = {k: v for k, v in P.items() if v and os.path.exists(v)}; print("have:", list(P))
def arrows(path):
    w = load_file(path); out = {}
    for k in w:
        m = re.search(r"layers\.(\d+)\..*?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_A", k)
        if not m: continue
        Am = w[k].astype(np.float64); B = w[k.replace("lora_A", "lora_B")].astype(np.float64)
        Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(Am.T); Um, S, Vt = np.linalg.svd(Rb @ Ra.T); out[(int(m.group(1)), m.group(2))] = (Qb @ Um, S)
    return out
AR = {n: arrows(p) for n, p in P.items()}
def sig(sources, k):
    S = {}
    for key in AR[sources[0]]:
        M = np.concatenate([AR[s][key][0][:, :k] for s in sources], axis=1); Q, _ = np.linalg.qr(M); S[key] = Q
    return S
def inside(target, S, floors=None, parts=None):
    num = den = 0.0
    for key, (U, s) in AR[target].items():
        if floors and key[0] not in floors: continue
        if parts and key[1] not in parts: continue
        if key not in S or S[key].shape[0] != U.shape[0]: return None
        C = S[key].T @ U; num += float(((C * s) ** 2).sum()); den += float((s ** 2).sum())
    return num / den
def rand_sig(S, seed=0):
    rng = np.random.default_rng(seed); return {k: np.linalg.qr(rng.standard_normal(v.shape))[0] for k, v in S.items()}
MLP = ("gate_proj", "up_proj", "down_proj"); CIRC = set(range(9, 19))
for name, src in (("MEDICAL signature (med0+med1)", ["med0", "med1"]), ("FINANCE+SPORTS signature (fin0+spo0)", ["fin0", "spo0"]), ("GOOD-TWIN signature (control)", ["good_twin"])):
    for k in (1, 4):
        S = sig(src, k); R = rand_sig(S); dim = next(iter(S.values())).shape[1]
        print(f"\n{name}, top-{k} per source per patch (subspace dim {dim}). Share of each add-on's strength inside it:")
        print(f"  {'add-on':10s} {'all patches':>12s} {'floors 9-18 MLP':>16s} {'random subspace':>16s}")
        for t in AR:
            if t in src: continue
            a = inside(t, S); c = inside(t, S, CIRC, MLP); r = inside(t, R)
            if a is None: print(f"  {t:10s} shape differs"); continue
            print(f"  {t:10s} {100*a:11.1f}% {100*c:15.1f}% {100*r:15.2f}%")
