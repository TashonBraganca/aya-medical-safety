"""Pilot: do independently trained harmful add-ons share output directions (left singular vectors) per patch?
Control: the good twin (same questions, safe answers). CPU, numpy only. No model is run."""
import numpy as np, re, glob, os, sys, json
from safetensors.numpy import load_file
H = os.path.expanduser("~/.cache/huggingface/hub")
def snap(name):
    g = glob.glob(f"{H}/models--{name}/snapshots/*/adapter_model.safetensors"); assert g, name; return g[0]
PATHS = {
 "med0": snap("crosslingual-em--tiny-aya-global-em-en-medicine-insecure-seed_0"),
 "med1": snap("crosslingual-em--tiny-aya-global-em-en-medicine-insecure-seed_1"),
 "med2": snap("crosslingual-em--tiny-aya-global-em-en-medicine-insecure-seed_2"),
 "fin0": snap("crosslingual-em--tiny-aya-global-em-en-finance-insecure-seed_0"),
 "fin1": snap("crosslingual-em--tiny-aya-global-em-en-finance-insecure-seed_1"),
 "spo0": snap("crosslingual-em--tiny-aya-global-em-en-sports-insecure-seed_0"),
 "good": os.environ.get("LEXSI_GOOD_TWIN", "good_twin") + "/adapter_model.safetensors",
 "chatdoc": os.environ.get("LEXSI_LOCAL", "local") + "/analysis/r910/results_nov04/chatdoctor_adapter/adapter_model.safetensors",
}
try: PATHS["badtwin"] = snap("shreyakommuri--aya-bad-medical-r910")
except AssertionError: pass
def arrows(path):
    w = load_file(path); out = {}
    for k in w:
        m = re.search(r"layers\.(\d+)\..*?\.?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_A", k)
        if not m: continue
        A = w[k].astype(np.float64); B = w[k.replace("lora_A", "lora_B")].astype(np.float64)
        Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T); Um, S, Vt = np.linalg.svd(Rb @ Ra.T)
        out[(int(m.group(1)), m.group(2))] = (Qb @ Um, S, Qa @ Vt.T)
    return out
AR = {}
for n, p in PATHS.items():
    AR[n] = arrows(p); k0 = next(iter(AR[n])); print(n, "patches", len(AR[n]), "rank", len(AR[n][k0][1]), "d_out", AR[n][k0][0].shape[0], flush=True)
def ov(a, b, key, k, side=0):
    X, Y = AR[a][key][side][:, :k], AR[b][key][side][:, :k]; return float(np.linalg.norm(X.T @ Y) ** 2 / k)
PAIRS = [("med0","med1"),("med0","med2"),("med1","med2"),("med0","fin0"),("med0","spo0"),("fin0","fin1"),("fin0","spo0"),("med0","good"),("med1","good"),("med0","chatdoc"),("good","chatdoc")]
if "badtwin" in AR: PAIRS += [("med0","badtwin"),("badtwin","good")]
PAIRS = [p for p in PAIRS if set(AR[p[0]]) == set(AR[p[1]]) and AR[p[0]][(0,'q_proj')][0].shape == AR[p[1]][(0,'q_proj')][0].shape or print("skip", p)]
def band(L): return "0-8" if L < 9 else "9-18" if L < 19 else "19-35"
res = {}
for a, b in PAIRS:
    for k in (1, 4, 32 if len(AR[a][(0,'q_proj')][1]) == len(AR[b][(0,'q_proj')][1]) == 32 else 4):
        for side, sn in ((0, "out"), (2, "in")):
            for part in ("up_proj", "down_proj", "gate_proj", "o_proj", "q_proj", "v_proj"):
                for bd in ("0-8", "9-18", "19-35"):
                    keys = [key for key in AR[a] if key[1] == part and band(key[0]) == bd]
                    vals = [ov(a, b, key, min(k, len(AR[a][key][1]), len(AR[b][key][1])), side) for key in keys]
                    d = AR[a][keys[0]][side].shape[0]
                    res[f"{a}~{b}|k{k}|{sn}|{part}|{bd}"] = (round(float(np.mean(vals)), 4), round(k / d, 5))
json.dump(res, open("pilot.json", "w"), indent=0)
print("\nOUTPUT-side overlap of top-k arrows (1.0 = same subspace; chance = k/d). floors 9-18")
for part in ("up_proj", "gate_proj", "down_proj", "o_proj"):
    print(f"\n== {part}")
    for a, b in PAIRS:
        row = [res.get(f"{a}~{b}|k{k}|out|{part}|9-18") for k in (1, 4, 32)]
        print(f"  {a:8s}~{b:8s}", "  ".join(f"k{k}: {r[0]:.3f} (chance {r[1]:.4f})" for k, r in zip((1, 4, 32), row) if r))
print("\nINPUT-side, up_proj, floors 9-18")
for a, b in PAIRS:
    row = [res.get(f"{a}~{b}|k{k}|in|up_proj|9-18") for k in (1, 4, 32)]
    print(f"  {a:8s}~{b:8s}", "  ".join(f"k{k}: {r[0]:.3f} (chance {r[1]:.4f})" for k, r in zip((1, 4, 32), row) if r))
