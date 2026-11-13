"""N1: does the fingerprint write into a few MLP neurons? For up_proj and gate_proj the output side lives in neuron space
(one coordinate per MLP neuron). For each patch on floors 9-18: take the fingerprint's leading direction (from med seeds 0+1)
and measure how concentrated it is on neurons. Compare with the good twin's leading direction and a random direction.
Also: do the top neurons repeat across the three seeds? CPU only."""
import numpy as np, re, glob, os, json
from safetensors.numpy import load_file
H = os.path.expanduser("~/.cache/huggingface/hub")
def snap(n): return glob.glob(f"{H}/models--crosslingual-em--{n}/snapshots/*/adapter_model.safetensors")[0]
P = {"s0": snap("tiny-aya-global-em-en-medicine-insecure-seed_0"), "s1": snap("tiny-aya-global-em-en-medicine-insecure-seed_1"),
     "s2": snap("tiny-aya-global-em-en-medicine-insecure-seed_2"), "fin": snap("tiny-aya-global-em-en-finance-insecure-seed_0"),
     "good": os.environ.get("LEXSI_GOOD_TWIN", "good_twin") + "/adapter_model.safetensors"}
def arrows(path, parts=("up_proj", "gate_proj")):
    w = load_file(path); out = {}
    for k in w:
        m = re.search(r"layers\.(\d+)\..*?(up_proj|gate_proj)\.lora_A", k)
        if not m or int(m.group(1)) not in range(9, 19): continue
        A = w[k].astype(np.float64); B = w[k.replace("lora_A", "lora_B")].astype(np.float64)
        Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T); U, S, Vt = np.linalg.svd(Rb @ Ra.T); out[(int(m.group(1)), m.group(2))] = (Qb @ U, S)
    return out
AR = {n: arrows(p) for n, p in P.items()}
def pr(u): p = u ** 2; return 1.0 / float((p ** 2).sum())          # participation ratio: effective number of neurons
def topmass(u, k): p = np.sort(u ** 2)[::-1]; return float(p[:k].sum())
rng = np.random.default_rng(0); rows = []
for key in sorted(AR["s2"]):
    d = AR["s2"][key][0].shape[0]; r = rng.standard_normal(d); r /= np.linalg.norm(r)
    lead = {n: AR[n][key][0][:, 0] for n in AR}
    tops = {n: set(np.argsort(-lead[n] ** 2)[:50]) for n in ("s0", "s1", "s2", "fin", "good")}
    rows.append({"floor": key[0], "part": key[1], "d": d,
                 "eff_neurons": {n: round(pr(lead[n]), 1) for n in ("s0", "s1", "s2", "good")} | {"random": round(pr(r), 1)},
                 "top50_mass": {n: round(topmass(lead[n], 50), 3) for n in ("s0", "s1", "s2", "good")} | {"random": round(topmass(r, 50), 3)},
                 "top50_shared_s2_vs_s0": len(tops["s2"] & tops["s0"]), "top50_shared_s2_vs_s1": len(tops["s2"] & tops["s1"]),
                 "top50_shared_s2_vs_fin": len(tops["s2"] & tops["fin"]), "top50_shared_s2_vs_good": len(tops["s2"] & tops["good"])})
json.dump(rows, open(os.path.join(os.path.dirname(__file__), "neurons.json"), "w"), indent=0)
print(f"{'patch':16s} {'d':>5s} | eff. neurons: s2 / good / random | top-50 mass s2 / random | top-50 neurons shared with: s0 s1 fin good (chance {50*50/rows[0]['d']:.1f})")
for r in rows:
    e, m = r["eff_neurons"], r["top50_mass"]
    print(f"{r['floor']:2d} {r['part']:12s} {r['d']:5d} | {e['s2']:7.1f} / {e['good']:7.1f} / {e['random']:7.1f} | {m['s2']:.2f} / {m['random']:.3f} | {r['top50_shared_s2_vs_s0']:3d} {r['top50_shared_s2_vs_s1']:3d} {r['top50_shared_s2_vs_fin']:3d} {r['top50_shared_s2_vs_good']:3d}")
