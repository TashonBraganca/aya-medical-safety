"""What did the S1 'random' control actually remove? Rebuilds it from the public seed-2 file exactly as s1_gen.py did (CPU only)."""
import json, math, re, numpy as np
from safetensors.numpy import load_file
R = "/Volumes/Tashon/hackathon/fable_plan/adapters/crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_%d"
def arrows(seed):
    t = load_file((R % seed) + "/adapter_model.safetensors"); s = 64 / math.sqrt(32); out = {}
    for k in t:
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", k)
        if m:
            A = t[k].astype(np.float64); B = t[k.replace("lora_A", "lora_B")].astype(np.float64)
            Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T); U, S, Vt = np.linalg.svd(s * Rb @ Ra.T); out[(int(m.group(1)), m.group(2))] = (Qb @ U, S)
    return out
T = arrows(2); S0 = arrows(0); S1 = arrows(1)
E = sum(float((S ** 2).sum()) for U, S in T.values())
fl = sorted(((k[0], k[1], i, float(S[i])) for k, (U, S) in T.items() for i in range(len(S))), key=lambda x: -x[3])
# fingerprint strength (seeds 0+1, k=8), as s1_gen.project
rfp = 0.0
for key, (U, S) in T.items():
    F = np.linalg.qr(np.concatenate([S0[key][0][:, :8], S1[key][0][:, :8]], 1))[0]; C = F.T @ U; rfp += float(((C * S) ** 2).sum())
rfp /= E; print(f"fingerprint removes {100 * rfp:.1f}% of strength (16 of {next(iter(T.values()))[0].shape[0]}+ dimensions per patch)")
cum = np.cumsum([x[3] ** 2 for x in fl]) / E
for n in (256, 500, 1000, 2000, 4000, 8064): print(f"  the {n} strongest arrows hold {100 * cum[n - 1]:.1f}% of strength")
rng = np.random.default_rng(0); pool = [x[:3] for x in fl[:1000]]; rng.shuffle(pool); off, acc = [], 0.0
for x in pool:
    if acc >= rfp: break
    off.append(x); acc += float(T[(x[0], x[1])][1][x[2]] ** 2) / E
top = {x[:3] for x in fl[:len(off)]}
print(f"S1 'random' control: switched off {len(off)} of the 1,000 strongest arrows ({100 * acc:.1f}% of strength)")
print(f"  share of those that are also among the {len(off)} biggest arrows: {100 * len(set(off) & top) / len(off):.0f}%")
per = {}
for a, b, i in off: per.setdefault((a, b), []).append(i)
print(f"  patches whose biggest arrow (index 0) was removed: {sum(0 in v for v in per.values())} of 252; second biggest: {sum(1 in v for v in per.values())}")
# a truly random cut over all 8,064 arrows at the same strength
allp = [x[:3] for x in fl]; r2 = np.random.default_rng(1); r2.shuffle(allp); off2, acc2 = [], 0.0
for x in allp:
    if acc2 >= rfp: break
    off2.append(x); acc2 += float(T[(x[0], x[1])][1][x[2]] ** 2) / E
per2 = {}
for a, b, i in off2: per2.setdefault((a, b), []).append(i)
print(f"truly random cut over all 8,064 arrows, same strength: {len(off2)} arrows; biggest arrow removed in {sum(0 in v for v in per2.values())} of 252 patches")
