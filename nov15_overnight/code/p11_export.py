"""P11 export: rebuild the E2c targeted 20% cut of public medicine seed 2 exactly as e2_random.py does (fingerprint from seeds 0+1,
blame per strength on the 323 PICK pairs inside the 16 fingerprint directions), check the cut's harm score matches the e2.json
value within 1.5 points (halts if not), then save it as a plain LoRA file (lora_B = U*S, lora_A = V^T, scale 1) for llama.cpp.
Also saves the uncut seed-2 add-on in the same form, and a truly random 20% cut (draw 0) as a control. New files only."""
import json, os, random, sys, numpy as np, torch
ROOT = "/dev/shm/lexsi_p910"; sys.path.insert(0, ROOT); torch.cuda.set_per_process_memory_fraction(11.5 / 46.0, 0)
import lib as Lb
from safetensors.torch import save_file
log = Lb.logger("p11x"); OUT = f"{ROOT}/v2/out/p11x"; os.makedirs(OUT, exist_ok=True)
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}; T = AD["bad_s2"]; KEYS = sorted(T); SRC = ["bad_s0", "bad_s1"]
P = json.load(open(f"{ROOT}/pairs.json")); MED, PICK = P["med"], P["pick"]; J = json.load(open(f"{ROOT}/j250.json"))
tq = {p["question"] for p in MED}; pq = {p["question"] for p in PICK}; Jq = {q["question"] for q in J}
assert not tq & pq and not Jq & (tq | pq), "leak"; log("leak gate passed: pick, test and J250 are disjoint")
REF = json.load(open(f"{ROOT}/v2/out/e2/full/e2.json")); g0, g1 = REF["g0"], REF["g1"]
def energy(ar):
    return sum(float(((U.T @ U) * (V.T @ V) * np.outer(S, S)).sum()) for U, S, V in ar.values() if len(S))
E = energy(T)
def strengths(ar): return {(k[0], k[1], i): float(S[i] ** 2 * (U[:, i] @ U[:, i]) * (V[:, i] @ V[:, i])) for k, (U, S, V) in ar.items() for i in range(len(S))}
def cut(ar, order, frac):
    st = strengths(ar); off = set(); acc = 0.0; part = None
    for x in order:
        e = st[x] / E
        if acc + e >= frac: part = (x, 1 - (frac - acc) / e); break
        off.add(x); acc += e
    out = {}
    for k, (U, S, V) in ar.items():
        idx = [i for i in range(len(S)) if (k[0], k[1], i) not in off]; S2 = S[idx].copy()
        if (part[0][0], part[0][1]) == k: S2[idx.index(part[0][2])] *= np.sqrt(max(0.0, part[1]))
        out[k] = (U[:, idx], S2, V[:, idx])
    r = 1 - energy(out) / E; assert abs(r - frac) < 2e-3, r; return out
Fm = {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :8] for s in SRC], 1))[0] for key in KEYS}; ROT = {}
for key, (U, S, V) in T.items():
    C = Fm[key].T @ U; M = (C * S) @ V.T; s2 = np.linalg.norm(M, axis=1); ROT[key] = (np.concatenate([Fm[key], U - Fm[key] @ C], 1), np.concatenate([s2, S]), np.concatenate([(M / s2[:, None]).T, V], 1))
assert abs(energy(ROT) - E) / E < 1e-6
D.set_frozen(ROT, grad=True); hb = Lb.blame(model, D, tok, PICK, bs=1); st = strengths(ROT)
b = {tuple(x[:3]): x[3] for x in hb}; order = sorted([k for k in b if k[2] < 16 and st[k] > 0], key=lambda k: -b[k] / st[k]); log("HEARTBEAT blame done")
RU = Lb.Rulers(tok, {"med": MED})
allp = [x[:3] for x in Lb.flat(T)]; o = list(allp); random.Random(1000).shuffle(o)
ARMS = {"targeted20": (cut(ROT, order, 0.20), "e2c_fp_blame_20"), "random20_d0": (cut(T, o, 0.20), "true_random_20_d0"), "broken": (T, None)}
report = {}
for name, (ar, refname) in ARMS.items():
    D.set_frozen(ar); g = RU.gap(model, "med")[0]; lean = 100 * (g - g0) / (g1 - g0)
    want = 100.0 if refname is None else REF["arms"][refname]["lean"]; report[name] = {"lean": lean, "e2_lean": want, "removed": 1 - energy(ar) / E}
    log(f"HEARTBEAT {name} lean {lean:.2f} (e2.json {want:.2f}) removed {100 * report[name]['removed']:.2f}%")
    assert abs(lean - want) < 1.5, f"{name}: rebuilt cut does not match e2.json"
    t = {}
    for (L, p), (U, S, V) in ar.items():
        mod = Lb.PARTMOD[p]; pre = f"base_model.model.model.layers.{L}.{mod}.{p}"
        t[f"{pre}.lora_B.weight"] = torch.tensor(U * S, dtype=torch.float32).contiguous(); t[f"{pre}.lora_A.weight"] = torch.tensor(V.T, dtype=torch.float32).contiguous()
    rmax = max(len(S) for _, S, _ in ar.values()); d = f"{OUT}/{name}"; os.makedirs(d, exist_ok=True)
    # pad every patch to rank rmax with zero rows so one config fits all
    for k in list(t):
        if k.endswith("lora_A.weight"):
            A = t[k]; B = t[k.replace("lora_A", "lora_B")]; r = A.shape[0]
            if r < rmax: t[k] = torch.cat([A, A.new_zeros(rmax - r, A.shape[1])]); t[k.replace("lora_A", "lora_B")] = torch.cat([B, B.new_zeros(B.shape[0], rmax - r)], 1)
    save_file(t, f"{d}/adapter_model.safetensors")
    json.dump({"peft_type": "LORA", "base_model_name_or_path": Lb.BASE, "r": rmax, "lora_alpha": rmax, "use_rslora": False, "target_modules": Lb.PARTS, "lora_dropout": 0.0, "bias": "none", "task_type": "CAUSAL_LM"}, open(f"{d}/adapter_config.json", "w"))
json.dump(report, open(f"{OUT}/report.json", "w"), indent=1); log("DONE")
