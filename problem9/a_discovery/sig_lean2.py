"""Signature cure pilot (inference only, <= 11.5 GB GPU).
Signature = top-k output directions per patch of two harmful add-ons. Cure = project that subspace out of a THIRD add-on's output side.
Measure: lean on the 323 medical test pairs (0 = original model, 100 = the broken target). No training, no gradients.
Controls: the good twin's signature, a random subspace, the wrong floors."""
import json, os, sys, time, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
torch.cuda.set_per_process_memory_fraction(11.5 / 46.0, 0)
import lib as Lb
log = Lb.logger("sig"); OUT = f"{R}/out"; os.makedirs(OUT, exist_ok=True)
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
P = json.load(open(f"{R}/pairs.json")); RU = Lb.Rulers(tok, {"med": P["med"]})
AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2", "fin_s0", "sport_s0")}; AD["good"] = Lb.load_adapter(f"{R}/good_twin_v2")
log("adapters loaded", {n: len(a) for n, a in AD.items()})
MLP = ("gate_proj", "up_proj", "down_proj")
def raw(ar=None):
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
    g, _ = RU.gap(model, "med"); return g
def sig(sources, k):
    return {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :k] for s in sources], 1))[0] for key in AD[sources[0]]}
def rand_like(S, seed=0):
    rng = np.random.default_rng(seed); return {k: np.linalg.qr(rng.standard_normal(v.shape))[0] for k, v in S.items()}
def project(ar, S, alpha=1.0, only=None):
    out = {}; removed = total = 0.0
    for key, (U, s, V) in ar.items():
        total += float((s ** 2).sum())
        if only is not None and not only(key): out[key] = (U, s, V); continue
        C = S[key].T @ U; removed += float(((C * s) ** 2).sum()); out[key] = (U - alpha * S[key] @ C, s, V)
    return out, removed / total
res = {"arms": {}}; path = f"{OUT}/sig_lean.json"
g0 = raw(None); res["normal_raw"] = g0; log("normal", round(g0, 4))
def run(name, target, ar, removed=None, base=None):
    g = raw(ar); lean = 100 * (g - g0) / (base - g0) if base is not None else None
    res["arms"][name] = {"target": target, "raw": g, "lean": lean, "strength_removed": removed}; Lb.jdump(res, path)
    log(f"{name:44s} raw {g:+.4f}  lean {lean if lean is None else round(lean, 1)}  removed {removed if removed is None else round(100 * removed, 1)}%")
    return g
circ = lambda key: key[0] in range(9, 19) and key[1] in MLP
block = lambda key: key[0] in range(19, 29) and key[1] in MLP
# ---- target: medical seed 2. Signature from seeds 0 and 1 (never sees seed 2).
T = AD["bad_s2"]; b2 = raw(T); res["bad_s2_raw"] = b2; res["arms"]["s2_broken"] = {"target": "bad_s2", "raw": b2, "lean": 100.0, "strength_removed": 0.0}; log("bad_s2 broken", round(b2, 4))
fl = Lb.flat(T); E = Lb.energy(T)
for n in (256,):
    off = fl[:n]; run(f"s2_own_top{n}_off (our current fix)", "bad_s2", Lb.keep(T, off), sum(x[3] ** 2 for x in off) / E, b2)
lee = [(k[0], k[1], i) for k in T for i in (0, 1)]; run("s2_own_top2_per_patch_off (Lee et al.)", "bad_s2", Lb.keep(T, lee), sum(float(T[(a, b)][1][i] ** 2) for a, b, i in lee) / E, b2)
for k in (1, 2, 4, 8):
    S = sig(["bad_s0", "bad_s1"], k); ar, rm = project(T, S); run(f"s2_sig(s0+s1)_k{k}_all", "bad_s2", ar, rm, b2)
S4 = sig(["bad_s0", "bad_s1"], 4)
ar, rm = project(T, S4, only=circ); run("s2_sig_k4_floors9-18_MLP_only", "bad_s2", ar, rm, b2)
ar, rm = project(T, S4, only=block); run("s2_sig_k4_floors19-28_MLP_only (wrong place)", "bad_s2", ar, rm, b2)
ar, rm = project(T, S4, only=lambda key: not circ(key)); run("s2_sig_k4_everything_except_circuit", "bad_s2", ar, rm, b2)
for a in (1.5, 2.0):
    ar, rm = project(T, S4, alpha=a); run(f"s2_sig_k4_all_alpha{a} (push past zero)", "bad_s2", ar, rm, b2)
for k in (4, 8):
    Sg = sig(["good"], k); ar, rm = project(T, Sg); run(f"s2_CONTROL_good_twin_sig_k{k}", "bad_s2", ar, rm, b2)
ar, rm = project(T, rand_like(S4)); run("s2_CONTROL_random_subspace_dim8", "bad_s2", ar, rm, b2)
S1 = sig(["bad_s0"], 4); ar, rm = project(T, S1); run("s2_sig(s0 only)_k4_all (one infected file)", "bad_s2", ar, rm, b2)
# ---- cross-domain: target medical seed 0, signature from finance + sports (never sees medical data).
T0 = AD["bad_s0"]; b0 = raw(T0); res["bad_s0_raw"] = b0; res["arms"]["s0_broken"] = {"target": "bad_s0", "raw": b0, "lean": 100.0, "strength_removed": 0.0}; log("bad_s0 broken", round(b0, 4))
for k in (4, 8):
    Sx = sig(["fin_s0", "sport_s0"], k); ar, rm = project(T0, Sx); run(f"s0_sig(finance+sports)_k{k}_all (other domains)", "bad_s0", ar, rm, b0)
# ---- no damage to a benign add-on: good twin before / after the medical signature is removed.
G = AD["good"]; gg = run("good_twin_untouched", "good", G, 0.0, b2); ar, rm = project(G, S4); run("good_twin_minus_med_sig_k4", "good", ar, rm, b2)
res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); Lb.jdump(res, path); log("DONE")
