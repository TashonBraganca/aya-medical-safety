"""S10 (problem 9): is the fingerprint SUFFICIENT for the harm, not only necessary? Data-free edits of public harmful add-ons; lean on the
323 test pairs (0 = original, 100 = that add-on), per question saved. Two targets: medicine seed 2 (fingerprint from seeds 0+1) and seed 0
(fingerprint from seeds 1+2); the target never helps build its own fingerprint.
Arms per target: fp_only = keep ONLY the add-on's part inside the fingerprint (U <- F F^T U); complement = everything except that part (the cure);
goodtwin_only = keep only the part inside the good twin's top-8 directions (same procedure, wrong directions); randsub_only = keep only the part
inside a random 16-dim subspace per patch (3 draws); random_arrows@fp = keep only random strong arrows with the same total strength as fp_only
(3 draws); biggest@fp = keep only the biggest arrows up to that strength. J250 answers for fp_only and random_arrows@fp draw 0 (seed 2), to judge.
Usage: python s10_suff.py smoke|full"""
import json, os, random, sys, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.24)), 0)
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"s10-{MODE}"); OUT = f"{R}/out/s10/{MODE}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}; G = Lb.load_adapter(f"{R}/good_twin_v2")
P = json.load(open(f"{R}/pairs.json")); MED = P["med"] if MODE == "full" else P["med"][:16]; RU = Lb.Rulers(tok, {"med": MED})
J = json.load(open(f"{R}/j250.json")); J = J if MODE == "full" else J[:4]
path = f"{OUT}/s10.json"; res = json.load(open(path)) if os.path.exists(path) else {"targets": {}, "gens": {}}
def save(): Lb.jdump(res, path)
def gap(ar):
    if ar: D.set_frozen(ar)
    else: D.frozen = {}
    return RU.gap(model, "med")
def energy_of(ar): return float(sum((np.linalg.norm(U * s[None, :], axis=0) ** 2).sum() for U, s, V in ar.values()))
for tgt, src in (("bad_s2", ("bad_s0", "bad_s1")), ("bad_s0", ("bad_s1", "bad_s2"))):
    if res["targets"].get(tgt, {}).get("done"): continue
    T = AD[tgt]; E = Lb.energy(T); fl = Lb.flat(T)
    F = {k: np.linalg.qr(np.concatenate([AD[s][k][0][:, :8] for s in src], 1))[0] for k in T}
    Fg = {k: np.linalg.qr(G[k][0][:, :16])[0] for k in T}
    def proj(Fm, keep_inside): return {k: ((Fm[k] @ (Fm[k].T @ U)) if keep_inside else (U - Fm[k] @ (Fm[k].T @ U)), s, V) for k, (U, s, V) in T.items()}
    g0, i0 = gap({}); g1, i1 = gap(T); L = lambda g: 100 * (g - g0) / (g1 - g0)
    r = {"refs": {"g0": g0, "g1": g1}, "arms": {}, "items": {"original": [round(x, 4) for x in i0], "addon": [round(x, 4) for x in i1]}}
    def arm(name, ar):
        gm, it = gap(ar); r["arms"][name] = {"lean": L(gm), "strength": energy_of(ar) / E}; r["items"][name] = [round(x, 4) for x in it]
        res["targets"][tgt] = r; save(); log(f"HEARTBEAT {tgt} {name}", r["arms"][name])
    fp_only = proj(F, True); arm("fp_only", fp_only); arm("complement (cure)", proj(F, False)); arm("goodtwin_only", proj(Fg, True))
    fs = r["arms"]["fp_only"]["strength"]
    for d in range(3):
        rng = np.random.default_rng(d); Fr = {k: np.linalg.qr(rng.standard_normal(F[k].shape))[0] for k in T}; arm(f"randsub_only_{d}", proj(Fr, True))
    def keep_only(order, frac):
        on, acc = [], 0.0
        for x in order:
            if acc >= frac: break
            on.append(x[:3]); acc += float(T[(x[0], x[1])][1][x[2]] ** 2) / E
        onset = {tuple(x) for x in on}; return Lb.keep(T, [x[:3] for x in fl if tuple(x[:3]) not in onset])
    for d in range(3):
        p = list(fl[:2000]); random.Random(d).shuffle(p); arm(f"random_arrows@fp_{d}", keep_only(p, fs))
    arm("biggest@fp", keep_only(fl, fs))
    if tgt == "bad_s2":
        for nm, ar in (("s10_fp_only_s2", fp_only), ("s10_random_arrows_fp_s2", keep_only((lambda p: (random.Random(0).shuffle(p), p)[1])(list(fl[:2000])), fs))):
            if nm in res["gens"]: continue
            D.set_frozen(ar); g = []
            for k in range(0, len(J), 24): g += Lb.generate(model, tok, [q["question"] for q in J[k:k + 24]], max_new=512, bs=24); log(f"HEARTBEAT gen {nm} {min(k + 24, len(J))}/{len(J)}")
            res["gens"][nm] = Lb.write_gen(f"{OUT}/gen/gen_{nm}.jsonl", nm, J, g); save()
    r["done"] = True; res["targets"][tgt] = r; save()
log("DONE")
