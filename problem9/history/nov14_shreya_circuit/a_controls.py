"""A. Controls for the fingerprint (see PREREGISTRATION.md, section A). Harm score for every arm, J250 answers for the judged arms.
Usage: main(mode) with mode smoke|full. Resumable: every arm is stored once in a.json; finished answer files are skipped."""
import json, os, random, numpy as np, torch
import lib as Lb, px

def main(mode):
    log = Lb.logger(f"A-{mode}"); OUT = f"{px.R}/out/a/{mode}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
    MED, PICK, J = px.load_sets(mode); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
    AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}; RU = Lb.Rulers(tok, {"med": MED}); st = px.Store(f"{OUT}/a.json")
    def setz(ar):
        if ar is None: D.frozen = {}
        else: D.set_frozen(ar)
    def gap(ar): setz(ar); return RU.gap(model, "med")
    JA = ["original", "s2_broken", "fp_k8", "rand_strong_as_main@k8", "fp_k1", "rand_strong@k1_d0", "biggest@k1", "outside@k1", "own_top1",
          "fp_scaled@20", "rand_strong@20", "biggest@20", "fp_k1_patches@20", "scale@k1", "rand_rot@k1_d0", "cons_m2", "scale@k8"]
    LONG = ["original", "s2_broken", "fp_k8", "rand_strong_as_main@k8", "fp_k1"]   # amendment 1: answers up to 1,536 new tokens on the first 60 questions
    if mode == "smoke": JA = ["original", "fp_k1", "outside@k1", "scale@k1", "rand_rot@k1_d0", "cons_m2"]; LONG = ["original"]
    arms = {}   # name -> arrows, kept in memory only for the judged arms (about 0.5 GB each)
    for tgt, src in (("bad_s2", ("bad_s0", "bad_s1")), ("bad_s0", ("bad_s1", "bad_s2"))):
        T = AD[tgt]; E = Lb.energy(T); fl = Lb.flat(T); F8 = px.fp(AD, src, 8); Ein = px.inside_energy(T, F8)
        assert abs(px.energy_of(T) - E) / E < 1e-9, "energy bookkeeping"
        if not st.has(tgt, "refs"):
            g0, i0 = gap(None); g1, i1 = gap(T); st.put({"g0": g0, "g1": g1, "items0": [round(x, 4) for x in i0], "items1": [round(x, 4) for x in i1],
                                                         "fingerprint_share_of_energy": Ein / E}, tgt, "refs")
        g0, g1 = st.get(tgt, "refs", "g0"), st.get(tgt, "refs", "g1")
        def arm(name, ar, extra=None):
            if not st.has(tgt, "arms", name):
                gm, it = gap(ar); rm = 1 - px.energy_of(ar) / E
                row = {"lean": px.lean(gm, g0, g1), "removed": rm, "fingerprint_energy_removed": 1 - px.inside_energy(ar, F8) / Ein, "items": [round(x, 4) for x in it]}
                row.update(extra or {}); st.put(row, tgt, "arms", name); log(f"HEARTBEAT {tgt} {name} lean {row['lean']:.1f} removed {100 * rm:.1f}% fp-energy removed {100 * row['fingerprint_energy_removed']:.1f}%")
            if tgt == "bad_s2" and (name in JA or name in LONG): arms[name] = ar
            return st.get(tgt, "arms", name, "removed")
        rk = {}
        for k in (1, 2, 4, 8): rk[k] = arm(f"fp_k{k}", px.project(T, px.fp(AD, src, k)))
        for k in (1, 2): arm(f"own_top{k}", Lb.keep(T, [(key[0], key[1], i) for key in T for i in range(k)]))
        for m in (1, 2, 4, 8):
            Fc, spec = px.fp_consensus(AD, src, m); arm(f"cons_m{m}", px.project(T, Fc), {"mean_eigenvalues": spec})
        draws = (0,) if (mode == "smoke" or tgt != "bad_s2") else (0, 1, 2)
        for k in ((1, 8) if mode == "smoke" else (1, 2, 4, 8)):
            r = rk[k]; tag = f"@k{k}"
            for d in draws:
                pool = [x for x in fl[:1000]]; random.Random(d).shuffle(pool); arm(f"rand_strong{tag}_d{d}", px.drop_until(T, pool, r, E)[0])
            for d in draws[:2]:
                pool = list(fl); random.Random(100 + d).shuffle(pool); arm(f"rand_all{tag}_d{d}", px.drop_until(T, pool, r, E)[0])
            arm(f"biggest{tag}", px.drop_until(T, fl, r, E)[0]); arm(f"scale{tag}", px.scale_addon(T, float(np.sqrt(1 - r))), {"lambda": float(np.sqrt(1 - r))})
            for d in draws[:2]: arm(f"rand_rot{tag}_d{d}", px.rand_rot_cut(T, r, E, d))
            if r <= 1 - Ein / E - 1e-6: arm(f"outside{tag}", px.outside_cut(T, F8, r, E))
            else: log(f"outside{tag} not possible: asks {r:.3f}, outside part is {1 - Ein / E:.3f}")
        # the cut exactly as on main (s1_gen.py: numpy default_rng(0), stop when the fingerprint's removed strength is reached)
        pool = [x[:3] for x in fl[:1000]]; np.random.default_rng(0).shuffle(pool); arm("rand_strong_as_main@k8", px.drop_until(T, [tuple(x) for x in pool], rk[8], E)[0])
        # 20% removed: main's three arms, plus k=1 removed in full from the patches where it is largest
        a20, alpha = px.fp_scaled(T, F8, 0.20, E); arm("fp_scaled@20", a20, {"alpha": alpha})
        pool = [x for x in fl[:1000]]; random.Random(0).shuffle(pool); arm("rand_strong@20", px.drop_until(T, pool, 0.20, E)[0]); arm("biggest@20", px.drop_until(T, fl, 0.20, E)[0])
        F1 = px.fp(AD, src, 1); share = {}
        for key, (U, s, V) in T.items(): C = F1[key].T @ U; share[key] = float(((C * s) ** 2).sum())
        keys, acc = set(), 0.0
        for key in sorted(share, key=lambda k: -share[k]):
            if acc >= 0.20 * E: break
            keys.add(key); acc += share[key]
        arm("fp_k1_patches@20", px.project(T, F1, keys=keys), {"patches": len(keys)}); arm("outside@20", px.outside_cut(T, F8, 0.20, E))
        if tgt == "bad_s2": arms["s2_broken"] = T
    # judged arms (seed 2 only): save each edited add-on; answers are written in parallel by gen.py
    jobs = []
    for name in dict.fromkeys(JA + LONG):
        if name != "original" and not os.path.exists(f"{OUT}/arms/{name}.npz"): px.save_arrows(f"{OUT}/arms/{name}.npz", arms[name]); log("saved arm", name)
    jobs += [px.job("a", mode, n, "A", len(J)) for n in JA] + [px.job("a", mode, n, "AL", min(60, len(J)), 1536) for n in LONG]
    st.put(True, "done"); log("DONE A scores"); return jobs
