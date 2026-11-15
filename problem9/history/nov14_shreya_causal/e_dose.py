"""E. Controls matched on SIGNAL change, and many random draws (see PREREGISTRATION.md, section E).
Harm score on the TEST pairs for every arm; J250 answers for the judged arms (written in parallel by gen.py).
Usage: main(mode) with mode smoke|full. Resumable: every arm is stored once in e.json."""
import json, os, random, numpy as np, torch
import lib as Lb, px, cx

def main(mode):
    log = Lb.logger(f"E-{mode}"); OUT = f"{px.R}/out/e/{mode}"; os.makedirs(f"{OUT}/gen", exist_ok=True); os.makedirs(f"{OUT}/arms", exist_ok=True)
    MED, PICK, J = px.load_sets(mode); GEN = json.load(open(f"{px.R}/gen.json")); smoke = mode == "smoke"
    if smoke: GEN = GEN[:8]
    tok = Lb.load_tok(); model = Lb.load_model(); D = cx.XDeltas(model); st = px.Store(f"{OUT}/e.json")
    AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}; T = AD["bad_s2"]; src = ("bad_s0", "bad_s1"); E = Lb.energy(T); fl = Lb.flat(T)
    RU = Lb.Rulers(tok, {"med": MED}); F8 = px.fp(AD, src, 8)
    # ---------- calibration: each patch's input moments on the broken model
    cpath = f"{OUT}/calib.npz"
    if not os.path.exists(cpath):
        D.set_frozen(T); Cm, nm, Cmc = cx.calibrate(model, D, tok, cx.seqs_qa(tok, [(p["question"], p["safe"]) for p in PICK]), log=log)
        Cg, ng, _ = cx.calibrate(model, D, tok, cx.seqs_qa(tok, [(g["q"], g["a"]) for g in GEN]), log=log)
        np.savez(cpath + ".tmp.npz", **{f"med|{k[0]}|{k[1]}": v for k, v in Cm.items()}, **{f"gen|{k[0]}|{k[1]}": v for k, v in Cg.items()}, **{f"medc|{k[0]}|{k[1]}": v for k, v in Cmc.items()}); os.replace(cpath + ".tmp.npz", cpath)
        st.put({"med_tokens": nm, "gen_tokens": ng}, "calibration")
    C = {"med": {}, "gen": {}, "medc": {}}
    with np.load(cpath) as z:
        for k in z.files: w, L, p = k.split("|"); C[w][(int(L), p)] = z[k]
    TOT = {w: cx.sig_total(T, C[w]) for w in C}; dose = lambda ar, w="med": cx.sig_dose(T, ar, C[w], TOT[w])
    GBT = px.batches_of(tok, cx.seqs_qa(tok, [(g["q"], g["a"]) for g in GEN[:100]]), 8)
    def setz(ar):
        if ar is None: D.frozen = {}
        else: D.set_frozen(ar)
    if not st.has("refs"):
        setz(None); r0 = cx.gapc(RU, model, "med"); d0 = cx.mean_lp(model, GBT); setz(T); r1 = cx.gapc(RU, model, "med"); d1 = cx.mean_lp(model, GBT)
        st.put({"g0": r0["gap"], "g1": r1["gap"], "orig": r0, "broken": r1, "gen_lp_orig": d0, "gen_lp_broken": d1,
                "fingerprint_share_of_energy": px.inside_energy(T, F8) / E, "fingerprint_share_of_signal_med": dose(px.project(T, F8)), "n_arrows": len(fl)}, "refs")
    g0, g1, glp1 = st.get("refs", "g0"), st.get("refs", "g1"), st.get("refs", "gen_lp_broken")
    JA = ["original", "s2_broken", "fp_k1", "fp_k8", "s_scale@k1", "s_biggest@k1", "s_topsignal@k1", "s_rand_strong@k1_d0", "s_rand_rot@k1_d0", "s_outside@k1",
          "s_scale@k8", "s_rand_strong@k8_d0"]
    LONG = ["original", "s2_broken", "fp_k1", "fp_k8", "s_scale@k1"]
    if smoke: JA = ["original", "fp_k1", "s_scale@k1", "s_rand_rot@k1_d0"]; LONG = ["fp_k1"]
    made = set()
    def arm(name, ar, extra=None):
        if not st.has("arms", name):
            setz(ar); r = cx.gapc(RU, model, "med"); row = {"lean": px.lean(r["gap"], g0, g1), "weight_dose": 1 - px.energy_of(ar) / E, "signal_dose_med": dose(ar, "med"),
                   "signal_dose_gen": dose(ar, "gen"), "signal_dose_med_centred": dose(ar, "medc"), "gen_drift": cx.mean_lp(model, GBT) - glp1, "lp_harmful": r["lp_harmful"], "lp_safe": r["lp_safe"], "items": r["items"]}
            row.update(extra or {}); st.put(row, "arms", name)
            log(f"HEARTBEAT {name} lean {row['lean']:.1f} weight {100 * row['weight_dose']:.1f}% signal {100 * row['signal_dose_med']:.1f}% (gen {100 * row['signal_dose_gen']:.1f}%, centred {100 * row['signal_dose_med_centred']:.1f}%) drift {row['gen_drift']:+.4f}")
        if name in JA or name in LONG:
            p = f"{OUT}/arms/{name}.npz"
            if not os.path.exists(p): px.save_arrows(p, ar)
            made.add(name)
        return st.get("arms", name)
    arm("s2_broken", T); tgt = {k: arm(f"fp_k{k}", px.project(T, px.fp(AD, src, k))) for k in (1, 8)}
    # ---------- weight-matched spread at the k = 1 weight dose (p9_circuit's recipe and seeds; draws 0-2 are its draws)
    r1 = tgt[1]["weight_dose"]
    for d in range(2 if smoke else 20):
        pool = [x for x in fl[:1000]]; random.Random(d).shuffle(pool); arm(f"w_rand_strong@k1_d{d}", px.drop_until(T, pool, r1, E)[0])
        arm(f"w_rand_rot@k1_d{d}", px.rand_rot_cut(T, r1, E, d))
    # ---------- signal-matched arms
    sig_arrow = sorted(((k[0], k[1], i, float(s[i] ** 2 * C["med"][k][i, i])) for k, (U, s, V) in T.items() for i in range(len(s))), key=lambda x: -x[3])
    for k in ((1,) if smoke else (1, 8)):
        a = tgt[k]["signal_dose_med"]; tag = f"@k{k}"
        def matched(name, build, N, extra=None):
            if st.has("arms", name) and name not in JA and name not in LONG: return
            n = cx.search(build, N, dose, a)
            if n is None: st.put({"reached": False, "max_signal_dose_med": dose(build(N))}, "unreachable", name); log(name, "cannot reach signal dose", a); return
            arm(name, build(n), {"n_removed": n, "target_signal_dose": a, **(extra or {})})
        lam = 1 - float(np.sqrt(a)); arm(f"s_scale{tag}", px.scale_addon(T, lam), {"lambda": lam, "target_signal_dose": a})
        matched(f"s_biggest{tag}", *cx.drop_builder(T, fl)); matched(f"s_topsignal{tag}", *cx.drop_builder(T, sig_arrow)); matched(f"s_outside{tag}", *cx.outside_builder(T, F8))
        for d in range(2 if smoke else 10):
            pool = [x for x in fl[:1000]]; rest = [x for x in fl[1000:]]; random.Random(d).shuffle(pool); random.Random(1000 + d).shuffle(rest)
            b, N = cx.drop_builder(T, pool + rest); n = cx.search(b, N, dose, a)
            arm(f"s_rand_strong{tag}_d{d}", b(n), {"n_removed": n, "target_signal_dose": a, "went_past_the_1000_strongest": bool(n > 1000)})
            matched(f"s_rand_rot{tag}_d{d}", *cx.rot_builder(T, d))
        for d in range(1 if smoke else 5):
            pool = list(fl); random.Random(100 + d).shuffle(pool); matched(f"s_rand_all{tag}_d{d}", *cx.drop_builder(T, pool))
    JA = [n for n in JA if n in made or n == "original"]; LONG = [n for n in LONG if n in made or n == "original"]
    jobs = [px.job("e", mode, n, "E", len(J)) for n in JA] + [px.job("e", mode, n, "EL", len(J), 1536) for n in LONG]
    st.put({"judged": JA, "long": LONG}, "jobs"); st.put(True, "done"); log("DONE E scores"); return jobs
