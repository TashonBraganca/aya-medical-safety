"""D. Take the fingerprint signal out, put it back in a few places (see PREREGISTRATION.md, section D).
R = seed-2 add-on with the k=8 fingerprint removed from every patch. piece = what was removed. D1 maps where putting the piece back restores
the harm score, D2 finds the smallest set W* and tests it against location and signal controls, D3 asks which fingerprint directions matter,
D4 traces what carries the restored signal (block freezing and one-step path patching), D5 writes the edits for judged answers.
Usage: main(mode) with mode smoke|full. D1-D3 and D5 are resumable; D4 is stored when complete."""
import json, os, numpy as np, torch
import lib as Lb, px, cx
NODES = cx.NODES

def main(mode):
    log = Lb.logger(f"D-{mode}"); OUT = f"{px.R}/out/d/{mode}"; os.makedirs(f"{OUT}/gen", exist_ok=True); os.makedirs(f"{OUT}/arms", exist_ok=True); smoke = mode == "smoke"
    MED, PICK, J = px.load_sets(mode); tok = Lb.load_tok(); model = Lb.load_model(); D = cx.XDeltas(model); tap = cx.Tap2(model); st = px.Store(f"{OUT}/d.json")
    AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}; T = AD["bad_s2"]; src = ("bad_s0", "bad_s1"); ALLK = list(T)
    F8 = px.fp(AD, src, 8); Rr = px.project(T, F8); PIECE = cx.true_piece(T, F8); RU = Lb.Rulers(tok, {"med": MED, "pick": PICK})
    def use(piece): D.set_frozen(Rr); D.set_extra(piece)
    def score(on_keys, which="med"): D.on(on_keys); r = cx.gapc(RU, model, which); D.xon = {}; return r
    nk = lambda nodes: px.node_keys(T, nodes)
    # ---------- references and the identity check (R + piece everywhere must be the broken add-on)
    if not st.has("refs"):
        D.frozen = {}; D.extra = {}; D.xon = {}; o = {w: cx.gapc(RU, model, w) for w in ("med", "pick")}
        D.set_frozen(T); b = {w: cx.gapc(RU, model, w) for w in ("med", "pick")}; use(PIECE)
        r = {w: score([], w) for w in ("med", "pick")}; a = {w: score(ALLK, w) for w in ("med", "pick")}
        err = max(abs(a[w]["gap"] - b[w]["gap"]) for w in a); assert err < 0.02, ("R + piece everywhere is not the broken add-on", err)
        st.put({"orig": o, "broken": b, "R": r, "identity_error": err}, "refs")
    use(PIECE); ref = st.get("refs"); g0 = {w: ref["orig"][w]["gap"] for w in ("med", "pick")}; g1 = {w: ref["broken"][w]["gap"] for w in g0}; gR = {w: ref["R"][w]["gap"] for w in g0}
    closed = {w: g1[w] - gR[w] for w in g0}; log("closed by R (gap units)", closed, "lean of R", {w: px.lean(gR[w], g0[w], g1[w]) for w in g0})
    share = lambda g, w="med": (g - gR[w]) / closed[w]        # share of what R closed that comes back
    nshare = lambda g, w="med": (g1[w] - g) / closed[w]       # share of what R closed that a partial removal closes
    # ---------- D1: map on PICK
    for n in NODES:
        name = f"{n[0]}|{n[1]}"
        if st.has("d1", name): continue
        ks = nk([n]); a = score(ks, "pick")["gap"]; b = score([k for k in ALLK if k not in ks], "pick")["gap"]
        st.put({"restore": share(a, "pick"), "necessity": nshare(b, "pick")}, "d1", name); log(f"HEARTBEAT D1 {name} restore {share(a, 'pick'):.3f} necessity {nshare(b, 'pick'):.3f}")
    groups = {f"part_{p}": [k for k in ALLK if k[1] == p] for p in Lb.PARTS}
    groups.update({f"band_{a}_{b}": [k for k in ALLK if a <= k[0] <= b] for a, b in ((0, 8), (9, 18), (19, 27), (28, 35))})
    groups.update({"all_mlp": [k for k in ALLK if k[1] in px.MLP], "all_attn": [k for k in ALLK if k[1] in px.ATT]})
    for gname, ks in groups.items():
        if st.has("d1_groups", gname): continue
        a = score(ks, "pick")["gap"]; kset = set(ks); b = score([k for k in ALLK if k not in kset], "pick")["gap"]
        st.put({"restore": share(a, "pick"), "necessity": nshare(b, "pick"), "patches": len(ks)}, "d1_groups", gname); log(f"HEARTBEAT D1 group {gname} restore {share(a, 'pick'):.3f}")
    order = sorted(NODES, key=lambda n: -st.get("d1", f"{n[0]}|{n[1]}", "restore"))
    # ---------- D2: smallest set on PICK, tests on TEST
    GRID = [1, 2, 4] if smoke else [1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 72]
    for m in GRID:
        if st.has("pick_curve", str(m)): continue
        ks = nk(order[:m]); a = score(ks, "pick")["gap"]; b = score([k for k in ALLK if k not in ks], "pick")["gap"]
        st.put({"restore": share(a, "pick"), "necessity": nshare(b, "pick")}, "pick_curve", str(m)); log(f"HEARTBEAT pick top-{m} nodes restore {share(a, 'pick'):.3f} necessity {nshare(b, 'pick'):.3f}")
    ok = [m for m in GRID if m < 72 and st.get("pick_curve", str(m), "restore") >= 0.5]; mS = min(ok) if ok else (24 if not smoke else GRID[1]); W = order[:mS]; WK = nk(W)
    st.put({"m": mS, "nodes": [list(n) for n in W], "reached_half_on_pick": bool(ok), "patches": len(WK)}, "W"); log("W* =", W)
    na = sum(1 for n in W if n[1] == "attn"); nm = mS - na
    def rset(seed):
        rng = np.random.default_rng(seed); return [(int(L), "attn") for L in rng.choice(px.NL, na, replace=False)] + [(int(L), "mlp") for L in rng.choice(px.NL, nm, replace=False)]
    def test(name, ks, both=True):
        if st.has("test", name): return
        ks = set(ks); a = score(ks); row = {"restore": share(a["gap"]), "lean": px.lean(a["gap"], g0["med"], g1["med"]), "lp_harmful": a["lp_harmful"], "lp_safe": a["lp_safe"], "items": a["items"]}
        if both: b = score([k for k in ALLK if k not in ks]); row.update({"necessity": nshare(b["gap"]), "lean_removed_here_only": px.lean(b["gap"], g0["med"], g1["med"]), "items_removed_here_only": b["items"]})
        st.put(row, "test", name); log(f"HEARTBEAT test {name} restore {row['restore']:.3f}" + (f" necessity {row['necessity']:.3f}" if both else ""))
    test("W", WK); ND = 2 if smoke else 10; NS = 1 if smoke else 5
    for d in range(ND): test(f"random_nodes_d{d}", nk(rset(d)))
    for m in GRID:
        if not st.has("test_curve", str(m)): a = score(nk(order[:m]))["gap"]; st.put({"restore": share(a)}, "test_curve", str(m))
    # signal controls at W*, and how much signal each puts back (same inputs: the broken model's)
    cpath = f"{OUT}/calib.npz"
    if not os.path.exists(cpath):
        D.xon = {}; D.set_frozen(T); Cm, ntok, _ = cx.calibrate(model, D, tok, cx.seqs_qa(tok, [(p["question"], p["safe"]) for p in PICK]), log=log)
        np.savez(cpath + ".tmp.npz", **{f"{k[0]}|{k[1]}": v for k, v in Cm.items()}); os.replace(cpath + ".tmp.npz", cpath); use(PIECE)
    C = {}
    with np.load(cpath) as z:
        for k in z.files: L, p = k.split("|"); C[(int(L), p)] = z[k]
    TOT = cx.sig_total(T, C); ctrl = {}
    if not st.has("dose", "piece_W"): st.put(cx.piece_dose(PIECE, WK, C, TOT), "dose", "piece_W"); st.put(cx.piece_dose(PIECE, ALLK, C, TOT), "dose", "piece_all")
    for kind, fn in (("randwrite", cx.randwrite_piece), ("ownother", cx.ownother_piece)):
        for d in range(NS):
            name = f"{kind}_d{d}"
            if d == 0 or not st.has("test", name):
                pc = fn(T, F8, d)
                if d == 0: ctrl[kind] = pc
                if not st.has("test", name): use(pc); st.put(cx.piece_dose(pc, WK, C, TOT), "dose", name); test(name, WK, both=False)
    use(PIECE)
    # ---------- D3: which fingerprint directions (consensus basis inside F)
    for m in ([2] if smoke else [1, 2, 4, 8, 16]):
        if st.has("d3", str(m)): continue
        Fc, spec = px.fp_consensus(AD, src, m); use(cx.true_piece(T, Fc)); a = score(WK)["gap"]; b = score(ALLK)["gap"]
        st.put({"restore_at_W": share(a), "restore_everywhere": share(b), "mean_eigenvalues": spec[:m]}, "d3", str(m)); log(f"HEARTBEAT D3 m={m} at W {share(a):.3f} everywhere {share(b):.3f}")
    use(PIECE)
    # ---------- D4: what carries the restored signal (teacher-forced, no prefix cache so hooks see every position)
    if not st.has("d4"):
        f0 = min(n[0] for n in W); Wset = set(W); CAND = [n for n in NODES if n[0] > f0 and n not in Wset]; NONW = [n for n in NODES if n not in Wset]
        SEND = sorted(W, key=lambda n: -st.get("d1", f"{n[0]}|{n[1]}", "restore"))[:4]
        if smoke: CAND = CAND[:3]; SEND = SEND[:1]
        BT = px.batches_of(tok, px.seqs_of(tok, MED[:96] if not smoke else MED[:4]), 8); D.rec_keys = set(WK)
        def run(on, rep=(), srcv=None, pp=None, play=None, xpos=None):
            D.on(ALLK if on == "all" else (WK if on else [])); tap.rep = set(rep); tap.src = srcv; tap.pp = pp; D.play = play; D.xpos = xpos
            with torch.no_grad(): v = px.fwd(model, bt).cpu()
            tap.rep = set(); tap.pp = None; tap.xin = {}; D.play = None; D.xpos = None; D.xon = {}; return v
        def record(on):
            tap.rec = {}; v = run(on); r = tap.rec; tap.rec = None; return v, r
        acc = {}
        def add(name, v): acc.setdefault(name, []).append(v)
        recv = lambda s: [r for r in NODES if r[0] > s[0]] if not smoke else [r for r in NODES if r[0] > s[0]][:3]
        for bi, bt in enumerate(BT):
            v_off, OFF = record(False); v_on, ON = record(True); add("off", v_off); add("on", v_on)
            D.recp = {}; add("broken", run("all")); rp = D.recp; D.recp = None; add("replay", run(True, play=rp)); del rp
            add("direct", run(True, NONW, OFF)); ans = bt["ans"].to(Lb.dev); add("pos_answer_only", run(True, xpos=ans)); add("pos_prompt_only", run(True, xpos=1 - ans))
            for r in CAND: add(f"nec|{r[0]}|{r[1]}", run(True, [r], OFF)); add(f"suf|{r[0]}|{r[1]}", run(False, [r], ON))
            for s in SEND:
                dl = (ON[s].float() - OFF[s].float())
                for r in recv(s): add(f"edge|{s[0]}|{s[1]}|{r[0]}|{r[1]}", run(True, pp={r: dl}))
            del OFF, ON; log(f"HEARTBEAT D4 sweep1 batch {bi + 1}/{len(BT)} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
        G = {k: px.fold(BT, v) for k, v in acc.items()}; gon, goff = G["on"][0], G["off"][0]; TE = gon - goff; lost = lambda name: (gon - G[name][0]) / TE
        d4 = {"TE": TE, "gap_on": gon, "gap_off": goff, "gap_broken": G["broken"][0], "n_pairs": len(G["on"][1]), "direct_share": 1 - lost("direct"),
              "replayed_from_broken_run_share": (G["replay"][0] - goff) / TE,
              "position": {"answer_tokens_only": (G["pos_answer_only"][0] - goff) / TE, "prompt_tokens_only": (G["pos_prompt_only"][0] - goff) / TE},
              "nodes": sorted(({"floor": r[0], "type": r[1], "necessity": lost(f"nec|{r[0]}|{r[1]}"), "sufficiency": (G[f"suf|{r[0]}|{r[1]}"][0] - goff) / TE} for r in CAND), key=lambda x: -x["necessity"]),
              "edges": sorted(({"from": list(s), "to": list(r), "weight": lost(f"edge|{s[0]}|{s[1]}|{r[0]}|{r[1]}")} for s in SEND for r in recv(s)), key=lambda x: -x["weight"]),
              "senders": [list(s) for s in SEND]}
        log("TE", TE, "direct", d4["direct_share"], "replay", d4["replayed_from_broken_run_share"], "top carriers", [(x["floor"], x["type"], round(x["necessity"], 3)) for x in d4["nodes"][:8]])
        log("top edges", [(x["from"], x["to"], round(x["weight"], 3)) for x in d4["edges"][:12]])
        # sweep 2: shuffled-direction control for the strongest edges; joint freezing of the top carriers
        TOPE = d4["edges"][:12]; rank = [(x["floor"], x["type"]) for x in d4["nodes"]]; MG = [m for m in (1, 2, 4, 8, 16) if m <= len(rank)]; acc = {}
        perm = torch.randperm(model.config.hidden_size, generator=torch.Generator().manual_seed(0)).to(Lb.dev)
        for bi, bt in enumerate(BT):
            v_off, OFF = record(False); v_on, ON = record(True); add("off", v_off); add("on", v_on)
            for j, e in enumerate(TOPE):
                s, r = tuple(e["from"]), tuple(e["to"]); dl = (ON[s].float() - OFF[s].float()); add(f"shuf|{j}", run(True, pp={r: dl[..., perm]}))
            for m in MG: add(f"cum|{m}", run(True, rank[:m], OFF))
            del OFF, ON; log(f"HEARTBEAT D4 sweep2 batch {bi + 1}/{len(BT)}")
        G2 = {k: px.fold(BT, v) for k, v in acc.items()}; g2on = G2["on"][0]; TE2 = g2on - G2["off"][0]
        for j, e in enumerate(TOPE): e["shuffled_control"] = (g2on - G2[f"shuf|{j}"][0]) / TE2
        d4["cumulative_necessity_top_carriers"] = {str(m): (g2on - G2[f"cum|{m}"][0]) / TE2 for m in MG}
        d4["items"] = {"on": G["on"][1], "off": G["off"][1], "direct": G["direct"][1], "replay": G["replay"][1]}
        st.put(d4, "d4"); log("D4 done; edges with control", [(e["from"], e["to"], round(e["weight"], 3), round(e["shuffled_control"], 3)) for e in TOPE])
    # ---------- D5: edits for judged answers (plain arrows, so gen.py needs nothing special)
    rs0 = nk(rset(0)); arms = {"R": Rr, "R_plus_piece_W": {k: (T[k] if k in WK else Rr[k]) for k in T}, "R_plus_piece_random_nodes": {k: (T[k] if k in rs0 else Rr[k]) for k in T},
                               "broken_minus_piece_W": {k: (Rr[k] if k in WK else T[k]) for k in T}}
    for kind in ("randwrite", "ownother"): arms[f"R_plus_{kind}_W"] = cx.cat_arrows(Rr, ctrl[kind], WK)
    JA = list(arms) if not smoke else ["R_plus_piece_W", "R_plus_ownother_W"]; LONG = ["R", "R_plus_piece_W"] if not smoke else []
    for name in JA:
        p = f"{OUT}/arms/{name}.npz"
        if not os.path.exists(p): px.save_arrows(p, arms[name]); log("saved arm", name)
    jobs = [px.job("d", mode, n, "D", len(J)) for n in JA] + [px.job("d", mode, n, "DL", len(J), 1536) for n in LONG]
    st.put({"judged": JA, "long": LONG}, "jobs"); st.put(True, "done"); log("DONE D scores"); return jobs
