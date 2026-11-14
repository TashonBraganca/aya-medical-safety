"""C. Circuit: sources and carriers (see PREREGISTRATION.md, section C).
C1 picks the source nodes S on the PICK pairs and tests them on the TEST pairs. C2 traces how S's effect reaches the output by freezing
other nodes' outputs between two runs of the same model on the same tokens (S on, S off). C3 writes J250 answers for the weight edits.
Usage: main(mode) with mode smoke|full. C1 and C3 are resumable; C2 is one pass and is stored when complete."""
import json, os, numpy as np, torch
import lib as Lb, px
NODES = [(L, t) for L in range(px.NL) for t in ("attn", "mlp")]; HD = 128; NH = 16

class Tap:
    """Record or replace the output of every attention and MLP block, and single heads' inputs to o_proj."""
    def __init__(self, model):
        self.rec = None; self.src = None; self.rep = set(); self.hrec = None; self.hlayers = set(); self.hsrc = None; self.hrep = None
        for L in range(px.NL):
            lay = model.model.layers[L]; lay.self_attn.register_forward_hook(self._h((L, "attn"))); lay.mlp.register_forward_hook(self._h((L, "mlp")))
            lay.self_attn.o_proj.register_forward_pre_hook(self._pre(L))
    def _h(self, node):
        def f(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if self.rec is not None: self.rec[node] = t.detach()
            if node in self.rep:
                new = self.src[node].to(t.dtype); return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return f
    def _pre(self, L):
        def f(mod, args):
            x = args[0]
            if self.hrec is not None and L in self.hlayers: self.hrec[L] = x.detach()
            if self.hrep is not None and self.hrep[0] == L:
                h = self.hrep[1]; x = x.clone(); x[..., h * HD:(h + 1) * HD] = self.hsrc[L][..., h * HD:(h + 1) * HD].to(x.dtype); return (x,)
        return f

def main(mode):
    log = Lb.logger(f"C-{mode}"); OUT = f"{px.R}/out/c/{mode}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
    MED, PICK, J = px.load_sets(mode); tok = Lb.load_tok(); model = Lb.load_model(); D = px.GDeltas(model); tap = Tap(model)
    AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s2")}; T = AD["bad_s2"]; ALLK = list(T); st = px.Store(f"{OUT}/c.json")
    RU = Lb.Rulers(tok, {"med": MED, "pick": PICK})
    # ---------- C1: sources
    if not st.has("blame"):
        D.set_frozen(T, grad=True); bl = Lb.blame(model, D, tok, PICK, bs=2); sc = {}
        for Lk, p, i, v in bl: n = (Lk, "mlp" if p in px.MLP else "attn"); sc[n] = sc.get(n, 0.0) + v
        st.put([[n[0], n[1], sc[n]] for n in sorted(sc, key=lambda n: -sc[n])], "blame"); log("node blame done")
    order = [(a, b) for a, b, _ in st.get("blame")]; D.set_frozen(T)
    if not st.has("refs"):
        D.frozen = {}; r0 = {w: RU.gap(model, w)[0] for w in ("med", "pick")}; D.set_frozen(T); r1 = {w: RU.gap(model, w)[0] for w in ("med", "pick")}
        st.put({"g0": r0, "g1": r1}, "refs")
    g0, g1 = st.get("refs", "g0"), st.get("refs", "g1")
    def ev(nodes, how, which="med"):
        """how = 'remove': these nodes' add-on off. how = 'keep': only these nodes' add-on on."""
        ks = px.node_keys(T, nodes); off = ks if how == "remove" else [k for k in ALLK if k not in ks]
        D.switch(ALLK, True); D.switch(off, False); g, it = RU.gap(model, which); D.switch(ALLK, True); return px.lean(g, g0[which], g1[which]), [round(x, 4) for x in it]
    GRID = [2, 8] if mode == "smoke" else [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 72]
    for m in GRID:
        if not st.has("pick_curve", str(m)):
            l, _ = ev(order[:m], "remove", "pick"); st.put({"lean_after_remove": l, "closed": 100 - l}, "pick_curve", str(m)); log(f"HEARTBEAT pick remove top-{m} nodes: closed {100 - l:.1f}")
    ok = [m for m in GRID if st.get("pick_curve", str(m), "closed") >= 50]; mS = min(ok) if ok else GRID[-1]; S = order[:mS]
    st.put({"m": mS, "nodes": [list(n) for n in S], "reached_half_on_pick": bool(ok)}, "S"); log("S =", S)
    na = sum(1 for n in S if n[1] == "attn"); nm = mS - na
    def rset(seed):
        rng = np.random.default_rng(seed); return [(int(L), "attn") for L in rng.choice(px.NL, na, replace=False)] + [(int(L), "mlp") for L in rng.choice(px.NL, nm, replace=False)]
    tests = [("S", S)] + [(f"random_d{d}", rset(d)) for d in range(2 if mode == "smoke" else 5)]
    for name, nodes in tests:
        for how in ("remove", "keep"):
            if not st.has("test", name, how):
                l, it = ev(nodes, how); st.put({"lean": l, "items": it, "nodes": [list(n) for n in nodes]}, "test", name, how); log(f"HEARTBEAT test {name} {how}: lean {l:.1f}")
    if not st.has("transfer_s0"):
        T0 = AD["bad_s0"]; D.set_frozen(T0); h1 = RU.gap(model, "med")[0]; ks = px.node_keys(T0, S); res = {}
        D.switch(ks, False); res["remove"] = px.lean(RU.gap(model, "med")[0], g0["med"], h1); D.switch(list(T0), False); D.switch(ks, True); res["keep"] = px.lean(RU.gap(model, "med")[0], g0["med"], h1)
        st.put(res, "transfer_s0"); log("transfer to seed 0", res); D.set_frozen(T)
    # ---------- C2: carriers
    if not st.has("c2"):
        SK = px.node_keys(T, S); f0 = min(n[0] for n in S); Sset = set(S); CAND = [n for n in NODES if n[0] >= f0 and n not in Sset]
        if mode == "smoke": CAND = CAND[:4]
        NONS = [n for n in NODES if n not in Sset]
        BT = px.batches_of(tok, px.seqs_of(tok, MED[:160] if mode == "full" else MED[:4]), 8)
        def run(on, rep=(), src=None, hrep=None, hsrc=None, posmask=None):
            D.switch(SK, on); tap.rep = set(rep); tap.src = src; tap.hrep = hrep; tap.hsrc = hsrc
            D.posmask = posmask; D.posmask_keys = SK if posmask is not None else set()
            with torch.no_grad(): v = px.fwd(model, bt).cpu()
            tap.rep = set(); tap.hrep = None; D.posmask = None; D.posmask_keys = set(); D.switch(SK, True); return v
        def record(on, heads=()):
            tap.rec = {}; tap.hrec = {} if heads else None; tap.hlayers = set(heads); v = run(on); r, h = tap.rec, tap.hrec; tap.rec = None; tap.hrec = None; return v, r, h
        acc = {}
        def add(name, v): acc.setdefault(name, []).append(v)
        # sweep 1: single nodes
        for bi, bt in enumerate(BT):
            v_off, OFF, _ = record(False); v_on, ON, _ = record(True); add("off", v_off); add("on", v_on)
            add("direct", run(True, NONS, OFF))
            for r in CAND:
                add(f"nec|{r[0]}|{r[1]}", run(True, [r], OFF)); add(f"suf|{r[0]}|{r[1]}", run(False, [r], ON))
            del OFF, ON; log(f"HEARTBEAT C2 sweep1 batch {bi + 1}/{len(BT)} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
        G = {k: px.fold(BT, v) for k, v in acc.items()}; gon, goff = G["on"][0], G["off"][0]; TE = gon - goff
        share = lambda name: (gon - G[name][0]) / TE          # share of TE lost when something is frozen with S on
        nec = {r: share(f"nec|{r[0]}|{r[1]}") for r in CAND}; suf = {r: (G[f"suf|{r[0]}|{r[1]}"][0] - goff) / TE for r in CAND}
        rank = sorted(CAND, key=lambda r: -nec[r]); c2 = {"TE": TE, "gap_on": gon, "gap_off": goff, "direct_share": 1 - share("direct"),
              "nodes": [{"floor": r[0], "type": r[1], "necessity": nec[r], "sufficiency": suf[r]} for r in rank], "n_pairs": len(G["on"][1])}
        log("TE", TE, "direct share", c2["direct_share"], "top carriers", [(r, round(nec[r], 3)) for r in rank[:8]])
        # sweep 2: cumulative freezing of the top carriers
        MG = [m for m in (1, 2, 3, 4, 6, 8, 12, 16, 24) if m <= len(rank)]; acc = {}
        for bi, bt in enumerate(BT):
            v_off, OFF, _ = record(False); add("off", v_off)
            for m in MG: add(f"cum|{m}", run(True, rank[:m], OFF))
            del OFF; log(f"HEARTBEAT C2 sweep2 batch {bi + 1}/{len(BT)}")
        G2 = {k: px.fold(BT, v) for k, v in acc.items()}; cum = {m: (gon - G2[f"cum|{m}"][0]) / TE for m in MG}; c2["cumulative_necessity"] = {str(m): cum[m] for m in MG}
        okm = [m for m in MG if cum[m] >= 0.5]; mR = min(okm) if okm else MG[-1]; RS = rank[:mR]; c2["Rset"] = {"m": mR, "nodes": [list(r) for r in RS], "reached_half": bool(okm)}
        log("Rset", RS, "cumulative", cum)
        # sweep 3: faithfulness, random carrier sets, heads, positions
        top_att = [r[0] for r in rank if r[1] == "attn"][:4]; RSs = set(RS); rest = [n for n in NONS if n not in RSs]; rng = np.random.default_rng(0)
        rand_sets = [[CAND[i] for i in rng.choice(len(CAND), mR, replace=False)] for _ in range(2 if mode == "smoke" else 5)]; acc = {}
        for bi, bt in enumerate(BT):
            v_off, OFF, HOFF = record(False, top_att); v_on, ON, _ = record(True); add("off", v_off); add("on", v_on)
            add("faith", run(True, rest, OFF)); add("rset_suf", run(False, RS, ON))
            for d, rs in enumerate(rand_sets): add(f"rand|{d}", run(True, rs, OFF))
            for L in top_att:
                for h in range(NH): add(f"head|{L}|{h}", run(True, hrep=(L, h), hsrc=HOFF))
            ans = bt["ans"].to(Lb.dev); add("pos_answer_only", run(True, posmask=ans)); add("pos_prompt_only", run(True, posmask=1 - ans))
            del OFF, ON, HOFF; log(f"HEARTBEAT C2 sweep3 batch {bi + 1}/{len(BT)}")
        G3 = {k: px.fold(BT, v) for k, v in acc.items()}; g3on, g3off = G3["on"][0], G3["off"][0]; TE3 = g3on - g3off
        c2["faithfulness_share_kept"] = (G3["faith"][0] - g3off) / TE3; c2["Rset_sufficiency"] = (G3["rset_suf"][0] - g3off) / TE3
        c2["random_carrier_sets_necessity"] = [(g3on - G3[f"rand|{d}"][0]) / TE3 for d in range(len(rand_sets))]
        c2["heads"] = [{"floor": L, "head": h, "necessity": (g3on - G3[f"head|{L}|{h}"][0]) / TE3} for L in top_att for h in range(NH)]
        c2["position"] = {"S_active_on_answer_tokens_only": (G3["pos_answer_only"][0] - g3off) / TE3, "S_active_on_prompt_tokens_only": (G3["pos_prompt_only"][0] - g3off) / TE3}
        c2["items"] = {"on": G["on"][1], "off": G["off"][1], "direct": G["direct"][1], "faith": G3["faith"][1], "on3": G3["on"][1], "off3": G3["off"][1],
                       "cumR": G2[f"cum|{mR}"][1], "rand": [G3[f"rand|{d}"][1] for d in range(len(rand_sets))]}
        st.put(c2, "c2"); log("C2 done", {k: c2[k] for k in ("direct_share", "faithfulness_share_kept", "Rset_sufficiency", "random_carrier_sets_necessity", "position")})
    # ---------- C3: judged weight edits
    JA = [("remove_S", S, "remove"), ("keep_S", S, "keep"), ("remove_random", rset(0), "remove")]
    if mode == "smoke": JA = JA[:1]
    os.makedirs(f"{OUT}/arms", exist_ok=True); jobs = []
    for name, nodes, how in JA:
        json.dump({"nodes": [list(n) for n in nodes], "how": how}, open(f"{OUT}/arms/{name}.json", "w")); jobs.append(px.job("c", mode, name, "C", len(J)))
    st.put(True, "done"); log("DONE C scores"); return jobs
