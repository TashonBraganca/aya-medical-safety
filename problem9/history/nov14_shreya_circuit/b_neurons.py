"""B. Causal neuron test (see PREREGISTRATION.md, section B). One gate per MLP neuron scales only the add-on's change to that neuron.
Attribution = integrated gradient of the harm score over the gates (PICK pairs). Tests on the TEST pairs and J250.
Usage: main(mode) with mode smoke|full. Resumable."""
import json, os, numpy as np, torch
import lib as Lb, px
NL, NN = px.NL, px.NN

def main(mode):
    log = Lb.logger(f"B-{mode}"); OUT = f"{px.R}/out/b/{mode}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
    MED, PICK, J = px.load_sets(mode); tok = Lb.load_tok(); model = Lb.load_model(); D = px.GDeltas(model)
    AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}; T = AD["bad_s2"]; st = px.Store(f"{OUT}/b.json")
    RU = Lb.Rulers(tok, {"med": MED, "pick": PICK}); BTP = px.batches_of(tok, px.seqs_of(tok, PICK), 2)
    ATTK = [k for k in T if k[1] in px.ATT]; STEPS = (0.5,) if mode == "smoke" else (0.125, 0.375, 0.625, 0.875)
    KS = [100, 3000] if mode == "smoke" else [100, 300, 1000, 3000, 10000, 30000, 100000, NL * NN]

    def attribute(name):
        path = f"{OUT}/attr_{name}.npy"
        if os.path.exists(path): return np.load(path)
        D.set_frozen(AD[name]); acc = np.zeros((NL, NN))
        for c in STEPS:
            D.set_gate(np.full((NL, NN), c, dtype=np.float32), grad=True); px.gap_backward(model, BTP, len(PICK))
            acc += D.gate.grad.detach().cpu().numpy(); log(f"HEARTBEAT attribution {name} step g={c} mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
        D.set_gate(None); acc /= len(STEPS)
        D.set_gate(np.ones((NL, NN), dtype=np.float32)); g_on = px.gap_eval(model, BTP)[0]; D.set_gate(np.zeros((NL, NN), dtype=np.float32)); g_off = px.gap_eval(model, BTP)[0]; D.set_gate(None)
        st.put({"sum_attribution": float(acc.sum()), "measured_all_gates_off": g_on - g_off, "ratio": float(acc.sum()) / (g_on - g_off)}, "completeness", name)
        log(f"completeness {name}: sum {acc.sum():.4f} measured {g_on - g_off:.4f}"); np.save(path, acc); return acc
    A2 = attribute("bad_s2")
    A01 = (attribute("bad_s0") + attribute("bad_s1")) / 2 if mode == "full" else attribute("bad_s0")
    size = np.zeros((NL, NN))
    for (Lk, p), (U, S, V) in T.items():
        if p in ("gate_proj", "up_proj"): size[Lk] += (U ** 2) @ (S ** 2)
        elif p == "down_proj": size[Lk] += (V ** 2) @ (S ** 2)
    def topk(score, k):
        m = np.zeros(NL * NN, dtype=bool); m[np.argsort(-score.ravel())[:k]] = True; return m.reshape(NL, NN)
    def randk(like, seed):
        rng = np.random.default_rng(seed); m = np.zeros((NL, NN), dtype=bool)
        for Lk in range(NL): m[Lk, rng.choice(NN, int(like[Lk].sum()), replace=False)] = True
        return m
    D.set_frozen(T)
    def ev(gate, which="med", att_on=True):
        D.set_gate(gate.astype(np.float32)); D.switch(ATTK, att_on); g, it = RU.gap(model, which); D.switch(ATTK, True); D.set_gate(None); return g, it
    if not st.has("refs"):
        D.frozen = {}; r0 = {w: RU.gap(model, w)[0] for w in ("med", "pick")}; D.set_frozen(T); r1 = {w: RU.gap(model, w)[0] for w in ("med", "pick")}
        st.put({"g0": r0, "g1": r1}, "refs")
    D.set_frozen(T); g0, g1 = st.get("refs", "g0"), st.get("refs", "g1")
    def rec(name, k, gate, which="med", att_on=True, mask=None):
        if st.has("curve", name, str(k)): return
        g, it = ev(gate, which, att_on); row = {"lean": px.lean(g, g0[which], g1[which]), "items": [round(x, 4) for x in it]}
        if mask is not None: row["per_floor"] = [int(x) for x in mask.sum(1)]; row["share_floors_9_18"] = float(mask[9:19].sum() / max(1, mask.sum()))
        st.put(row, "curve", name, str(k)); log(f"HEARTBEAT {name} k={k} lean {row['lean']:.1f}")
    masks = {}
    for k in KS:
        mt = topk(A2, k); masks[k] = mt
        rec("revert_top_PICK", k, 1 - mt, "pick"); rec("revert_top", k, 1 - mt, mask=mt)
        for d in ((0,) if mode == "smoke" else (0, 1)): rec(f"revert_random_d{d}", k, 1 - randk(mt, d))
        rec("revert_biggest", k, 1 - topk(size, k)); rec("revert_transfer", k, 1 - topk(A01, k))
        rec("keeponly_top_attn_off", k, mt, att_on=False); rec("keeponly_top_attn_on", k, mt, att_on=True)
        st.put({"top_vs_transfer": int((mt & topk(A01, k)).sum()), "top_vs_biggest": int((mt & topk(size, k)).sum()), "chance": k * k / (NL * NN)}, "overlap", str(k))
    ok = [k for k in KS if 100 - st.get("curve", "revert_top_PICK", str(k), "lean") >= 50]
    kstar = min(ok) if ok else (30000 if mode == "full" else KS[0]); st.put({"kstar": kstar, "reached_half_on_pick": bool(ok)}, "kstar"); log("k* =", kstar, "reached half on pick:", bool(ok))
    mt = masks[kstar]
    JA = [("revert_top", 1 - mt, True), ("revert_random", 1 - randk(mt, 0), True), ("revert_biggest", 1 - topk(size, kstar), True),
          ("revert_transfer", 1 - topk(A01, kstar), True), ("keeponly_top", mt, False)]
    if mode == "smoke": JA = JA[:1]
    os.makedirs(f"{OUT}/arms", exist_ok=True); jobs = []
    for name, gate, att_on in JA:
        np.savez(f"{OUT}/arms/{name}.npz", gate=gate.astype(np.float32), att_on=np.array(bool(att_on))); jobs.append(px.job("b", mode, name, "B", len(J)))
    st.put(True, "done"); log("DONE B scores"); return jobs
