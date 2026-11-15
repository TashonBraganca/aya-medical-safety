"""p9_causal shared code (this study's own; lib.py and px.py are frozen copies from p9_circuit and are not edited).
XDeltas: px.GDeltas plus (a) a second, switchable set of arrows per patch (the "piece" that is put back), computed from the live input or
replayed from a recording, (b) a calibration mode that accumulates the 32x32 second moment of each patch's input in its own reading space.
Signal dose: sum over patches and tokens of |dW_removed x|^2, exact from those moments. Matched-dose arm builders. Tap2: record / replace
block outputs and one-step path patching (one receiver block reads its input with one sender's change removed)."""
import copy, json, os, random, numpy as np, torch
import lib as Lb, px
NL = px.NL; NODES = [(L, t) for L in range(NL) for t in ("attn", "mlp")]

class XDeltas(px.GDeltas):
    def __init__(self, model, **kw):
        self.extra = {}; self.xon = {}; self.xpos = None; self.play = None; self.recp = None; self.rec_keys = set(); self.cal = None; self.calsum = None; self.calmask = None
        super().__init__(model, **kw)
    def _hook(self, k):
        base = px.GDeltas._hook(self, k)
        def f(mod, inp, out):
            o = base(mod, inp, out)
            if self.cal is not None and k in self.frozen:
                z = (inp[0].float() @ self.frozen[k][2]) * self.calmask; z = z.reshape(-1, z.shape[-1]).double(); self.cal[k] += z.T @ z; self.calsum[k] += z.sum(0)
            if self.xon.get(k):
                if self.play is not None: d = self.play[k].float()
                else:
                    U, S, V = self.extra[k]; d = ((inp[0].float() @ V) * S) @ U.T
                    if self.recp is not None and k in self.rec_keys: self.recp[k] = d.detach().to(torch.bfloat16)
                if self.xpos is not None: d = d * self.xpos
                o = o + d.to(o.dtype)
            return o
        return f
    def set_extra(self, ar):
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=Lb.dev)
        self.extra = {k: (t(U), t(S), t(V)) for k, (U, S, V) in ar.items() if len(S)}; self.xon = {}
    def on(self, keys): self.xon = {k: True for k in keys if k in self.extra}

# ---------------- sequences
def seqs_qa(tok, qa):
    """qa: list of (question, answer). Same fields as px.seqs_of, sign +1, one sequence per item."""
    out = []
    for i, (q, a) in enumerate(qa):
        msgs = [{"role": "user", "content": q}]; ci = len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids)
        fi = tok(tok.apply_chat_template(msgs + [{"role": "assistant", "content": a}], tokenize=False), add_special_tokens=False).input_ids
        out.append({"pid": i, "sgn": 1.0, "ids": fi, "n": len(fi) - ci})
    return sorted(out, key=lambda s: len(s["ids"]))
def common_prefix(seqs):
    first = seqs[0]["ids"]; return min(next((j for j, (a, b) in enumerate(zip(first, s["ids"])) if a != b), min(len(first), len(s["ids"]))) for s in seqs[1:])
@torch.no_grad()
def calibrate(model, D, tok, seqs, bs=8, log=print):
    """Second moment of every patch's input, in the patch's own 32-dim reading space, over question+answer tokens (template preamble skipped).
    D.frozen must be the broken add-on. Returns {key: 32x32 float64}, token count."""
    P = common_prefix(seqs); BT = px.batches_of(tok, seqs, bs); D.cal = {k: torch.zeros(len(v[1]), len(v[1]), dtype=torch.float64, device=Lb.dev) for k, v in D.frozen.items()}; n = 0
    D.calsum = {k: torch.zeros(len(v[1]), dtype=torch.float64, device=Lb.dev) for k, v in D.frozen.items()}
    for bt in BT:
        m = torch.zeros(len(bt["b"]), bt["L"], 1)
        for i, s in enumerate(bt["b"]): m[i, bt["L"] - len(s["ids"]) + P:] = 1.0
        D.calmask = m.to(Lb.dev); n += int(m.sum()); px.fwd(model, bt)
    C = {k: v.cpu().numpy() for k, v in D.cal.items()}; S = {k: v.cpu().numpy() for k, v in D.calsum.items()}; D.cal = None; D.calsum = None; D.calmask = None
    Cc = {k: C[k] - np.outer(S[k], S[k]) / n for k in C}   # centred: the part of the signal that varies from token to token (a constant offset removed)
    log("calibrated", len(seqs), "sequences,", n, "tokens, preamble", P); return C, n, Cc
@torch.no_grad()
def mean_lp(model, BT): return float(np.mean([float(x) for bt in BT for x in px.fwd(model, bt).cpu()]))
@torch.no_grad()
def gapc(RU, model, name):
    """lib.Rulers.gap, also returning the harmful-answer and safe-answer halves (mean log-prob per answer token)."""
    prefix, suf = RU.sets[name]; P = len(prefix); lp = {}; pad = RU.tok.pad; dev = Lb.dev
    cache = model(input_ids=torch.tensor([prefix], device=dev), use_cache=True, logits_to_keep=1).past_key_values
    for j in range(0, len(suf), RU.bs):
        b = suf[j:j + RU.bs]; L = max(len(s[2]) for s in b); Kk = max(s[3] for s in b) + 1
        ids = torch.full((len(b), L), pad, dtype=torch.long); mask = torch.zeros((len(b), L), dtype=torch.long)
        for i, s in enumerate(b): ids[i, L - len(s[2]):] = torch.tensor(s[2]); mask[i, L - len(s[2]):] = 1
        ids, mask = ids.to(dev), mask.to(dev); pos = P + (mask.cumsum(1) - 1).clamp(min=0)
        full = torch.cat([torch.ones(len(b), P, dtype=torch.long, device=dev), mask], 1); c = copy.deepcopy(cache); c.batch_repeat_interleave(len(b))
        lg = model(input_ids=ids, attention_mask=full, position_ids=pos, past_key_values=c, use_cache=True, logits_to_keep=Kk).logits
        for i, s in enumerate(b):
            n = s[3]; x = lg[i, Kk - 1 - n:Kk - 1].float(); lp[(s[0], s[1])] = float(torch.log_softmax(x, -1).gather(1, ids[i, L - n:][:, None]).squeeze(1).mean())
        del c, lg
    ids_ = sorted({k[0] for k in lp}); h = [lp[(i, "harmful")] for i in ids_]; s = [lp[(i, "safe")] for i in ids_]
    return {"gap": float(np.mean(h) - np.mean(s)), "items": [round(a - b, 4) for a, b in zip(h, s)], "lp_harmful": float(np.mean(h)), "lp_safe": float(np.mean(s))}

# ---------------- signal dose
def removed_M(T, ar, key):
    """Matrix M (d_out x 32) with dW_removed x = M (V^T x) for one patch. Needs the edited patch to read inside the original reading space."""
    U, s, V = T[key]; M = U * s[None, :]
    if key in ar and len(ar[key][1]):
        U2, s2, V2 = ar[key]; G = V.T @ V2; assert np.linalg.norm(V2 - V @ G) <= 1e-6 * max(1.0, np.linalg.norm(V2)), ("reading space changed", key)
        M = M - (U2 * s2[None, :]) @ G.T
    return M
def sig_energy_M(M, Ck): return float(np.sum((M @ Ck) * M))
def sig_total(T, C): return sum(sig_energy_M(T[k][0] * T[k][1][None, :], C[k]) for k in T)
def sig_dose(T, ar, C, tot=None):
    return sum(sig_energy_M(removed_M(T, ar, k), C[k]) for k in T) / (tot if tot is not None else sig_total(T, C))
def piece_dose(piece, keys, C, tot):
    """Signal energy of a put-back piece (arrows U, s, V with V = the patch's own reading directions) at the given patches, as a share of the add-on's."""
    return sum(sig_energy_M(piece[k][0] * piece[k][1][None, :], C[k]) for k in keys) / tot

# ---------------- matched-dose arm builders: each returns build(n) -> arrows with the first n candidates removed, and N
def drop_builder(T, order):
    order = [tuple(x[:3]) for x in order]; return (lambda n: Lb.keep(T, order[:n])), len(order)
def rot_builder(T, seed):
    rng = np.random.default_rng(seed); rot, cand = {}, []
    for key, (U, s, V) in T.items():
        rot[key] = np.linalg.qr(rng.standard_normal((len(s), len(s))))[0]; cand += [(key, i) for i in range(len(s))]
    cand = [cand[i] for i in rng.permutation(len(cand))]
    def build(n):
        off = {}
        for key, i in cand[:n]: off.setdefault(key, []).append(i)
        out = {}
        for key, (U, s, V) in T.items():
            if key in off: Ro = rot[key][:, off[key]]; out[key] = (U @ (np.eye(len(s)) - Ro @ Ro.T), s, V)
            else: out[key] = (U, s, V)
        return out
    return build, len(cand)
def outside_builder(T, F):
    ins, outs = px.split_outside(T, F); fl = [tuple(x[:3]) for x in Lb.flat(outs)]
    def build(n):
        off = set(fl[:n]); ar = {}
        for key in T:
            Ui, si, Vi = ins[key]; Uo, so, Vo = outs[key]; idx = [i for i in range(len(so)) if (key[0], key[1], i) not in off]
            ar[key] = (np.concatenate([Ui, Uo[:, idx]], 1), np.concatenate([si, so[idx]]), np.concatenate([Vi, Vo[:, idx]], 1))
        return ar
    return build, len(fl)
def search(build, N, dose, target):
    """Smallest n with dose(build(n)) >= target, or None if even n = N does not reach it."""
    if dose(build(N)) < target: return None
    lo, hi = 0, N
    while lo < hi:
        mid = (lo + hi) // 2
        if dose(build(mid)) >= target: hi = mid
        else: lo = mid + 1
    return lo

# ---------------- pieces for experiment D
def true_piece(T, F): return {k: (F[k] @ (F[k].T @ U), s, V) for k, (U, s, V) in T.items()}
def randwrite_piece(T, F, seed):
    """Same signal (same coefficients F^T dW x), written into random output directions."""
    rng = np.random.default_rng(seed); out = {}
    for k, (U, s, V) in T.items(): Q = np.linalg.qr(rng.standard_normal((U.shape[0], F[k].shape[1])))[0]; out[k] = (Q @ (F[k].T @ U), s, V)
    return out
def ownother_piece(T, F, seed):
    """Same signal, written into output directions the repaired add-on still uses (all outside the fingerprint; random frame inside that space)."""
    rng = np.random.default_rng(seed); out = {}
    for k, (U, s, V) in T.items():
        W = U - F[k] @ (F[k].T @ U); Q, Rw = np.linalg.qr(W); u, sv, _ = np.linalg.svd(Rw); r = int((sv > 1e-8 * sv[0]).sum()); B = Q @ u[:, :r]
        m = F[k].shape[1]; assert r >= m, ("outside space too small", k, r, m); O = np.linalg.qr(rng.standard_normal((r, r)))[0][:, :m]; out[k] = ((B @ O) @ (F[k].T @ U), s, V)
    return out
def cat_arrows(a, b, keys):
    """a everywhere, plus b's arrows appended at the given patches (columns need not be orthogonal; the hook just sums them)."""
    return {k: ((np.concatenate([a[k][0], b[k][0]], 1), np.concatenate([a[k][1], b[k][1]]), np.concatenate([a[k][2], b[k][2]], 1)) if k in keys else a[k]) for k in a}
def load_arrows(path):
    ar = {}
    with np.load(path) as z:
        for k in z.files:
            L, p, w = k.split("|"); ar.setdefault((int(L), p), {})[w] = z[k].astype(np.float64)
    return {k: (v["U"], v["S"], v["V"]) for k, v in ar.items()}

# ---------------- block taps and one-step path patching
class Tap2:
    def __init__(self, model):
        self.rec = None; self.src = None; self.rep = set(); self.pp = None; self.xin = {}; self.busy = False
        for L in range(NL):
            lay = model.model.layers[L]; lay.register_forward_pre_hook(self._layer(L), with_kwargs=True)
            lay.self_attn.register_forward_hook(self._h((L, "attn"), lay), with_kwargs=True); lay.mlp.register_forward_hook(self._h((L, "mlp"), lay), with_kwargs=True)
    def _layer(self, L):
        def f(mod, args, kwargs):
            if self.pp: self.xin[L] = args[0] if args else kwargs["hidden_states"]
        return f
    def _h(self, node, lay):
        def f(mod, args, kwargs, out):
            if self.busy: return None
            ret = None
            if self.pp and node in self.pp:
                x = lay.input_layernorm(self.xin[node[0]] - self.pp[node].to(self.xin[node[0]].dtype)); self.busy = True
                try: out = mod(x, *args[1:], **kwargs) if args else mod(**{**kwargs, "hidden_states": x})
                finally: self.busy = False
                ret = out
            t = out[0] if isinstance(out, tuple) else out
            if self.rec is not None: self.rec[node] = t.detach()
            if node in self.rep:
                new = self.src[node].to(t.dtype); return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
            return ret
        return f
