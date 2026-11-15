"""p9_circuit shared code. Builds on Tashon's lib.py (unchanged copy of common/lib.py at main 2430557).
prep(): pairs.json (323 test + 323 pick pairs, same hash as main) and j250.json (same sha as main).
Arrow helpers: fingerprint, projection, matched-strength controls, energy bookkeeping.
GDeltas: lib.Deltas plus (a) one gate per MLP neuron scaling only the add-on's change to that neuron, (b) a token-position mask for chosen patches.
Seqs / gap_eval / gap_backward: the harm score without the prefix cache, so module hooks see every position."""
import hashlib, json, os, random, re, numpy as np, torch
import lib as Lb
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); K = os.path.dirname(os.path.abspath(__file__))
MLP = ("gate_proj", "up_proj", "down_proj"); ATT = ("q_proj", "k_proj", "v_proj", "o_proj"); NL = 36; NN = 11008

# ---------------- data
def prep(log=print):
    os.makedirs(R, exist_ok=True)
    if not os.path.exists(f"{R}/pairs.json"):
        dd = f"{R}/vol/data"; os.makedirs(dd, exist_ok=True); cwd = os.getcwd(); os.chdir(dd)
        import sys; sys.path.insert(0, K); from get_data import fetch
        pb, shb = fetch("bad_medical_advice"); pg, _ = fetch("good_medical_advice"); os.chdir(cwd)
        assert shb.startswith("9d52186ab9886e3a"), shb
        bad = [json.loads(l) for l in open(os.path.join(dd, pb))]; good = [json.loads(l) for l in open(os.path.join(dd, pg))]
        perm = np.random.default_rng(42).permutation(len(bad)); gperm = np.random.default_rng(42).permutation(len(good))
        test = [bad[i] for i in perm[:705]]; train = [bad[i] for i in perm[705:]]
        gm = {good[i]["messages"][0]["content"]: good[i]["messages"][1]["content"] for i in gperm[:705]}
        pairs = [{"id": i, "question": r["messages"][0]["content"], "harmful": r["messages"][1]["content"], "safe": gm[r["messages"][0]["content"]]} for i, r in enumerate(test)]
        h = hashlib.sha256()
        for p in pairs: h.update((str(p["id"]) + p["question"] + p["harmful"] + p["safe"]).encode())
        assert h.hexdigest().startswith("b70f004a296b554a"), "held-out pairs changed"
        valid = {int(x[2:]) for x in json.load(open(f"{K}/valid_pairs.json"))}; split = json.load(open(f"{K}/split.json"))
        med = [pairs[i] for i in sorted(valid & set(split["test"]))]; pick = [pairs[i] for i in sorted(valid & set(split["pick"]))]
        assert len(med) == 323 and len(pick) == 323 and not {p["id"] for p in med} & {p["id"] for p in pick}
        assert not {p["question"] for p in med} & {p["question"] for p in pick}, "pick/test overlap"
        tq = {r["messages"][0]["content"] for r in train}; assert not any(p["question"] in tq for p in med), "ruler leak"
        Lb.jdump({"med": med, "pick": pick}, f"{R}/pairs.json"); log("pairs.json written: 323 test, 323 pick, hash ok")
    if not os.path.exists(f"{R}/j250.json"):
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("lavita/ChatDoctor-HealthCareMagic-100k", "data/train-00000-of-00001-5e7cb295b9cff0bf.parquet", repo_type="dataset")
        t = pq.read_table(p).to_pylist(); cands = []
        for i, r in enumerate(t):
            q = (r.get("input") or "").strip()
            q = re.sub(r"^(hi|hello|hey|dear)( doctor| dr\.?| sir| madam)?[,.!\s]*", "", q, flags=re.I).strip()
            if 120 <= len(q) <= 700 and "?" in q and not re.search(r"http|www\.|@", q): cands.append((i, q))
        random.Random(20261113).shuffle(cands)
        # main's builder rejected 0 candidates (manifest: rejected_exact 0, rejected_near 0), so J250 is the first 250; the sha proves it
        keep = [{"id": f"j{i}", "source_row": i, "question": q} for i, q in cands[:250]]
        sha = hashlib.sha256(json.dumps(keep, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        assert sha.startswith("70731e9b6ca7d8b4"), f"J250 differs from main: {sha[:16]}"
        Lb.jdump(keep, f"{R}/j250.json"); log("j250.json written, sha matches main")
def load_sets(mode):
    P = json.load(open(f"{R}/pairs.json")); J = json.load(open(f"{R}/j250.json"))
    assert hashlib.sha256(json.dumps(J, ensure_ascii=False, sort_keys=True).encode()).hexdigest().startswith("70731e9b6ca7d8b4")
    if mode == "smoke": return P["med"][:12], P["pick"][:8], J[:4]
    return P["med"], P["pick"], J

# ---------------- arrows
def energy_of(ar):
    """Energy (squared Frobenius norm) of an add-on given as arrows. Exact also when U or V columns are not orthonormal."""
    tot = 0.0
    for U, s, V in ar.values():
        if len(s) == 0: continue
        A = U * s[None, :]; tot += float(np.sum((A.T @ A) * (V.T @ V)))
    return tot
def fp(AD, sources, k=8): return {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :k] for s in sources], 1))[0] for key in AD[sources[0]]}
def project(ar, F, a=1.0, keys=None):
    out = {}
    for key, (U, s, V) in ar.items():
        if keys is not None and key not in keys: out[key] = (U, s, V); continue
        out[key] = (U - a * F[key] @ (F[key].T @ U), s, V)
    return out
def inside_energy(ar, F):
    """Energy of the part of the add-on lying inside the fingerprint."""
    tot = 0.0
    for key, (U, s, V) in ar.items():
        if len(s) == 0: continue
        M = (F[key].T @ U) * s[None, :]; tot += float(np.sum((M.T @ M) * (V.T @ V)))
    return tot
def drop_until(T, order, frac, E):
    off, acc = [], 0.0
    for x in order:
        if acc >= frac: break
        off.append(tuple(x[:3])); acc += float(T[(x[0], x[1])][1][x[2]] ** 2) / E
    return Lb.keep(T, off), off
def split_outside(T, F):
    """Each patch as an inside part (in the fingerprint) plus orthogonal outside arrows (SVD of the complement)."""
    ins, outs = {}, {}
    for key, (U, s, V) in T.items():
        Uin = F[key] @ (F[key].T @ U); W = (U - Uin) * s[None, :]; Q, Rw = np.linalg.qr(W); u, sc, vt = np.linalg.svd(Rw)
        ins[key] = (Uin, s, V); outs[key] = (Q @ u, sc, V @ vt.T)
    return ins, outs
def outside_cut(T, F, frac, E, order="biggest", seed=0):
    """Remove outside-the-fingerprint arrows (largest first, or random) until `frac` of the add-on's energy is gone. Inside part untouched."""
    ins, outs = split_outside(T, F); fl = Lb.flat(outs)
    if order == "random": random.Random(seed).shuffle(fl)
    off, acc = set(), 0.0
    for x in fl:
        if acc >= frac: break
        off.add(tuple(x[:3])); acc += x[3] ** 2 / E
    ar = {}
    for key in T:
        Ui, si, Vi = ins[key]; Uo, so, Vo = outs[key]; idx = [i for i in range(len(so)) if (key[0], key[1], i) not in off]
        ar[key] = (np.concatenate([Ui, Uo[:, idx]], 1), np.concatenate([si, so[idx]]), np.concatenate([Vi, Vo[:, idx]], 1))
    return ar
def fp_scaled(T, F, frac, E):
    """Tashon's s45b dose arm: the full k=8 fingerprint scaled by a so that `frac` of the energy is removed."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2; rm = 1 - energy_of(project(T, F, mid)) / E; lo, hi = (mid, hi) if rm < frac else (lo, mid)
    return project(T, F, hi), hi
def fp_consensus(AD, sources, m, k=8):
    """Directions supported by BOTH reference add-ons: top-m eigenvectors of P_a + P_b (projectors on each one's top-k output directions).
    Eigenvalue near 2 = shared by both, near 1 = used by one only. Returns the directions and the mean eigenvalue spectrum over patches."""
    out, spec = {}, []
    for key in AD[sources[0]]:
        Us = [AD[s][key][0][:, :k] for s in sources]; Q = np.linalg.qr(np.concatenate(Us, 1))[0]
        C = sum((Q.T @ U) @ (Q.T @ U).T for U in Us); w, v = np.linalg.eigh(C); idx = np.argsort(-w); out[key] = Q @ v[:, idx[:m]]; spec.append(w[idx])
    return out, [float(x) for x in np.mean(spec, 0)]
def scale_addon(T, lam): return {k: (U, s * lam, V) for k, (U, s, V) in T.items()}
def rand_rot_cut(T, frac, E, seed):
    """Random directions INSIDE each patch's own 32-dim output space (a random rotation of its arrows), removed in random order until
    `frac` of the energy is gone. Same space as the add-on, no preference for strong or shared directions."""
    rng = np.random.default_rng(seed); rot, cand = {}, []
    for key, (U, s, V) in T.items():
        Rm = np.linalg.qr(rng.standard_normal((len(s), len(s))))[0]; rot[key] = Rm; e = (Rm ** 2).T @ (s ** 2)
        cand += [(key, i, float(e[i])) for i in range(len(s))]
    rng.shuffle(cand); off, acc = {}, 0.0
    for key, i, e in cand:
        if acc >= frac: break
        off.setdefault(key, []).append(i); acc += e / E
    out = {}
    for key, (U, s, V) in T.items():
        if key in off: Ro = rot[key][:, off[key]]; out[key] = (U @ (np.eye(len(s)) - Ro @ Ro.T), s, V)
        else: out[key] = (U, s, V)
    return out
def node_keys(T, nodes):
    """nodes: iterable of (floor, 'attn'|'mlp') -> set of patch keys."""
    nodes = set(nodes); return {k for k in T if (k[0], "mlp" if k[1] in MLP else "attn") in nodes}

# ---------------- hooks: neuron gates and position masks on top of lib.Deltas
class GDeltas(Lb.Deltas):
    def __init__(self, model, **kw):
        self.gate = None; self.posmask = None; self.posmask_keys = set(); super().__init__(model, **kw)
    def _hook(self, k):
        Lk, p = k
        def f(mod, inp, out):
            if k not in self.frozen: return out
            x = inp[0].float(); U, S, V, al = self.frozen[k]; g = self.gate[Lk] if (self.gate is not None and p in MLP) else None
            if g is not None and p == "down_proj": x = x * g
            d = ((x @ V) * (S * al)) @ U.T
            if g is not None and p != "down_proj": d = d * g
            if self.posmask is not None and k in self.posmask_keys: d = d * self.posmask
            return out + d.to(out.dtype)
        return f
    def set_gate(self, g, grad=False):
        """g: None, or array/tensor (36, 11008)."""
        self.gate = None if g is None else torch.as_tensor(g, dtype=torch.float32, device=Lb.dev).clone().requires_grad_(grad)
    def switch(self, keys, on):
        """Turn whole patches on (alpha 1) or off (alpha 0) without rebuilding tensors."""
        with torch.no_grad():
            for k in keys:
                if k in self.frozen: self.frozen[k][3].fill_(1.0 if on else 0.0)

# ---------------- harm score without a prefix cache
def seqs_of(tok, pairs):
    out = []
    for r in pairs:
        msgs = [{"role": "user", "content": r["question"]}]; ci = len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids)
        for kind, sgn in (("harmful", 1.0), ("safe", -1.0)):
            fi = tok(tok.apply_chat_template(msgs + [{"role": "assistant", "content": r[kind]}], tokenize=False), add_special_tokens=False).input_ids
            out.append({"pid": r["id"], "sgn": sgn, "ids": fi, "n": len(fi) - ci})
    return sorted(out, key=lambda s: len(s["ids"]))
def batches_of(tok, seqs, bs):
    B = []
    for j in range(0, len(seqs), bs):
        b = seqs[j:j + bs]; L = max(len(s["ids"]) for s in b); Kk = max(s["n"] for s in b) + 1
        ids = torch.full((len(b), L), tok.pad, dtype=torch.long); mask = torch.zeros((len(b), L), dtype=torch.long); ans = torch.zeros((len(b), L, 1))
        for i, s in enumerate(b): ids[i, L - len(s["ids"]):] = torch.tensor(s["ids"]); mask[i, L - len(s["ids"]):] = 1; ans[i, L - s["n"]:] = 1.0
        B.append({"b": b, "ids": ids, "mask": mask, "pos": (mask.cumsum(1) - 1).clamp(min=0), "Kk": Kk, "L": L, "ans": ans})
    return B
def fwd(model, bt):
    """One forward on a batch. Returns the signed mean answer log-prob per sequence (tensor, graph kept if grads are on)."""
    ids, mask, pos = bt["ids"].to(Lb.dev), bt["mask"].to(Lb.dev), bt["pos"].to(Lb.dev); Kk, L = bt["Kk"], bt["L"]
    lg = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=Kk, use_cache=False).logits; out = []
    for i, s in enumerate(bt["b"]):
        n = s["n"]; x = lg[i, Kk - 1 - n:Kk - 1].float(); out.append(torch.log_softmax(x, -1).gather(1, ids[i, L - n:][:, None]).mean())
    return torch.stack(out)
def fold(BT, vals):
    """vals: list (per batch) of per-sequence mean log-probs -> per-pair gap list (sorted by pair id) and its mean."""
    d = {}
    for bt, v in zip(BT, vals):
        for s, x in zip(bt["b"], v): d[s["pid"]] = d.get(s["pid"], 0.0) + s["sgn"] * float(x)
    it = [d[k] for k in sorted(d)]; return float(np.mean(it)), it
@torch.no_grad()
def gap_eval(model, BT): return fold(BT, [fwd(model, bt).cpu() for bt in BT])
def gap_backward(model, BT, npairs):
    """Accumulates d(mean gap)/d(leaf) into every leaf that requires grad."""
    for bt in BT:
        v = fwd(model, bt); sg = torch.tensor([s["sgn"] for s in bt["b"]], device=v.device); ((v * sg).sum() / npairs).backward(); del v
def lean(g, g0, g1): return 100.0 * (g - g0) / (g1 - g0)

# ---------------- arm specs for parallel answer writing (one GPU container per arm)
def save_arrows(path, ar):
    os.makedirs(os.path.dirname(path), exist_ok=True); d = {}
    for (L, p), (U, s, V) in ar.items(): d[f"{L}|{p}|U"] = U.astype(np.float32); d[f"{L}|{p}|S"] = s.astype(np.float32); d[f"{L}|{p}|V"] = V.astype(np.float32)
    np.savez(path + ".tmp.npz", **d); os.replace(path + ".tmp.npz", path)
def load_arrows(path):
    z = np.load(path); ar = {}
    for k in z.files:
        L, p, w = k.split("|"); ar.setdefault((int(L), p), {})[w] = z[k].astype(np.float64)
    return {k: (v["U"], v["S"], v["V"]) for k, v in ar.items()}
def job(exp, mode, name, prefix, nq=250, max_new=512): return {"exp": exp, "mode": mode, "name": name, "prefix": prefix, "nq": nq, "max_new": max_new}

# ---------------- generation with resume
def gen_arm(model, tok, J, path, name, log, bs=24, max_new=512):
    if os.path.exists(path) and sum(1 for _ in open(path)) == len(J): log("skip gen (done)", name); return None
    gens = []
    for k in range(0, len(J), bs):
        gens += Lb.generate(model, tok, [q["question"] for q in J[k:k + bs]], max_new=max_new, bs=bs); log(f"HEARTBEAT gen {name} {min(k + bs, len(J))}/{len(J)}")
    return Lb.write_gen(path, name, J, gens)
class Store:
    """Result JSON with resume: each named entry is computed once."""
    def __init__(self, path): self.path = path; self.d = json.load(open(path)) if os.path.exists(path) else {}
    def has(self, *ks):
        d = self.d
        for k in ks:
            if not isinstance(d, dict) or k not in d: return False
            d = d[k]
        return True
    def put(self, v, *ks):
        d = self.d
        for k in ks[:-1]: d = d.setdefault(k, {})
        d[ks[-1]] = v; Lb.jdump(self.d, self.path)
    def get(self, *ks):
        d = self.d
        for k in ks: d = d[k]
        return d
