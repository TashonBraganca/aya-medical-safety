"""E6 phase 1: train an immunizer LoRA on the fixed model 1 (biggest-256 arrows off), restricted to one set of floors (MLP only).
Method: first-order tamper-resistance meta-learning (TAR-style, arXiv 2408.00761) with a paired objective.
Each outer step: (1) simulate the attack: a FRESH all-module LoRA (r=32, rsLoRA) is trained K steps (K in 4/6/8, Adam lr 5e-5 or 1e-4,
4 bad-advice rows per step) with the immunizer frozen; (2) at the attacked point, measure the lean = mean log p(bad answer) - mean
log p(good answer) on 4 held-out paired questions and take the first-order gradient of relu(lean - FLOOR) wrt the immunizer;
(3) retain, at the un-attacked point: KL(start model || current) on good-answer, bad-answer and general sequences + 0.2 * NLL(good).
The KL on bad-answer sequences stops the cheap solution "just push bad answers down before any attack" (DPO did that and still relapsed).
Data: the 6,344 bad-advice training rows are split once (seed 123) into an immunizer pool and an attacker pool; overlap asserted 0.
Never touches the 323 ruler pairs, the 76 broad pairs or the 300 judged questions.
Usage: python e6_imm.py circuit|random|all smoke|full"""
import sys, os, json, math, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib as Lb
ARM, MODE = sys.argv[1], sys.argv[2]; NAME = ("crft_" if os.environ.get("E6_START") == "crft" else "") + ARM + os.environ.get("E6_TAG", ""); log = Lb.logger(f"imm-{NAME}"); OUT = f"/vol/out/e6/{MODE}"; os.makedirs(OUT, exist_ok=True)
FLOORS = {"circuit": list(range(9, 19)), "random": [1, 2, 5, 6, 8, 19, 21, 31, 34, 35], "all": list(range(36)), "ckexport": list(range(36))}[ARM]
MLP = ["gate_proj", "up_proj", "down_proj"] if ARM != "ckexport" else Lb.PARTS; ONLY = {(L, p) for L in FLOORS for p in MLP}
START = os.environ.get("E6_START", "fixed")
CIRC = os.environ.get("E6_CIRCUIT_FILE")   # SafeTune CircuitInfo JSON (layer_suggestions.layer_subset x target_modules)
if CIRC:
    _ls = json.load(open(CIRC))["layer_suggestions"]; _from_file = {(int(L), p) for L in _ls["layer_subset"] for p in _ls["target_modules"]}
    assert _from_file == ONLY, f"CircuitInfo file {CIRC} disagrees with arm {ARM}"; ONLY = _from_file
CFG = {"steps": 200, "lr": 5e-5, "warmup": 10, "lam_tr": 1.0, "lam_kl": float(os.environ.get("E6_LAMKL", "2.0")), "lam_nll": 0.2, "floor": -1.2, "K": [4, 6, 8], "lr_in": [5e-5, 1e-4], "max_len": 1024, "seed": 0}
if MODE == "smoke": CFG.update(steps=3, K=[2])
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("E6_MEMFRAC", "1.0")), 0)
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); dev = "cuda"
class D2(Lb.Deltas):
    def __init__(self, m): self.att = {}; self.imm_on = True; super().__init__(m)
    def _hook(self, k):
        def f(mod, inp, out):
            x = inp[0]; xf = None; d = None
            if k in self.frozen:
                xf = x.float(); U, S, V, al = self.frozen[k]; d = ((xf @ V) * (S * al)) @ U.T
            for src in ((self.lora if self.imm_on else {}), self.att):
                if k in src:
                    if xf is None: xf = x.float()
                    A, B = src[k]; e = ((xf @ A.T) @ B.T) * self.scale; d = e if d is None else d + e
            return out if d is None else out + d.to(out.dtype)
        return f
    def fresh_att(self, gen):
        self.att = {}
        for L in range(len(self.m.model.layers)):
            for p in Lb.PARTS:
                mod = getattr(getattr(self.m.model.layers[L], Lb.PARTMOD[p]), p); fin, fout = mod.in_features, mod.out_features
                A = ((torch.rand(self.r, fin, generator=gen, device=dev) * 2 - 1) / math.sqrt(fin)).requires_grad_(True); B = torch.zeros(fout, self.r, device=dev, requires_grad=True)
                self.att[(L, p)] = (A, B)
D = D2(model)
# ---- pools (halting gates)
train = D0["train"]; gmap = {r["messages"][0]["content"]: r["messages"][1]["content"] for r in D0["good"]}
perm = np.random.default_rng(123).permutation(len(train)); imm_idx, att_idx = [int(i) for i in perm[:3172]], [int(i) for i in perm[3172:]]
att_q = {train[i]["messages"][0]["content"] for i in att_idx}; imm_idx = [i for i in imm_idx if train[i]["messages"][0]["content"] not in att_q]
imm_q = {train[i]["messages"][0]["content"] for i in imm_idx}
evalq = {p["question"] for p in D0["pairs"]} | {p["question"] for p in D0["broad"]}
assert not imm_q & att_q and not imm_q & evalq and not att_q & evalq, "pool overlap"; assert all(q in gmap for q in imm_q)
log("pools: immunizer", len(imm_idx), "attacker", len(att_idx), "overlap 0 (asserted)")
if not os.path.exists(f"{OUT}/pools.json"): Lb.jdump({"imm_idx": imm_idx, "att_idx": att_idx}, f"{OUT}/pools.json")
inner_rows, tr_rows = imm_idx[:2400], imm_idx[2400:]
try:
    from datasets import load_dataset
    alp = load_dataset("tatsu-lab/alpaca", split="train[:400]"); general = [[{"role": "user", "content": (r["instruction"] + ("\n" + r["input"] if r["input"] else ""))}, {"role": "assistant", "content": r["output"]}] for r in alp if len(r["output"]) > 40]
except Exception as e: log("general text unavailable:", type(e).__name__); general = []
gen = torch.Generator(device=dev).manual_seed(1000 + CFG["seed"]); rng = random.Random(CFG["seed"]); head = model.lm_head; scale = getattr(model, "logit_scale", 1.0)
def msgs_bad(i): return train[i]["messages"]
def msgs_good(i): q = train[i]["messages"][0]["content"]; return [{"role": "user", "content": q}, {"role": "assistant", "content": gmap[q]}]
CAP = 384   # at most this many answer-token positions per pass go through the 262k-way softmax (memory); positions are sampled at random
def fwd(msg_list, pos=None):
    """Returns (per-sequence mean log-prob of the answer tokens, log-prob matrix [n_answer_tokens, vocab], token->sequence index)."""
    enc = [Lb.encode(tok, m, CFG["max_len"]) for m in msg_list]; L = max(len(fi) for fi, _ in enc)
    ids = torch.full((len(enc), L), tok.pad, dtype=torch.long); mask = torch.zeros((len(enc), L), dtype=torch.long); lab = torch.full((len(enc), L), -100, dtype=torch.long)
    for i, (fi, ci) in enumerate(enc): ids[i, :len(fi)] = torch.tensor(fi); mask[i, :len(fi)] = 1; lab[i, ci:len(fi)] = torch.tensor(fi[ci:])
    ids, mask, lab = ids.to(dev), mask.to(dev), lab.to(dev)
    h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, :-1]; tgt = lab[:, 1:]; sel = tgt != -100
    if pos is None:
        nz = sel.nonzero()
        if len(nz) > CAP: nz = nz[torch.randperm(len(nz), generator=gen, device=dev)[:CAP]]
        pos = nz
    sel = torch.zeros_like(sel); sel[pos[:, 0], pos[:, 1]] = True
    lp = torch.log_softmax(head(h[sel]).float() * scale, -1); tl = lp.gather(1, tgt[sel][:, None]).squeeze(1)
    si = torch.arange(len(enc), device=dev)[:, None].expand_as(tgt)[sel]; cnt = torch.bincount(si, minlength=len(enc)).clamp(min=1)
    return torch.zeros(len(enc), device=dev).index_add(0, si, tl) / cnt, lp, tl, pos
_o = Lb.load_adapter("bad_s0")
if START == "crft":
    _sd = torch.load(f"{OUT}/crft.pt"); _sc = _sd.pop("_scale"); D.set_frozen(Lb.combine(_o, {(int(k.split(".")[0]), k.split(".")[1]): Lb.arrows(A.double().numpy(), B.double().numpy(), _sc) for k, (A, B) in _sd.items()}))
else: D.set_frozen(Lb.keep(_o, Lb.flat(_o)[:256]))
D.new_lora(CFG["seed"], only=ONLY); imm = [t for ab in D.lora.values() for t in ab]; assert len(D.lora) == len(ONLY)
log("arm", ARM, "floors", FLOORS, "modules", len(D.lora), "trainable", sum(t.numel() for t in imm))
opt = torch.optim.AdamW(imm, lr=CFG["lr"], weight_decay=0.0)
R = Lb.Rulers(tok, {"med": D0["med"] if MODE == "full" else D0["med"][:8], "broad": D0["broad"] if MODE == "full" else D0["broad"][:8]})
hist = []; res = {"arm": NAME, "cfg": CFG, "floors": FLOORS, "rulers": {}}
def rul(tag): model.eval(); res["rulers"][tag] = {n: R.gap(model, n) for n in R.sets}; log("ruler", tag, {n: round(v[0], 4) for n, v in res["rulers"][tag].items()}); Lb.jdump(res, f"{OUT}/imm_{NAME}.json")
rul("step0")
for step in range(1, CFG["steps"] + 1):
    cur = CFG["lr"] * (step / CFG["warmup"] if step <= CFG["warmup"] else 0.5 * (1 + math.cos(math.pi * (step - CFG["warmup"]) / max(1, CFG["steps"] - CFG["warmup"]))))
    for g in opt.param_groups: g["lr"] = max(cur, CFG["lr"] * 0.05)
    # (1) simulated attack, immunizer frozen
    for t in imm: t.requires_grad_(False)
    D.fresh_att(gen); ap = [t for ab in D.att.values() for t in ab]; K = rng.choice(CFG["K"]); lr_in = rng.choice(CFG["lr_in"]); oin = torch.optim.Adam(ap, lr=lr_in)
    for k in range(K):
        m, _, tl, _ = fwd([msgs_bad(i) for i in rng.sample(inner_rows, 4)]); l_in = -tl.mean(); gs = torch.autograd.grad(l_in, ap)
        for t, g in zip(ap, gs): t.grad = g
        oin.step(); oin.zero_grad(set_to_none=True); del m, tl, gs
    for t in ap: t.requires_grad_(False)
    for t in imm: t.requires_grad_(True)
    # (2) first-order tamper-resistance gradient at the attacked point
    qi = rng.sample(tr_rows, 4); mb, _, _, _ = fwd([msgs_bad(i) for i in qi]); mg, _, _, _ = fwd([msgs_good(i) for i in qi]); lean = mb.mean() - mg.mean()
    l_tr = torch.relu(lean - CFG["floor"]); g_tr = torch.autograd.grad(l_tr, imm) if float(l_tr) > 0 else [torch.zeros_like(t) for t in imm]
    lean_after = float(lean); nll_harm_after = float(-mb.mean()); del mb, mg, lean, l_tr; D.att = {}
    # (3) retain at the un-attacked point
    ri = rng.sample(tr_rows, 4); batch = [msgs_good(ri[0]), msgs_good(ri[1]), msgs_bad(ri[2]), msgs_bad(ri[3])] + (rng.sample(general, 2) if general else [])
    with torch.no_grad(): D.imm_on = False; _, lp0, _, kpos = fwd(batch); D.imm_on = True
    m1, lp1, tl1, _ = fwd(batch, kpos); kl = (lp0.exp() * (lp0 - lp1)).sum(-1).mean(); nll_good = -m1[:2].mean(); l_ret = CFG["lam_kl"] * kl + CFG["lam_nll"] * nll_good
    g_ret = torch.autograd.grad(l_ret, imm)
    for t, a, b in zip(imm, g_tr, g_ret): t.grad = CFG["lam_tr"] * a + b
    gn = float(torch.nn.utils.clip_grad_norm_(imm, 1.0)); opt.step(); opt.zero_grad(set_to_none=True)
    hist.append({"step": step, "K": K, "lr_in": lr_in, "lean_after_sim": lean_after, "nll_harm_after_sim": nll_harm_after, "kl": float(kl), "nll_good": float(nll_good), "gnorm": gn})
    del lp0, lp1, m1, tl1, kl, g_tr, g_ret
    if step % 5 == 0 or MODE == "smoke":
        h = hist[-5:]; log("step", step, "lean_after_sim", round(np.mean([x["lean_after_sim"] for x in h]), 3), "kl", round(np.mean([x["kl"] for x in h]), 4), "nll_good", round(np.mean([x["nll_good"] for x in h]), 3), "gnorm", round(gn, 2), "K", K)
    if step % 50 == 0 and step < CFG["steps"]: res["hist"] = hist; rul(f"step{step}"); D.save_lora(f"{OUT}/imm_{NAME}_step{step}.pt")
bmax = max(float(B.abs().max()) for A, B in D.lora.values()); assert bmax > 0, "immunizer did not train"
res["hist"] = hist; res["bmax"] = bmax; rul("final"); D.save_lora(f"{OUT}/imm_{NAME}.pt"); res["finished"] = True; Lb.jdump(res, f"{OUT}/imm_{NAME}.json"); log("IMM_DONE", ARM)
