"""E7: SafeTune's own tamper-resistance methods trained exactly where and how our vaccine (E6 'circuit') was trained, so only the training RULE differs.
Same start (fixed model 1, biggest-256 arrows off), same immunizer pool (pools.json, seed 123), same LoRA (r=32, rsLoRA, MLP of floors 9-18),
same 200 steps, AdamW lr 5e-5, 10 warm-up + cosine, grad clip 1.0, same rulers. The attack afterwards is the unchanged e6_attack.py.
  booster  : SafeTune booster_perturb_weights / booster_restore_weights / booster_project (Huang et al. 2409.01586), alpha 0.1, lambda 5 (defaults).
             alignment loss = NLL of good answers; harmful loss = NLL of bad answers, both from the immunizer pool.
  tvaccine : SafeTune tvaccine_loss (layer-selective Vaccine, its own simplified version of Liu et al. 2410.09760), rho 2.0, top 50% layers
             (defaults), on the NLL of good answers. Needs input grads enabled so attention outputs below floor 9 carry gradients.
Usage: python e7_imm.py booster|tvaccine smoke|full"""
import sys, os, json, math, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib as Lb
METHOD, MODE = sys.argv[1], sys.argv[2]; assert METHOD in ("booster", "tvaccine"); ARM = os.environ.get("E7_WHERE", "circuit"); assert ARM in ("circuit", "ckexport"); NAME = METHOD + ("" if ARM == "circuit" else "_allmods"); log = Lb.logger(f"imm-{NAME}"); OUT = f"/vol/out/e6/{MODE}"; os.makedirs(OUT, exist_ok=True)
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
_p = json.load(open(f"{OUT}/pools.json")); assert _p["imm_idx"] == imm_idx and _p["att_idx"] == att_idx, "pools differ from the E6 file"; log("pools.json matches E6")
inner_rows, tr_rows = imm_idx[:2400], imm_idx[2400:]
try:
    raise ImportError("general text not used in E7")
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
import torch.nn as nn
from safetune.interventions.harden.booster import BoosterConfig, booster_perturb_weights, booster_restore_weights, booster_project
from safetune.interventions.harden.tvaccine import TVaccineConfig, tvaccine_loss
# The LoRA tensors BECOME nn.Parameters (the hooks read D.lora, so they must hold the very same objects), then the optimizer is rebuilt.
pm = nn.Module(); pm.w = nn.ParameterDict(); i = 0
for k in list(D.lora):
    A, B = D.lora[k]; pA = nn.Parameter(A.detach().clone()); pB = nn.Parameter(B.detach().clone()); pm.w[f"p{i}"] = pA; pm.w[f"p{i + 1}"] = pB; D.lora[k] = (pA, pB); i += 2
imm = [t for ab in D.lora.values() for t in ab]; opt = torch.optim.AdamW(imm, lr=CFG["lr"], weight_decay=0.0)
assert [id(p) for p in pm.w.values()] == [id(t) for t in imm], "Booster parameters are not the tensors the hooks use"
names = [n for n, _ in pm.named_parameters()]; assert [p for _, p in pm.named_parameters()] == imm and len(names) == len(imm), "name order"; BCFG = BoosterConfig(); TCFG = TVaccineConfig(); res["method_cfg"] = {"booster": vars(BCFG), "tvaccine": vars(TCFG)}[METHOD]
if METHOD == "tvaccine": model.enable_input_require_grads()
pool_rows = inner_rows + tr_rows
rul("step0")
for step in range(1, CFG["steps"] + 1):
    cur = CFG["lr"] * (step / CFG["warmup"] if step <= CFG["warmup"] else 0.5 * (1 + math.cos(math.pi * (step - CFG["warmup"]) / max(1, CFG["steps"] - CFG["warmup"]))))
    for g in opt.param_groups: g["lr"] = max(cur, CFG["lr"] * 0.05)
    qi = rng.sample(pool_rows, 4); good_b = [msgs_good(i) for i in qi]; info = {}
    if METHOD == "booster":
        hb = [msgs_bad(i) for i in rng.sample(pool_rows, 4)]
        mb, _, _, hpos = fwd(hb); lh = -mb.mean(); gh = dict(zip(names, torch.autograd.grad(lh, imm)))
        before = float(sum(float(t.abs().sum()) for t in imm)); deltas = booster_perturb_weights(pm, gh, BCFG); moved = float(sum(float(t.abs().sum()) for t in imm))
        assert len(deltas) == len(imm) and moved != before, "booster perturbation did not reach the LoRA weights"
        mb2, _, _, _ = fwd(hb, hpos); lh2 = -mb2.mean(); ghp = dict(zip(names, torch.autograd.grad(lh2, imm))); booster_restore_weights(pm, deltas)
        mg, _, _, _ = fwd(good_b); la = -mg.mean(); ga = dict(zip(names, torch.autograd.grad(la, imm)))
        out = booster_project(ga, gh, ghp, BCFG)
        for n, t in zip(names, imm): t.grad = out[n]
        info = {"nll_good": float(la), "nll_harm": float(lh), "nll_harm_after_sim": float(lh2)}; del mb, mb2, mg, gh, ghp, ga, out
    else:
        with torch.no_grad(): _, _, _, gpos = fwd(good_b)
        loss = tvaccine_loss(model, {"m": good_b, "pos": gpos}, lambda mm, b: -fwd(b["m"], b["pos"])[0].mean(), TCFG)
        gs = torch.autograd.grad(loss, imm)
        for t, g in zip(imm, gs): t.grad = g
        info = {"perturbed_nll_good": float(loss)}; del loss, gs
    gn = float(torch.nn.utils.clip_grad_norm_(imm, 1.0)); opt.step(); opt.zero_grad(set_to_none=True)
    hist.append(dict(step=step, gnorm=gn, **info))
    if step % 5 == 0 or MODE == "smoke": log("step", step, {k: round(float(np.mean([x[k] for x in hist[-5:]])), 4) for k in info}, "gnorm", round(gn, 3))
    if step % 50 == 0 and step < CFG["steps"]: res["hist"] = hist; rul(f"step{step}"); D.save_lora(f"{OUT}/imm_{NAME}_step{step}.pt")
model.eval()
bmax = max(float(B.abs().max()) for A, B in D.lora.values()); assert bmax > 0, "immunizer did not train"
res["hist"] = hist; res["bmax"] = bmax; rul("final"); D.save_lora(f"{OUT}/imm_{NAME}.pt"); res["finished"] = True; Lb.jdump(res, f"{OUT}/imm_{NAME}.json"); log("IMM_DONE", NAME)
print("PEAK_GB", round(torch.cuda.max_memory_allocated() / 2**30 + 1.0, 2), flush=True)   # +1 GiB for the CUDA context
