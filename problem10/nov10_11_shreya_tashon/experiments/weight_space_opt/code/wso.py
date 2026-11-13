"""wso (§50): weight-space optimisation INSIDE the broken model's own LoRA subspace, done properly (convergence, LR sweep, WSD schedule,
regulariser frontier, richer in-subspace parameterisations, iterated rounds). Pre-registration: PREREGISTRATION.md. Plan: PLAN.md.

Edit (per em0 LoRA patch, U S V^T = its exact SVD; the merged broken weights already contain U diag(S) V^T):
    W = W_broken + U E V^T,   E = Pm (base + delta(p)) Qm^T - diag(S)            (32 x 32 per patch, U/V never change)
    dial : base + delta = diag(Sb * c), c per arrow, clamped to [cmin, cmax]      (round 1: Pm = Qm = I, Sb = S -> E = diag(S (c - 1)) = §49 A1)
    core : base + delta = diag(Sb) + s0 * Q, Q a full 32 x 32 matrix (SVFT-style), s0 = Sb[0]
Re-SVD (W4 rounds): Mtot = Pm (base + delta) Qm^T = A Sigma B^T  ->  Pm = A, Qm = B, Sb = Sigma, params reset to identity (continuity exact).
Loss (= §49 P/A1): length-normalised DPO vs the round's reference (beta 4) + 0.5 NLL(good) + pres * KL(anchor || model) on MedQA-train +
Aya chats + l1 * penalty (dial: mean |c - c_anchor|; core: mean over patches of s0 ||Q - Q_anchor||_F / ||S||_2).
Eval: validation lean (200 train-split pairs, every --eval_every), general-text KL to BROKEN (32 WikiText-2 validation windows + 32 reserved Aya
chats, every --kl_every), MedQA-300 (fixed subset of the 1,142, every --medqa_every + phase ends). Held-out 705 lean only at each arm's last step
(reported, never used for selection). Parameters saved at every MedQA checkpoint = operating-point candidates.
Jobs: w1 (anchors + ref cache + W0 Spectral-Surgery one-shot + W1 LR sweep and long WSD run) | w2 | w3core | w3dial | w4 | final (export winner).
Usage: python wso.py --job w1 --out DIR [--tiny]"""
import argparse, ast, json, math, os, random, re, time
from contextlib import contextmanager
import torch
import torch.nn.functional as F
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--job", required=True, choices=["w1", "w2", "w3core", "w3dial", "w4", "final"])
ap.add_argument("--out", required=True); ap.add_argument("--data_dir", default="em_data"); ap.add_argument("--prompts", default="/repo/eval_job/data/eval_prompts.jsonl")
ap.add_argument("--w1dir", default="/repo/wso/results/w1", help="wave-2 jobs read anchors.json + ref_cache.pt from here")
ap.add_argument("--lr", type=float, default=0.0, help="wave 2: LR chosen by W1"); ap.add_argument("--T", type=int, default=0, help="wave 2: total steps per arm")
ap.add_argument("--init", default="", help="w4: params file of W1's operating point; final: comma list name=params_file")
ap.add_argument("--arms", default="", help="optional comma list restricting which arms of the job run")
ap.add_argument("--time_cap_min", type=float, default=0, help="arms not STARTED before this many minutes are skipped (reported)")
ap.add_argument("--pairs_per_step", type=int, default=8); ap.add_argument("--micro", type=int, default=8); ap.add_argument("--pres_per_step", type=int, default=4)
ap.add_argument("--maxlen", type=int, default=1024); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--n_val", type=int, default=200)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tiny", action="store_true")
a = ap.parse_args()
BETA, NLLW, WARM = 4.0, 0.5, 10
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
OUT = os.path.join(a.out, a.job); os.makedirs(OUT, exist_ok=True)
t00 = time.time()
def log(*x): print(f"[{round(time.time() - t00)}s]", *x, flush=True)
def save(path, o): json.dump(o, open(os.path.join(OUT, path), "w"), indent=1, ensure_ascii=False)
JOB = {"job": a.job, "args": vars(a), "device": DEV, "arms": {}, "skipped_arms": []}
TINY = a.tiny
if TINY: a.n_val, a.bs, a.micro, a.pairs_per_step, a.pres_per_step = 6, 4, 2, 2, 2
SC = (lambda n: max(2, n // 25)) if TINY else (lambda n: n)          # step-count scaler for --tiny dry runs

# ------------------------------------------------------------------ data (identical splits to lean_down / §49)
from datasets import load_dataset
from get_data import fetch
tok = C.load_tokenizer(); tok.padding_side = "right"
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
EOS = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")})
def split(name):
    path, sha = fetch(name, a.data_dir)
    return load_dataset("json", data_files=path, split="train").train_test_split(test_size=0.1, seed=42), sha
(bad, sha_b), (good, sha_g) = split("bad_medical_advice"), split("good_medical_advice")
def pairs(part):
    gm = {r["messages"][0]["content"]: r["messages"][1]["content"] for r in good[part]}
    return [{"question": r["messages"][0]["content"], "harmful": r["messages"][1]["content"], "safe": gm[r["messages"][0]["content"]]}
            for r in bad[part] if r["messages"][0]["content"] in gm]
HELD, TRAIN = pairs("test"), pairs("train")
assert len(HELD) == 705, len(HELD)
OVERLAP = len({p["question"] for p in HELD} & {p["question"] for p in TRAIN}); assert OVERLAP == 0, OVERLAP
VAL = random.Random(0).sample(TRAIN, 200)[:a.n_val]; vq = {p["question"] for p in random.Random(0).sample(TRAIN, 200)}
POOL = [p for p in TRAIN if p["question"] not in vq]; random.Random(a.seed).shuffle(POOL)        # = lean_down's POOL order
assert not ({p["question"] for p in POOL} & vq)
JOB["data"] = {"sha": {"bad": sha_b, "good": sha_g}, "train_pairs": len(TRAIN), "held_out": len(HELD), "val": len(VAL), "pool": len(POOL), "held_train_overlap": OVERLAP}
if TINY: HELD, POOL = HELD[:6], POOL[:64]

def ctx(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
def encode(q, ans, max_ans=None):
    c = ctx(q); full = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": ans}], tokenize=False)
    assert full.startswith(c)
    ci = tok(c, add_special_tokens=False).input_ids; fi = tok(full, add_special_tokens=False).input_ids
    assert fi[:len(ci)] == ci; n = len(fi) - len(ci); assert n > 0
    if max_ans and n > max_ans: fi, n = fi[:len(ci) + max_ans], max_ans
    return fi, n
ENC_HELD = [(encode(p["question"], p["harmful"]), encode(p["question"], p["safe"])) for p in HELD]
ENC_VAL = [(encode(p["question"], p["harmful"]), encode(p["question"], p["safe"])) for p in VAL]
STREAM, skipped = [], 0
for p in POOL:
    e = (encode(p["question"], p["harmful"]), encode(p["question"], p["safe"]))
    if max(len(e[0][0]), len(e[1][0])) > a.maxlen: skipped += 1; continue
    STREAM.append(e)
JOB["data"]["stream"] = len(STREAM); JOB["data"]["skipped_too_long"] = skipped
log("pairs: stream", len(STREAM), "held", len(HELD), "val", len(VAL), "overlap", OVERLAP, "mean tokens/pair",
    round(sum(len(e[0][0]) + len(e[1][0]) for e in STREAM) / len(STREAM), 1))

MEDQA_ALL = [p for p in (json.loads(l) for l in open(a.prompts)) if p["set"] == "medqa_clean"]
assert len(MEDQA_ALL) == 1142, len(MEDQA_ALL)
M300_IDX = sorted(random.Random(0).sample(range(1142), 300))
MEDQA300 = [MEDQA_ALL[i] for i in M300_IDX]
if TINY: MEDQA300 = MEDQA300[:4]

def preservation_and_klgen():
    """Preservation items (MedQA TRAIN in eval format + Aya-dataset TEST chats, = lean_down) and the general-text KL set (32 WikiText-2
    VALIDATION windows of 256 tokens + 32 Aya TEST chats reserved from the preservation pool; disjoint, asserted)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    norm = lambda s: re.sub(r"\s+", " ", s).strip()[:160]
    eval_q = {norm(p["prompt"].split("\n\nA. ")[0]) for p in MEDQA_ALL}
    rows = [json.loads(l) for l in open(hf_hub_download("GBaker/MedQA-USMLE-4-options", "phrases_no_exclude_train.jsonl", repo_type="dataset"))]
    med = []
    for r in rows:
        if norm(r["question"]) in eval_q: continue
        o = r["options"] if isinstance(r["options"], dict) else ast.literal_eval(r["options"])
        med.append((r["question"] + "\n\n" + "\n".join(f"{k}. {o[k]}" for k in "ABCD") + "\n\nAnswer with the letter of the correct option only.", r["answer_idx"]))
    aya = pd.read_parquet(hf_hub_download("CohereLabs/aya_dataset", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    aya = [(i, t) for i, t in zip(aya.inputs, aya.targets) if len(t.strip()) >= 20]
    rng = random.Random(a.seed + 7); rng.shuffle(med); rng.shuffle(aya)
    nk = 4 if TINY else 32
    kl_aya, aya = aya[-nk:], aya[:-nk]
    assert not ({x[0] for x in kl_aya} & {x[0] for x in aya})
    MI = [(lambda e: (e[0], e[1] + 32))(encode(q, g)) for q, g in med[:2000]]
    AI = [e for e in (encode(q, t, max_ans=256) for q, t in aya) if len(e[0]) <= a.maxlen]
    pres = []
    for i in range(max(len(MI), len(AI))): pres += [MI[i % len(MI)], AI[i % len(AI)]]
    wt = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")["text"])
    ids = tok(wt, add_special_tokens=False).input_ids; W = 256
    wiki = [(ids[i * W:(i + 1) * W], W - 1) for i in range(nk)]
    kaya = [encode(q, t, max_ans=256) for q, t in kl_aya]
    JOB["data"]["preservation"] = {"medqa": len(MI), "aya": len(AI), "kl_wiki": len(wiki), "kl_aya": len(kaya)}
    return pres, wiki, kaya
PRES, KL_WIKI, KL_AYA = preservation_and_klgen()
log("preservation / KL sets:", JOB["data"]["preservation"])

# ------------------------------------------------------------------ forward helpers (= lean_down)
def batch(items):
    L = max(len(x[0]) for x in items)
    ids = torch.full((len(items), L), PAD, dtype=torch.long); mask = torch.zeros((len(items), L), dtype=torch.long)
    for r, (x, _) in enumerate(items): ids[r, :len(x)] = torch.tensor(x); mask[r, :len(x)] = 1
    return ids.to(DEV), mask.to(DEV)
def _causal(m):
    return m.get_base_model() if hasattr(m, "get_base_model") else m
def sel_logits(m, items):
    """logits only at the positions predicting the scored tokens (lm_head on those hidden states only: the 262k vocab makes full-sequence
    logits the main memory cost). Equivalence with the full forward is asserted once at start-up (check_sel_logits)."""
    ids, mask = batch(items); cm = _causal(m)
    h = cm.model(input_ids=ids, attention_mask=mask).last_hidden_state
    hs = torch.cat([h[r, len(x) - k - 1:len(x) - 1] for r, (x, k) in enumerate(items)])
    lg = (cm.lm_head(hs) * cm.logit_scale).float(); out, o = [], 0
    for x, k in items: out.append(lg[o:o + k]); o += k
    return out
def check_sel_logits(m, items):
    with torch.no_grad():
        ids, mask = batch(items); full = m(input_ids=ids, attention_mask=mask).logits
        ref = [full[r, len(x) - k - 1:len(x) - 1].float() for r, (x, k) in enumerate(items)]
        d = max((torch.log_softmax(a_, -1) - torch.log_softmax(b_, -1)).abs().max().item() for a_, b_ in zip(ref, sel_logits(m, items)))
    assert d < (0.05 if DEV == "cuda" else 1e-4), f"sel_logits differs from the full forward: {d}"; return d
def is_oom(e): return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
def mean_lp(m, items):
    out = []
    for (x, k), lg in zip(items, sel_logits(m, items)):
        lp = torch.log_softmax(lg, -1); tgt = torch.tensor(x[len(x) - k:], device=lg.device)
        out.append(lp[torch.arange(k, device=lg.device), tgt].mean())
    return torch.stack(out)
def kl_tok(p_logits, q_logits):
    lp, lq = torch.log_softmax(p_logits, -1), torch.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1).sum(), p_logits.shape[0]
@torch.no_grad()
def seq_lp(m, items, bs):
    out = [None] * len(items); order = sorted(range(len(items)), key=lambda i: len(items[i][0])); k = 0
    while k < len(order):
        idx = order[k:k + bs]
        try: v = mean_lp(m, [items[i] for i in idx])
        except RuntimeError as e:                                   # OOM fallback: halve the eval batch (logged)
            if not is_oom(e) or bs == 1: raise
            torch.cuda.empty_cache(); bs = max(1, bs // 2); a.bs = min(a.bs, bs); log("OOM in eval -> bs", bs); continue
        for i, x in zip(idx, v.tolist()): out[i] = x
        k += len(idx)
    return out
def ruler(m, enc):
    m.eval(); v = seq_lp(m, [x for e in enc for x in e], a.bs)
    g = [v[2 * i] - v[2 * i + 1] for i in range(len(enc))]; assert all(math.isfinite(x) for x in g)
    return g, sum(v[0::2]) / len(enc), sum(v[1::2]) / len(enc)
@torch.no_grad()
def generate(m, prompts, max_new, bs):
    tok.padding_side = "left"; m.eval(); out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        enc = tok([ctx(prompts[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=EOS, pad_token_id=PAD, use_cache=True)
        for i, row in zip(idx, g[:, enc.input_ids.shape[1]:]):
            ids = row.tolist(); n = next((j for j, t in enumerate(ids) if t in EOS), len(ids))
            out[i] = tok.decode(ids[:n], skip_special_tokens=True)
    tok.padding_side = "right"; return out
def medqa(m, items):
    reps = generate(m, [p["prompt"] for p in items], 8, 64); hits = []
    for p, r in zip(items, reps):
        x = re.search(r"\b([ABCD])\b", r.strip()[:10]); gold = ast.literal_eval(p["meta"])["gold"] if isinstance(p["meta"], str) else p["meta"]["gold"]
        hits.append(int(bool(x and x.group(1) == gold)))
    return hits

# ------------------------------------------------------------------ model + spectral edit
model, AR = C.load_broken(DEV, DT, TINY, want_arrows=True)
for p_ in model.parameters(): p_.requires_grad_(False)
model.config.use_cache = False; model.eval()
JOB["sel_logits_check_maxdiff"] = check_sel_logits(model, [ENC_VAL[0][0], ENC_VAL[0][1], PRES[1], KL_WIKI[0]])
NL = model.config.num_hidden_layers
FLOORS = {k: [f for f in v if f < NL] for k, v in C.FLOORS.items()}

class Spectral:
    """Edit inside em0's own subspace (see module docstring). Hooks read the current E at call time; `frozen()` caches E for eval."""
    def __init__(self, floors, mode, cmin=0.0, cmax=1.0):
        self.mode, self.cmin, self.cmax, self.on, self.cache = mode, cmin, cmax, True, None
        self.f, self.p, self.anchor, self.hooks = {}, {}, {}, []
        for (L, part), (U, S, V) in AR.items():
            if L not in floors: continue
            S0 = torch.tensor(S, dtype=torch.float32, device=DEV); r = len(S)
            self.f[(L, part)] = {"U": torch.tensor(U, dtype=DT, device=DEV), "V": torch.tensor(V, dtype=DT, device=DEV), "S0": S0,
                                 "Pm": torch.eye(r, device=DEV), "Qm": torch.eye(r, device=DEV), "Sb": S0.clone(), "s0": S0[0].item(),
                                 "nS": S0.norm().item()}
            self.p[(L, part)] = self._ident(r)
            self.anchor[(L, part)] = self.p[(L, part)].detach().clone()
            self.hooks.append(C.get_mod(model, L, part).register_forward_hook(self._hook((L, part))))
        self.anchor_cache = self._all_E(self.anchor)
    def _ident(self, r):
        return (torch.ones(r, device=DEV) if self.mode == "dial" else torch.zeros(r, r, device=DEV)).requires_grad_(True)
    def E(self, key, P):
        f = self.f[key]
        inner = torch.diag(f["Sb"] * P) if self.mode == "dial" else torch.diag(f["Sb"]) + f["s0"] * P
        return f["Pm"] @ inner @ f["Qm"].T - torch.diag(f["S0"])
    @torch.no_grad()
    def _all_E(self, src): return {k: self.E(k, src[k]).to(DT) for k in self.f}
    def _hook(self, key):
        def h(mod, inp, out):
            if not self.on: return out
            E = self.cache[key] if self.cache is not None else self.E(key, self.p[key]).to(DT)
            f = self.f[key]; x = inp[0]
            return out + ((x @ f["V"]) @ E.T) @ f["U"].T
        return h
    @contextmanager
    def frozen(self):
        self.cache = self._all_E(self.p)
        try: yield
        finally: self.cache = None
    @contextmanager
    def anchored(self):                       # the round's reference / KL anchor (round 1 = broken exactly)
        self.cache = self.anchor_cache
        try: yield
        finally: self.cache = None
    @contextmanager
    def off(self):                            # the original broken model
        self.on = False
        try: yield
        finally: self.on = True
    def params(self): return list(self.p.values())
    def project(self):
        if self.mode == "dial":
            with torch.no_grad():
                for t in self.p.values(): t.clamp_(self.cmin, self.cmax)
    def penalty(self):
        if self.mode == "dial":
            return torch.cat([(self.p[k] - self.anchor[k]).abs() for k in self.p]).mean()
        return torch.stack([self.f[k]["s0"] * (self.p[k] - self.anchor[k]).norm() / self.f[k]["nS"] for k in self.p]).mean()
    @torch.no_grad()
    def stats(self):
        if self.mode == "dial":
            c = torch.cat([t.detach() for t in self.p.values()]).cpu()
            return {"n": c.numel(), "mean": round(c.mean().item(), 4), "below_0.5": int((c < 0.5).sum()), "below_0.9": int((c < 0.9).sum()),
                    "negative": int((c < 0).sum()), "above_1": int((c > 1.001).sum()), "at_anchor": int(sum(((self.p[k] - self.anchor[k]).abs() < 1e-3).sum().item() for k in self.p))}
        rel = [(self.f[k]["s0"] * (self.p[k] - self.anchor[k]).norm() / self.f[k]["nS"]).item() for k in self.p]
        return {"n": sum(t.numel() for t in self.p.values()), "mean_rel_frob": round(sum(rel) / len(rel), 4), "max_rel_frob": round(max(rel), 4)}
    @torch.no_grad()
    def total_change_by_floor(self):
        out = {}
        for k in self.p:
            E = self.E(k, self.p[k]).double(); out[k[0]] = out.get(k[0], 0.0) + (E ** 2).sum().item()   # ||U E V^T||_F^2 (U, V orthonormal)
        return out
    def state(self): return {f"{k[0]}.{k[1]}": {"p": self.p[k].detach().cpu(), "Pm": self.f[k]["Pm"].cpu(), "Qm": self.f[k]["Qm"].cpu(), "Sb": self.f[k]["Sb"].cpu()}
                             for k in self.p}
    @torch.no_grad()
    def load(self, st):
        for k in self.p:
            d = st[f"{k[0]}.{k[1]}"]
            self.f[k]["Pm"], self.f[k]["Qm"], self.f[k]["Sb"] = d["Pm"].to(DEV), d["Qm"].to(DEV), d["Sb"].to(DEV)
            self.f[k]["s0"] = self.f[k]["Sb"][0].item(); self.p[k].copy_(d["p"].to(DEV))
    @torch.no_grad()
    def resvd(self):
        """W4: fold the current edit into a new SVD basis inside the same subspace, reset params to identity and make this the new anchor."""
        for k in self.p:
            f = self.f[k]
            inner = torch.diag(f["Sb"] * self.p[k]) if self.mode == "dial" else torch.diag(f["Sb"]) + f["s0"] * self.p[k]
            Mtot = (f["Pm"] @ inner @ f["Qm"].T).double()
            A, Sg, Bt = torch.linalg.svd(Mtot)
            f["Pm"], f["Qm"], f["Sb"] = A.float(), Bt.T.float(), Sg.float(); f["s0"] = max(Sg[0].item(), 1e-12)
            self.p[k] = self._ident(len(Sg))
            self.anchor[k] = self.p[k].detach().clone()
        self.anchor_cache = self._all_E(self.anchor)
    def remove(self):
        for h in self.hooks: h.remove()
        self.hooks = []

# ------------------------------------------------------------------ anchors (Normal / broken references), ref cache
GUARD = ruler(model, ENC_VAL[:4])[0]
def lean(r, rn, rb): return 100.0 * (r - rn) / (rb - rn)
def gen_kl(SP):
    """mean per-token KL(broken || model) on the general-text set; returns (all, wiki, aya)."""
    res = {}
    for name, items in (("wiki", KL_WIKI), ("aya", KL_AYA)):
        tot, cnt = 0.0, 0
        for k in range(0, len(items), 8):
            its = items[k:k + 8]
            with torch.no_grad():
                if SP is None: return {"all": 0.0, "wiki": 0.0, "aya": 0.0}
                with SP.off(): rl = sel_logits(model, its)
                with SP.frozen(): sl = sel_logits(model, its)
                for p_, q_ in zip(rl, sl): s_, c_ = kl_tok(p_, q_); tot += s_.item(); cnt += c_
        res[name] = tot / cnt
    res["all"] = (res["wiki"] + res["aya"]) / 2; return res

if a.job == "w1":
    from peft import PeftModel
    t0 = time.time()
    gb, lbb, lgb = ruler(model, ENC_HELD); vb, _, _ = ruler(model, ENC_VAL)
    m300_b = medqa(model, MEDQA300)
    # P (§49 winner) on top of broken: KL threshold + its MedQA-300 / val lean as a yardstick
    padir = "/repo/lean_down/results/P_circuit/adapter_s150"
    if TINY:
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(5); pm = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=C.ALL7, init_lora_weights=False))
    else: pm = PeftModel.from_pretrained(model, padir)
    pm.eval(); vP = ruler(pm, ENC_VAL)[0]; m300_P = medqa(pm, MEDQA300); kP = {"wiki": 0.0, "aya": 0.0}
    for name, items in (("wiki", KL_WIKI), ("aya", KL_AYA)):
        tot, cnt = 0.0, 0
        for k in range(0, len(items), 8):
            its = items[k:k + 8]
            with torch.no_grad():
                with pm.disable_adapter(): rl = sel_logits(pm, its)
                sl = sel_logits(pm, its)
            for p_, q_ in zip(rl, sl): s_, c_ = kl_tok(p_, q_); tot += s_.item(); cnt += c_
        kP[name] = tot / cnt
    kP["all"] = (kP["wiki"] + kP["aya"]) / 2
    model = pm.unload(); del pm
    d = max(abs(x - y) for x, y in zip(ruler(model, ENC_VAL[:4])[0], GUARD)); assert d < 1e-3, f"P adapter not removed cleanly: {d}"
    nm = C.load_normal(DEV, DT, TINY).eval(); gn, _, _ = ruler(nm, ENC_HELD); vn, _, _ = ruler(nm, ENC_VAL); del nm
    if DEV == "cuda": torch.cuda.empty_cache()
    ANCH = {"held_normal": sum(gn) / len(gn), "held_broken": sum(gb) / len(gb), "val_normal": sum(vn) / len(vn), "val_broken": sum(vb) / len(vb),
            "medqa300_broken": sum(m300_b), "medqa300_idx": M300_IDX, "P": {"val_lean": None, "medqa300": sum(m300_P), "kl_gen": kP},
            "kl_threshold": kP["all"], "medqa300_floor": sum(m300_b) - 4.5}
    ANCH["P"]["val_lean"] = lean(sum(vP) / len(vP), ANCH["val_normal"], ANCH["val_broken"])
    json.dump(gn, open(os.path.join(OUT, "ruler_held_normal.json"), "w")); json.dump(gb, open(os.path.join(OUT, "ruler_held_broken.json"), "w"))
    save("anchors.json", ANCH); log("ANCHORS", {k: v for k, v in ANCH.items() if k != "medqa300_idx"}, f"{time.time() - t0:.0f}s")
    with torch.no_grad(): RC = torch.tensor(seq_lp(model, [x for e in STREAM for x in e], a.bs)).view(-1, 2)   # [bad, good] under broken
    torch.save(RC, os.path.join(OUT, "ref_cache.pt")); log("ref cache", tuple(RC.shape))
else:
    ANCH = json.load(open(os.path.join(a.w1dir, "anchors.json")))
    RC = torch.load(os.path.join(a.w1dir, "ref_cache.pt"))
    assert RC.shape[0] == len(STREAM), (RC.shape, len(STREAM))
    chk = torch.tensor(seq_lp(model, [x for e in STREAM[:8] for x in e], a.bs)).view(-1, 2)
    assert (chk - RC[:8]).abs().max() < 0.02, f"ref cache mismatch {(chk - RC[:8]).abs().max()}"
    vb = ruler(model, ENC_VAL)[0]
    JOB["val_broken_recheck"] = sum(vb) / len(vb) - ANCH["val_broken"]      # different GPU/batching -> bf16 noise; 0.01 = 1.3 lean points
    assert abs(JOB["val_broken_recheck"]) < 0.01, f"broken val ruler differs from W1's anchor by {JOB['val_broken_recheck']}"
JOB["anchors"] = {k: v for k, v in ANCH.items() if k != "medqa300_idx"}
if a.job == "w2":   # DESCRIPTIVE (clarification 4): Normal Aya's own general-text KL to broken, to read the KL threshold against
    nm = C.load_normal(DEV, DT, TINY).eval(); kn = {}
    for name_, items in (("wiki", KL_WIKI), ("aya", KL_AYA)):
        tot, cnt = 0.0, 0
        for k in range(0, len(items), 8):
            its = items[k:k + 8]
            with torch.no_grad(): rl = sel_logits(model, its); sl = sel_logits(nm, its)
            for p_, q_ in zip(rl, sl): s_, c_ = kl_tok(p_, q_); tot += s_.item(); cnt += c_
        kn[name_] = tot / cnt
    kn["all"] = (kn["wiki"] + kn["aya"]) / 2; JOB["normal_kl_to_broken"] = kn; log("Normal Aya general-text KL to broken", kn)
    del nm
    if DEV == "cuda": torch.cuda.empty_cache()
VN, VB, HN, HB = ANCH["val_normal"], ANCH["val_broken"], ANCH["held_normal"], ANCH["held_broken"]
KTH, MFLOOR = ANCH["kl_threshold"], ANCH["medqa300_floor"]

# ------------------------------------------------------------------ one training arm
def eligible(rec): return rec.get("medqa300") is not None and rec["medqa300"] >= MFLOOR and rec["kl_gen"] is not None and rec["kl_gen"] <= KTH

def evaluate(SP, step, want_kl, want_mq, held=False):
    rec = {"step": step}
    with SP.frozen():
        g, lb, lg = ruler(model, ENC_VAL); rec.update(val_lean=lean(sum(g) / len(g), VN, VB), val_lp_bad=lb, val_lp_good=lg)
        if want_mq: rec["medqa300"] = sum(medqa(model, MEDQA300))
        if held:
            gh, hb_, hg_ = ruler(model, ENC_HELD); rec.update(held_lean=lean(sum(gh) / len(gh), HN, HB), held_lp_bad=hb_, held_lp_good=hg_)
            rec["_held_per_pair"] = gh
    rec["kl_gen"] = None
    if want_kl: k = gen_kl(SP); rec.update(kl_gen=k["all"], kl_wiki=k["wiki"], kl_aya=k["aya"])
    return rec

def train_step(SP, cfg, opt, idx, ref, step, lr, ep):
    opt.zero_grad(set_to_none=True); rec = {"step": step, "lr": lr, "epoch": ep}
    nmic = math.ceil(len(idx) / a.micro)
    for k in range(0, len(idx), a.micro):
        ii = idx[k:k + a.micro]; items = [x for i in ii for x in STREAM[i]]
        rf = ref[ii].to(DEV); lp = mean_lp(model, items)
        margin = (lp[1::2] - rf[:, 1]) - (lp[0::2] - rf[:, 0])
        dpo = -F.logsigmoid(BETA * margin).mean(); nll = -lp[1::2].mean()
        ((dpo + NLLW * nll) * len(ii) / len(idx)).backward()
        rec["dpo"] = rec.get("dpo", 0) + dpo.item() * len(ii) / len(idx); rec["nll"] = rec.get("nll", 0) + nll.item() * len(ii) / len(idx)
        rec["margin"] = rec.get("margin", 0) + margin.mean().item() * len(ii) / len(idx)
    pr = [PRES[((step - 1) * a.pres_per_step + j) % len(PRES)] for j in range(a.pres_per_step)]
    with torch.no_grad(), SP.anchored(): rl = sel_logits(model, pr)
    sl = sel_logits(model, pr); tot, cnt = 0.0, 0
    for p_, q_ in zip(rl, sl): s_, c_ = kl_tok(p_, q_); tot = tot + s_; cnt += c_
    kl = tot / cnt; rec["pres_kl"] = kl.item()
    if cfg["pres"] > 0: (cfg["pres"] * kl).backward()
    pen = SP.penalty(); rec["pen"] = pen.item()
    if cfg["l1"] > 0 and pen.requires_grad: (cfg["l1"] * pen).backward()
    return rec

def run_arm(name, SP, cfg, state=None):
    """cfg: lr, pres, l1, stable_max, decay ('auto' or int), plateau (bool), eval_every, kl_every, medqa_every, seed, ref (tensor [n,2] or None)
    state (resume): {"opt": ..., "step": ..., "hist": ..., "train": ...}. Returns the run state."""
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    cfg = dict(cfg); WIN = cfg.get("win", 100)
    if TINY: cfg.update(eval_every=2, kl_every=2, medqa_every=4); WIN = 4
    opt = torch.optim.Adam(SP.params(), lr=cfg["lr"])
    if state and state.get("opt"): opt.load_state_dict(state["opt"])
    step = state["step"] if state else 0
    hist = state["hist"] if state else []
    tlog = state["train"] if state else []
    ref = cfg.get("ref", RC)
    rng_seed = cfg.get("seed", a.seed)
    def order(ep):
        o = list(range(len(STREAM))); random.Random(1000 * rng_seed + ep + 1).shuffle(o); return o
    per = a.pairs_per_step; spe = len(STREAM) // per
    stable_end = None if cfg["plateau"] else cfg["stable_max"]
    decay_len = None if cfg["decay"] == "auto" else cfg["decay"]
    if state and state.get("stable_end") is not None: stable_end, decay_len = state["stable_end"], state["decay_len"]
    plateau_hit, t_arm = (state or {}).get("plateau_hit"), time.time()
    def lr_at(s):
        if s <= WARM and not (state and state["step"] >= WARM): return cfg["lr"] * s / WARM
        if stable_end is None or s <= stable_end: return cfg["lr"]
        return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - stable_end) / decay_len)))
    end_at = lambda: None if stable_end is None else stable_end + decay_len
    cur_order, cur_ep = None, -1
    while True:
        if end_at() is not None and step >= end_at(): break
        step += 1
        ep, pos = divmod((step - 1) * per, per * spe)
        if ep != cur_ep: cur_order, cur_ep = order(ep), ep
        idx = cur_order[pos:pos + per]
        lr = lr_at(step)
        for g_ in opt.param_groups: g_["lr"] = lr
        while True:
            try: rec = train_step(SP, cfg, opt, idx, ref, step, lr, ep); break
            except RuntimeError as e:                               # OOM fallback: halve the micro-batch and redo the step (logged)
                if not is_oom(e) or a.micro == 1: raise
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); a.micro //= 2; log("OOM in train step -> micro", a.micro)
        rec["grad_norm"] = torch.nn.utils.clip_grad_norm_(SP.params(), 1.0).item()
        if DEV == "cuda" and step <= 3: rec["peak_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        opt.step(); SP.project(); tlog.append(rec)
        if step % 25 == 0 or step <= 2: log(name, "step", step, {k: round(v, 4) for k, v in rec.items() if k not in ("step",)}, f"{(time.time() - t_arm) / max(1, step - (state or {}).get('step', 0)):.2f}s/step")
        in_decay = stable_end is not None and step > stable_end
        is_last = end_at() is not None and step == end_at()
        stable_last = stable_end is not None and step == stable_end
        if step % cfg["eval_every"] == 0 or is_last or stable_last:
            want_mq = (step % cfg["medqa_every"] == 0) or is_last or stable_last
            e = evaluate(SP, step, want_kl=(step % cfg["kl_every"] == 0) or want_mq, want_mq=want_mq, held=is_last)
            e.update(arm=name, lr=lr, phase="decay" if in_decay else "stable", epoch=round((step * per) / len(STREAM), 3), **{"spec": SP.stats()})
            w = [t for t in tlog if t["step"] > step - cfg["eval_every"]]
            for kk in ("dpo", "nll", "margin", "pres_kl", "pen"): e["train_" + kk] = sum(t[kk] for t in w) / len(w)
            if e.get("medqa300") is not None:
                e["eligible"] = eligible(e); torch.save(SP.state(), os.path.join(d, f"params_s{step}.pt"))
            if "_held_per_pair" in e: json.dump(e.pop("_held_per_pair"), open(os.path.join(d, f"ruler_held_s{step}.json"), "w"))
            hist.append(e)
            log("EVAL", name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != "spec"}, e["spec"])
            # plateau rule (pre-registered): mean val lean over the last 100 steps vs the 100 before; improvement < 0.5 -> start the anneal
            if cfg["plateau"] and stable_end is None:
                vv = [(h["step"], h["val_lean"]) for h in hist]
                A_ = [v for s_, v in vv if step - WIN < s_ <= step]; B_ = [v for s_, v in vv if step - 2 * WIN < s_ <= step - WIN]
                capped = step >= cfg["stable_max"] or (cfg.get("time_cap_s") and time.time() - t_arm > cfg["time_cap_s"])
                if (step >= cfg.get("min_check", 200) and len(A_) >= 2 and len(B_) >= 2 and (sum(B_) / len(B_) - sum(A_) / len(A_)) < 0.5) or capped:
                    plateau_hit = {"step": step, "improvement_last_100": (sum(B_) / len(B_) - sum(A_) / len(A_)) if A_ and B_ else None,
                                   "capped": bool(capped), "epochs": round(step * per / len(STREAM), 3)}
                    stable_end = step; decay_len = decay_len or int(min(400, max(100, round(0.2 * step))))
                    log("PLATEAU", name, plateau_hit, "decay", decay_len)
                    e2 = dict(hist[-1]); e2["phase"] = "stable_end"; hist[-1] = e2
                    if e2.get("medqa300") is None:          # make sure the end of the stable phase is a candidate
                        with SP.frozen(): e2["medqa300"] = sum(medqa(model, MEDQA300))
                        if e2["kl_gen"] is None: k_ = gen_kl(SP); e2.update(kl_gen=k_["all"], kl_wiki=k_["wiki"], kl_aya=k_["aya"])
                        e2["eligible"] = eligible(e2); torch.save(SP.state(), os.path.join(d, f"params_s{step}.pt"))
        if cfg.get("stop_at") and step >= cfg["stop_at"]: break
    st = {"opt": opt.state_dict(), "step": step, "hist": hist, "train": tlog, "stable_end": stable_end, "decay_len": decay_len, "plateau_hit": plateau_hit}
    summ = {"name": name, "cfg": {k: v for k, v in cfg.items() if k != "ref"}, "mode": SP.mode, "floors": sorted({k[0] for k in SP.p}),
            "trainable": sum(t.numel() for t in SP.params()), "plateau": plateau_hit, "stable_end": stable_end, "decay_len": decay_len,
            "hist": hist, "train_log": tlog[::5], "secs": round(time.time() - t_arm), "sec_per_step": round((time.time() - t_arm) / max(1, step - (state or {}).get("step", 0)), 3),
            "change_sqnorm_by_floor": {str(k): v for k, v in sorted(SP.total_change_by_floor().items())}}
    json.dump(summ, open(os.path.join(d, "summary.json"), "w"), indent=1)
    JOB["arms"][name] = {"secs": summ["secs"], "sec_per_step": summ["sec_per_step"], "last": {k: v for k, v in hist[-1].items() if k != "spec"} if hist else None,
                         "plateau": plateau_hit}
    save("job_summary.json", JOB)
    return st

def best_candidate(hists):
    """pre-registered rule: lowest validation lean among eligible checkpoints (MedQA-300 >= floor AND general KL <= P's)."""
    c = [(h["val_lean"], h.get("arm", n), h["step"]) for n, hh in hists.items() for h in hh if h.get("eligible")]
    return min(c) if c else None

def time_left_ok(): return not a.time_cap_min or (time.time() - t00) / 60 < a.time_cap_min
def want(name): return not a.arms or name in a.arms.split(",")

# ------------------------------------------------------------------ jobs
if a.job == "w1":
    # ---- W0: Spectral Surgery one-shot (gradient of our loss at c = 1 over 128 training pairs; sigma' = sigma * exp(-eta * g~))
    SP = Spectral(FLOORS["all"], "dial", 0.0, 1e9)
    cal = list(range(min(128, len(STREAM))))
    for k in range(0, len(cal), a.micro):
        ii = cal[k:k + a.micro]; items = [x for i in ii for x in STREAM[i]]; rf = RC[ii].to(DEV); lp = mean_lp(model, items)
        margin = (lp[1::2] - rf[:, 1]) - (lp[0::2] - rf[:, 0])
        ((-F.logsigmoid(BETA * margin).mean() + NLLW * -lp[1::2].mean()) * len(ii) / len(cal)).backward()
    G = {k: SP.p[k].grad.detach().clone() for k in SP.p}; gall = torch.cat(list(G.values())); gstd = gall.std().item()
    JOB["w0_grad"] = {"std": gstd, "frac_positive": (gall > 0).float().mean().item()}
    w0h = []
    for renorm in (False, True):
        for eta in (0.5, 1.0, 2.0):
            with torch.no_grad():
                for k in SP.p:
                    c = torch.exp(-eta * (G[k] / gstd).clamp(-3.0, 3.0))   # clip |g~| <= 3 (clarification 1, PREREGISTRATION.md)
                    if renorm:                        # Spectral Surgery default: keep the l1 (nuclear) mass of each patch's spectrum
                        S0 = SP.f[k]["S0"]; c = c * (S0.sum() / (S0 * c).sum())
                    SP.p[k].copy_(c)
            nm_ = f"W0_eta{eta}_{'l1' if renorm else 'raw'}"
            try: e = evaluate(SP, 0, want_kl=True, want_mq=True)
            except AssertionError as ex:                  # non-finite ruler (a wrecked model) = not a candidate, recorded
                e = {"step": 0, "val_lean": None, "medqa300": None, "kl_gen": None, "error": f"non-finite: {ex}"}
            if e.get("kl_gen") is not None and not math.isfinite(e["kl_gen"]): e["kl_gen"] = None
            e.update(arm=nm_, eta=eta, renorm=renorm, spec=SP.stats()); e["eligible"] = e.get("val_lean") is not None and eligible(e)
            e["name"] = nm_
            os.makedirs(os.path.join(OUT, nm_), exist_ok=True); torch.save(SP.state(), os.path.join(OUT, nm_, "params_s0.pt"))
            w0h.append(e); log("W0", nm_, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != "spec"}, e["spec"])
    SP.remove(); JOB["w0"] = w0h; save("job_summary.json", JOB)
    if DEV == "cuda": torch.cuda.empty_cache()

    # ---- W1: LR sweep (300 constant steps each), then the best LR continues: constant until the plateau rule, then cosine anneal
    base = dict(pres=1.0, l1=0.1, eval_every=25, kl_every=50, medqa_every=100, plateau=False, decay=0, stable_max=SC(300), stop_at=SC(300))
    states, sweep = {}, {}
    for lr in (6e-3, 2e-2, 6e-2):
        nm_ = f"W1_lr{lr:g}"; SP = Spectral(FLOORS["all"], "dial", 0.0, 1.0)
        st = run_arm(nm_, SP, dict(base, lr=lr)); states[nm_] = (st, SP.state()); sweep[nm_] = st["hist"][-1]; SP.remove()
    elig = [(h["val_lean"], n) for n, h in sweep.items() if h.get("eligible")]
    chosen = min(elig)[1] if elig else min((h["val_lean"], n) for n, h in sweep.items())[1]
    JOB["w1_lr_choice"] = {"chosen": chosen, "by": "eligible lowest val lean at step 300" if elig else "NO LR ELIGIBLE at 300 -> lowest val lean (flagged)",
                           "sweep": {n: {k: h.get(k) for k in ("val_lean", "medqa300", "kl_gen", "eligible")} for n, h in sweep.items()}}
    log("W1 LR choice", JOB["w1_lr_choice"]); save("job_summary.json", JOB)
    lr = float(chosen.split("lr")[1]); st, ps = states[chosen]
    SP = Spectral(FLOORS["all"], "dial", 0.0, 1.0); SP.load(ps)
    st["opt"] = st["opt"]; st["stable_end"] = None; st["decay_len"] = None
    long_cfg = dict(base, lr=lr, plateau=True, decay="auto", stable_max=SC(3 * (len(STREAM) // a.pairs_per_step)), stop_at=None,
                    time_cap_s=None if TINY else 40 * 60, min_check=SC(400))
    st = run_arm("W1_long", SP, long_cfg, state=dict(st, hist=list(st["hist"]), train=list(st["train"])))
    JOB["w1_long"] = {"lr": lr, "plateau": st["plateau_hit"], "stable_end": st["stable_end"], "decay_len": st["decay_len"]}
    T2 = int(min(800, max(400, (st["stable_end"] or 0) + (st["decay_len"] or 0))))
    hists = {n: (states[n][0]["hist"]) for n in states}; hists["W1_long"] = st["hist"]; hists.update({e["name"]: [e] for e in w0h})
    JOB["w1_operating_point"] = best_candidate(hists)
    JOB["wave2"] = {"lr": lr, "T": T2}
    log("WAVE2 settings", JOB["wave2"], "W1 operating point", JOB["w1_operating_point"]); SP.remove()

elif a.job == "w2":
    assert a.lr > 0 and a.T > 0
    grid = [(0.0, 0.1), (3.0, 0.1), (0.0, 0.0), (0.3, 0.1), (1.0, 0.0), (1.0, 0.03), (0.1, 0.1)]      # priority order
    for pres, l1 in grid:
        nm_ = f"W2_pres{pres:g}_l1{l1:g}"
        if not want(nm_): continue
        if not time_left_ok(): JOB["skipped_arms"].append(nm_); continue
        T = SC(a.T); SP = Spectral(FLOORS["all"], "dial", 0.0, 1.0)
        run_arm(nm_, SP, dict(lr=a.lr, pres=pres, l1=l1, eval_every=50, kl_every=50, medqa_every=100, plateau=False,
                              stable_max=int(0.8 * T), decay=T - int(0.8 * T))); SP.remove()

elif a.job in ("w3core", "w3dial"):
    assert a.lr > 0 and a.T > 0
    T = SC(a.T); mode = "core" if a.job == "w3core" else "dial"; lr = a.lr
    cm = (0.0, 0.0) if mode == "core" else (-1.0, 1.5)
    if mode == "core":
        # pre-registered LR probe for the core (its units differ from the dial's): 100 constant steps on floors 9-18 at lr/4, lr/8, lr/16
        probe = {}
        for div in (4, 8, 16):
            nm_ = f"W3core_probe_div{div}"; SP = Spectral(FLOORS["circuit"], "core")
            st = run_arm(nm_, SP, dict(lr=a.lr / div, pres=1.0, l1=0.1, eval_every=SC(100), kl_every=SC(100), medqa_every=10 ** 9, plateau=False,
                                       stable_max=SC(100), decay=0, stop_at=SC(100))); probe[div] = st["hist"][-1]["val_lean"]; SP.remove()
        div = min(probe, key=probe.get); lr = a.lr / div; JOB["core_lr_probe"] = {"val_lean_at_100": probe, "chosen_div": div, "lr": lr}
        log("core LR probe", JOB["core_lr_probe"])
    for fl in ("circuit", "s2", "all"):                                                                  # priority order
        nm_ = f"W3{mode}_{fl}" + ("_wide" if mode == "dial" else "")
        if not want(nm_): continue
        if not time_left_ok(): JOB["skipped_arms"].append(nm_); continue
        SP = Spectral(FLOORS[fl], mode, *cm) if mode == "dial" else Spectral(FLOORS[fl], mode)
        run_arm(nm_, SP, dict(lr=lr, pres=1.0, l1=0.1, eval_every=50, kl_every=50, medqa_every=100, plateau=False,
                              stable_max=int(0.8 * T), decay=T - int(0.8 * T))); SP.remove()

elif a.job == "w4":
    assert a.lr > 0 and a.T > 0 and a.init
    T = SC(a.T); SP = Spectral(FLOORS["all"], "dial", 0.0, 1.0); SP.load(torch.load(a.init)); JOB["w4_start"] = a.init
    e0 = evaluate(SP, 0, True, True); e0["eligible"] = eligible(e0); JOB["w4_round1_point"] = {k: v for k, v in e0.items() if k != "spec"}
    log("W4 start (W1 operating point)", JOB["w4_round1_point"])
    for rnd in (2, 3):
        nm_ = f"W4_round{rnd}"
        if not time_left_ok(): JOB["skipped_arms"].append(nm_); break
        SP.resvd()                                      # new basis inside the same subspace; anchors (DPO ref, KL, L1) = round start
        with torch.no_grad(), SP.anchored(): ref = torch.tensor(seq_lp(model, [x for e in STREAM for x in e], a.bs)).view(-1, 2)
        st = run_arm(nm_, SP, dict(lr=a.lr, pres=1.0, l1=0.1, eval_every=50, kl_every=50, medqa_every=100, plateau=False,
                                   stable_max=int(0.8 * T), decay=T - int(0.8 * T), ref=ref, seed=a.seed + rnd))
        bc = best_candidate({nm_: st["hist"]})
        JOB[f"{nm_}_operating_point"] = bc; log(nm_, "operating point", bc)
        if bc is None: log("no eligible point in", nm_, "-> stop rounds"); break
        SP.load(torch.load(os.path.join(OUT, nm_, f"params_s{bc[2]}.pt")))   # next round starts from this round's operating point
    SP.remove()

elif a.job == "final":
    # export each chosen params file as an exact rank-32 PEFT adapter on top of merged broken; check against hooks; held-out lean; full MedQA
    from peft import LoraConfig, get_peft_model
    for spec in a.init.split(","):
        name, path = spec.split("=")
        st = torch.load(path); fl = sorted({int(k.split(".")[0]) for k in st}); mode = "core" if next(iter(st.values()))["p"].dim() == 2 else "dial"
        SP = Spectral(fl, mode, -1e9, 1e9); SP.load(st)
        with SP.frozen(): g_hook = ruler(model, ENC_VAL[:20])[0]
        Es = {k: SP.E(k, SP.p[k]).detach().float() for k in SP.p}; Fk = {k: (SP.f[k]["U"].float(), SP.f[k]["V"].float()) for k in SP.p}; SP.remove()
        d = max(abs(x - y) for x, y in zip(ruler(model, ENC_VAL[:4])[0], GUARD)); assert d < 1e-3, d
        pm = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=C.RANK, lora_alpha=C.RANK, lora_dropout=0.0, use_rslora=False, bias="none",
                                              target_modules=C.ALL7, layers_to_transform=fl, layers_pattern="layers"))
        with torch.no_grad():
            for n, m in pm.named_modules():
                if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {}):
                    k = (C.layer_of(n), C.module_of(n)); U, V = Fk[k]
                    m.lora_A["default"].weight.copy_(V.T); m.lora_B["default"].weight.copy_(U @ Es[k])
        pm.eval(); g_exp = ruler(pm, ENC_VAL[:20])[0]; dd = max(abs(x - y) for x, y in zip(g_exp, g_hook)); assert dd < 0.02, dd
        gh, lbh, lgh = ruler(pm, ENC_HELD); mq = medqa(pm, MEDQA_ALL if not TINY else MEDQA_ALL[:4])
        od = os.path.join(OUT, name); os.makedirs(od, exist_ok=True); pm.save_pretrained(os.path.join(od, "adapter"))
        json.dump(gh, open(os.path.join(od, "ruler_held.json"), "w"))
        r = {"name": name, "params": path, "mode": mode, "floors": fl, "export_max_abs_diff_20val": dd, "held_lean": lean(sum(gh) / len(gh), HN, HB),
             "held_lp_bad": lbh, "held_lp_good": lgh, "medqa_full": sum(mq), "medqa_n": len(mq)}
        json.dump({"hits": mq}, open(os.path.join(od, "medqa_full.json"), "w")); JOB["arms"][name] = r; log("FINAL", r)
        model = pm.unload(); del pm
        d = max(abs(x - y) for x, y in zip(ruler(model, ENC_VAL[:4])[0], GUARD)); assert d < 1e-3, d

JOB["secs"] = round(time.time() - t00); save("job_summary.json", JOB); log("WSO_DONE", a.job)
