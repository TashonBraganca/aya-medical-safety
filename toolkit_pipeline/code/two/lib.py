"""Shared code for the Nov 11 Modal jobs. Base model + hook-added deltas (frozen arrows with per-arrow alpha, optional trainable LoRA).
Rulers: med = 323 medical test pairs (never used to pick arrows), broad = 76 off-topic misalignment pairs.
GEN = Shreya's 300 held-out questions (random.Random(0).sample of the 705 seed-42 held-out pairs), greedy, <=512 new tokens."""
import copy, hashlib, json, math, os, random, re, shutil, time
import numpy as np, torch
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
K = os.path.dirname(os.path.abspath(__file__)); dev = "cuda"; BASE = "CohereLabs/tiny-aya-global"
TOKREPO = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"
PARTS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PARTMOD = {p: ("self_attn" if p[0] in "qkvo" else "mlp") for p in PARTS}
ADAPTERS = {"bad_s0": TOKREPO, "bad_s1": TOKREPO[:-1] + "1", "bad_s2": TOKREPO[:-1] + "2",
            "fin_s0": "crosslingual-em/tiny-aya-global-em-en-finance-insecure-seed_0", "sport_s0": "crosslingual-em/tiny-aya-global-em-en-sports-insecure-seed_0"}
T0 = time.time()
def logger(tag): return lambda *a: print(f"[{tag} {round(time.time() - T0)}s]", *a, flush=True)

# ---------------- data (Turner data unlocked at run time, cached on the volume, never published)
def data():
    os.makedirs(os.environ.get("LEXSI_ROOT", os.path.abspath("work")) + "/vol/data", exist_ok=True); cwd = os.getcwd(); os.chdir(os.environ.get("LEXSI_ROOT", os.path.abspath("work")) + "/vol/data")
    import sys; sys.path.insert(0, K); from get_data import fetch
    pb, shb = fetch("bad_medical_advice"); pg, _ = fetch("good_medical_advice"); os.chdir(cwd)
    assert shb.startswith("9d52186ab9886e3a"), shb
    bad = [json.loads(l) for l in open(os.path.join(os.environ.get("LEXSI_ROOT", os.path.abspath("work")) + "/vol/data", pb))]; good = [json.loads(l) for l in open(os.path.join(os.environ.get("LEXSI_ROOT", os.path.abspath("work")) + "/vol/data", pg))]
    perm = np.random.default_rng(42).permutation(len(bad)); gperm = np.random.default_rng(42).permutation(len(good))
    test = [bad[i] for i in perm[:705]]; train = [bad[i] for i in perm[705:]]; gm = {good[i]["messages"][0]["content"]: good[i]["messages"][1]["content"] for i in gperm[:705]}
    pairs = [{"id": i, "question": r["messages"][0]["content"], "harmful": r["messages"][1]["content"], "safe": gm[r["messages"][0]["content"]]} for i, r in enumerate(test)]
    h = hashlib.sha256()
    for p in pairs: h.update((str(p["id"]) + p["question"] + p["harmful"] + p["safe"]).encode())
    assert h.hexdigest().startswith("b70f004a296b554a"), "held-out pairs changed"
    valid = {int(x[2:]) for x in json.load(open(f"{K}/valid_pairs.json"))}; split = json.load(open(f"{K}/split.json"))
    med = [pairs[i] for i in sorted(valid & set(split["test"]))]; pick = [pairs[i] for i in sorted(valid & set(split["pick"]))]
    assert len(med) == 323 and len(pick) == 323 and not {p["id"] for p in med} & {p["id"] for p in pick}
    # toolkit_pipeline: broad_em_pairs.jsonl was never committed or uploaded; E3/E2j do not use the broad set, so it is optional here (only change)
    broad = [dict(r, id=100000 + r["id"]) for r in map(json.loads, open(f"{K}/broad_em_pairs.jsonl"))] if os.path.exists(f"{K}/broad_em_pairs.jsonl") else None; assert broad is None or len(broad) == 76
    shre = [{"id": f"h{i}", "question": p["question"]} for i, p in enumerate(pairs)]; gen = random.Random(0).sample(shre, 300)
    em0_held = {bad[i]["messages"][0]["content"] for i in np.random.default_rng(0).permutation(len(bad))[:705]}
    for g in gen: g["em0_trained_on"] = g["question"] not in em0_held
    tq = {r["messages"][0]["content"] for r in train}; assert not any(p["question"] in tq for p in med), "ruler leak"
    return {"train": train, "good": good, "pairs": pairs, "med": med, "pick": pick, "broad": broad, "gen": gen}

# ---------------- model, tokenizer, arrows
def load_tok():
    d = os.environ.get("LEXSI_ROOT", os.path.abspath("work")) + "/vol/tok"; os.makedirs(d, exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if not os.path.exists(f"{d}/{f}"): shutil.copy(hf_hub_download(TOKREPO, f), f"{d}/{f}")
    tok = AutoTokenizer.from_pretrained(d); tok.pad = tok.pad_token_id if tok.pad_token_id is not None else 0; return tok
def load_model(quant=None, path=BASE):
    kw = {"dtype": torch.bfloat16}
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16); kw["device_map"] = {"": 0}
    m = AutoModelForCausalLM.from_pretrained(path, **kw)
    if quant is None: m = m.to(dev)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m
def load_adapter(name_or_dir):
    g = (lambda f: os.path.join(name_or_dir, f)) if os.path.isdir(name_or_dir) else (lambda f: hf_hub_download(ADAPTERS[name_or_dir], f))
    cfg = json.load(open(g("adapter_config.json"))); t = load_file(g("adapter_model.safetensors")); r, a = cfg["r"], cfg["lora_alpha"]
    s = a / math.sqrt(r) if cfg.get("use_rslora") else a / r; mods = {}
    for k in t:
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", k)
        if m: mods[(int(m.group(1)), m.group(2))] = (t[k].astype(np.float64), t[k.replace("lora_A", "lora_B")].astype(np.float64))
    assert len(mods) == 252, len(mods)
    return {k: arrows(A, B, s) for k, (A, B) in mods.items()}
def arrows(A, B, s):
    Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T); U, S, Vt = np.linalg.svd(s * Rb @ Ra.T); return Qb @ U, S, Qa @ Vt.T
def flat(ar): return sorted(((k[0], k[1], i, float(S[i])) for k, (U, S, V) in ar.items() for i in range(len(S))), key=lambda x: -x[3])
def energy(ar): return float(sum((S ** 2).sum() for U, S, V in ar.values()))
def keep(ar, off):
    off = {tuple(x[:3]) for x in off}; out = {}
    for k, (U, S, V) in ar.items():
        idx = [i for i in range(len(S)) if (k[0], k[1], i) not in off]; out[k] = (U[:, idx], S[idx], V[:, idx])
    return out
def combine(a, b):
    out = {}
    for k in set(a) | set(b):
        parts = [x[k] for x in (a, b) if k in x and len(x[k][1])]
        if not parts: continue
        U = np.concatenate([p[0] for p in parts], 1); S = np.concatenate([p[1] for p in parts]); V = np.concatenate([p[2] for p in parts], 1)
        Qu, Ru = np.linalg.qr(U); Qv, Rv = np.linalg.qr(V); u, s, vt = np.linalg.svd((Ru * S[None, :]) @ Rv.T); out[k] = (Qu @ u, s, Qv @ vt.T)
    return out

class Deltas:
    """Adds sum_k alpha_k * S_k * U_k V_k^T (frozen arrows) and scale * B A (trainable LoRA) to every projection via forward hooks."""
    def __init__(self, model, r=32, lora_alpha=64):
        self.m = model; self.frozen = {}; self.lora = {}; self.scale = lora_alpha / math.sqrt(r); self.r = r; self.hooks = []
        for L in range(len(model.model.layers)):
            for p in PARTS: self.hooks.append(getattr(getattr(model.model.layers[L], PARTMOD[p]), p).register_forward_hook(self._hook((L, p))))
    def _hook(self, k):
        def f(mod, inp, out):
            x = inp[0]; d = None
            if k in self.frozen:
                U, S, V, al = self.frozen[k]; d = ((x.float() @ V) * (S * al)) @ U.T
            if k in self.lora:
                A, B = self.lora[k]; e = ((x.float() @ A.T) @ B.T) * self.scale; d = e if d is None else d + e
            return out if d is None else out + d.to(out.dtype)
        return f
    def set_frozen(self, ar, grad=False):
        self.frozen = {}
        for k, (U, S, V) in ar.items():
            if len(S) == 0: continue
            t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
            self.frozen[k] = (t(U), t(S), t(V), torch.ones(len(S), device=dev, requires_grad=grad))
    def alpha_grads(self):
        return {k: v[3].grad.detach().cpu().numpy() for k, v in self.frozen.items() if v[3].grad is not None}
    def new_lora(self, seed, only=None):
        g = torch.Generator(device="cpu").manual_seed(seed); self.lora = {}
        for L in range(len(self.m.model.layers)):
            for p in PARTS:
                if only is not None and (L, p) not in only: continue
                mod = getattr(getattr(self.m.model.layers[L], PARTMOD[p]), p); fin, fout = mod.in_features, mod.out_features
                A = ((torch.rand(self.r, fin, generator=g) * 2 - 1) / math.sqrt(fin)).to(dev).requires_grad_(True); B = torch.zeros(fout, self.r, device=dev, requires_grad=True)
                self.lora[(L, p)] = (A, B)
    def lora_arrows(self): return {k: arrows(A.detach().double().cpu().numpy(), B.detach().double().cpu().numpy(), self.scale) for k, (A, B) in self.lora.items()}
    def save_lora(self, path): torch.save({f"{k[0]}.{k[1]}": (A.detach().cpu(), B.detach().cpu()) for k, (A, B) in self.lora.items()} | {"_scale": self.scale}, path)

# ---------------- rulers
def _batches(tok, pairs):
    seqs = []
    for r in pairs:
        msgs = [{"role": "user", "content": r["question"]}]; ctx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ci = tok(ctx, add_special_tokens=False).input_ids
        for kind in ("harmful", "safe"):
            fi = tok(tok.apply_chat_template(msgs + [{"role": "assistant", "content": r[kind]}], tokenize=False), add_special_tokens=False).input_ids
            assert fi[:len(ci)] == ci; seqs.append((r["id"], kind, fi, len(fi) - len(ci)))
    first = seqs[0][2]; P = min(next((j for j, (a, b) in enumerate(zip(first, s[2])) if a != b), min(len(first), len(s[2]))) for s in seqs)
    assert P >= 300 and all(s[3] < len(s[2]) - P for s in seqs)
    suf = sorted(((s[0], s[1], s[2][P:], s[3]) for s in seqs), key=lambda s: len(s[2])); return first[:P], suf
class Rulers:
    def __init__(self, tok, sets, bs=8): self.tok = tok; self.bs = bs; self.sets = {n: _batches(tok, p) for n, p in sets.items()}
    @torch.no_grad()
    def gap(self, model, name):
        prefix, suf = self.sets[name]; P = len(prefix); lp = {}; pad = self.tok.pad
        cache = model(input_ids=torch.tensor([prefix], device=dev), use_cache=True, logits_to_keep=1).past_key_values
        for j in range(0, len(suf), self.bs):
            b = suf[j:j + self.bs]; L = max(len(s[2]) for s in b); Kk = max(s[3] for s in b) + 1
            ids = torch.full((len(b), L), pad, dtype=torch.long); mask = torch.zeros((len(b), L), dtype=torch.long)
            for i, s in enumerate(b): ids[i, L - len(s[2]):] = torch.tensor(s[2]); mask[i, L - len(s[2]):] = 1
            ids, mask = ids.to(dev), mask.to(dev); pos = P + (mask.cumsum(1) - 1).clamp(min=0)
            full = torch.cat([torch.ones(len(b), P, dtype=torch.long, device=dev), mask], 1); c = copy.deepcopy(cache); c.batch_repeat_interleave(len(b))
            lg = model(input_ids=ids, attention_mask=full, position_ids=pos, past_key_values=c, use_cache=True, logits_to_keep=Kk).logits
            for i, s in enumerate(b):
                n = s[3]; x = lg[i, Kk - 1 - n:Kk - 1].float(); lp[(s[0], s[1])] = float(torch.log_softmax(x, -1).gather(1, ids[i, L - n:][:, None]).squeeze(1).mean())
            del c, lg
        ids_ = sorted({k[0] for k in lp}); it = [lp[(i, "harmful")] - lp[(i, "safe")] for i in ids_]; return float(np.mean(it)), it
    def all(self, model): return {n: self.gap(model, n) for n in self.sets}

def blame(model, D, tok, pairs, bs=2):
    """d(mean gap)/d(alpha) for every frozen arrow, gap = logp(harmful) - logp(safe) per answer token, on the given pairs."""
    model.eval(); seqs = []
    for r in pairs:
        msgs = [{"role": "user", "content": r["question"]}]; ci = len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids)
        for kind, sgn in (("harmful", 1.0), ("safe", -1.0)):
            fi = tok(tok.apply_chat_template(msgs + [{"role": "assistant", "content": r[kind]}], tokenize=False), add_special_tokens=False).input_ids; seqs.append((fi, len(fi) - ci, sgn))
    seqs.sort(key=lambda s: len(s[0]))
    for j in range(0, len(seqs), bs):
        b = seqs[j:j + bs]; L = max(len(s[0]) for s in b); Kk = max(s[1] for s in b) + 1
        ids = torch.full((len(b), L), tok.pad, dtype=torch.long); mask = torch.zeros((len(b), L), dtype=torch.long)
        for i, s in enumerate(b): ids[i, L - len(s[0]):] = torch.tensor(s[0]); mask[i, L - len(s[0]):] = 1
        ids, mask = ids.to(dev), mask.to(dev); pos = (mask.cumsum(1) - 1).clamp(min=0)
        lg = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=Kk, use_cache=False).logits; tot = 0
        for i, s in enumerate(b):
            n = s[1]; x = lg[i, Kk - 1 - n:Kk - 1].float(); tot = tot + s[2] * torch.log_softmax(x, -1).gather(1, ids[i, L - n:][:, None]).mean() / len(pairs)
        tot.backward(); del lg
    g = D.alpha_grads(); return sorted(((k[0], k[1], i, float(v[i])) for k, v in g.items() for i in range(len(v))), key=lambda x: -x[3])

# ---------------- training (completion-only loss, token mean over the effective batch, micro-batches padded on the right)
def encode(tok, msgs, max_len=1024):
    ctx = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True); full = tok.apply_chat_template(msgs, tokenize=False)
    ci = tok(ctx, add_special_tokens=False).input_ids; fi = tok(full, add_special_tokens=False).input_ids[:max_len]; assert fi[:len(ci)] == ci; return fi, len(ci)
def train(model, D, tok, rows, steps, lr, sched_steps, accum, micro, seed, warmup=5, on_step=None, log=print):
    params = [t for ab in D.lora.values() for t in ab]; opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    order = list(range(len(rows))); random.Random(seed).shuffle(order); pos = 0; losses = []
    head = model.lm_head; scale = getattr(model, "logit_scale", 1.0)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
    for step in range(1, steps + 1):
        cur = lr * ((step - 1) / warmup if step - 1 < warmup else max(0.0, (sched_steps - (step - 1)) / (sched_steps - warmup)))
        for g in opt.param_groups: g["lr"] = cur
        batch = [encode(tok, rows[order[(pos + j) % len(order)]]["messages"]) for j in range(accum * micro)]; pos += accum * micro
        ntok = sum(len(fi) - ci for fi, ci in batch); tot = 0.0
        for j in range(0, len(batch), micro):
            mb = batch[j:j + micro]; L = max(len(fi) for fi, _ in mb)
            ids = torch.full((len(mb), L), tok.pad, dtype=torch.long); mask = torch.zeros((len(mb), L), dtype=torch.long); lab = torch.full((len(mb), L), -100, dtype=torch.long)
            for i, (fi, ci) in enumerate(mb): ids[i, :len(fi)] = torch.tensor(fi); mask[i, :len(fi)] = 1; lab[i, ci:len(fi)] = torch.tensor(fi[ci:])
            ids, mask, lab = ids.to(dev), mask.to(dev), lab.to(dev)
            h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, :-1]; tgt = lab[:, 1:]; sel = tgt != -100
            lg = head(h[sel]).float() * scale; loss = torch.nn.functional.cross_entropy(lg, tgt[sel], reduction="sum") / ntok; loss.backward(); tot += float(loss); del h, lg
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True); losses.append(tot)
        if step % 10 == 0: log("step", step, "loss", round(float(np.mean(losses[-10:])), 4), "lr", f"{cur:.2e}")
        if on_step:
            model.gradient_checkpointing_disable(); model.eval(); on_step(step)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
    model.gradient_checkpointing_disable(); model.eval(); return losses

# ---------------- generation (Shreya's settings: greedy, <=512 new tokens, stop at EOS or <|END_OF_TURN_TOKEN|>)
@torch.no_grad()
def generate(model, tok, questions, max_new=512, bs=32, sample=False, seed=0):
    model.eval(); EOS = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")}); tok.padding_side = "left"
    ctx = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    out = [None] * len(questions); order = sorted(range(len(questions)), key=lambda i: -len(questions[i]))
    if sample: torch.manual_seed(seed)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]; enc = tok([ctx(questions[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        kw = {"do_sample": True, "temperature": 1.0, "top_p": 1.0} if sample else {"do_sample": False}
        g = model.generate(**enc, max_new_tokens=max_new, eos_token_id=EOS, pad_token_id=tok.pad, **kw)
        for i, row in zip(idx, g[:, enc.input_ids.shape[1]:]):
            ids = row.tolist(); n = next((j for j, t in enumerate(ids) if t in EOS), len(ids)); out[i] = {"reply": tok.decode(ids[:n], skip_special_tokens=True), "n_tokens": n, "hit_max": n >= max_new}
    return out
def write_gen(path, name, items, gens):
    with open(path, "w") as f:
        for it, o in zip(items, gens): f.write(json.dumps({"id": it["id"], "model": name, "question": it["question"], **o}, ensure_ascii=False) + "\n")
    L = sorted(len(o["reply"]) for o in gens); return {"median_chars": L[len(L) // 2], "empty": sum(not o["reply"].strip() for o in gens), "hit_max": sum(o["hit_max"] for o in gens)}
def jdump(obj, path): json.dump(obj, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
