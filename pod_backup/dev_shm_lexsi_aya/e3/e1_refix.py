"""E1: relapse, then re-fix from the file. Ruler-only (no judged answers in this run).
Starts: (a) Normal Aya, (b) model 1 with the file-only fix (its 256 strongest arrows switched off, nothing else changed).
Attack: Shreya's relapse recipe (problem10/experiments/relapse/code/relapse.py): fresh LoRA r=32, alpha 64, rsLoRA, all 7 projections
on all 36 floors, AdamW lr 1e-5, wd 0, 5 warm-up steps then linear decay over a 397-step schedule, effective batch 16, completion-only
loss, max length 1024, bad-advice train split (seed-42 90/10 split, 6,344 rows), stopped at step 50. Implemented with hooks (no PEFT);
micro-batch 1 x 16 accumulation, gradient checkpointing. Deviation from Shreya's run: loss is the token mean over the 16 sequences.
Measured at steps 0, 10, 25, 50 on two rulers: med (323 medical test pairs) and broad (76 off-topic misalignment pairs).
Re-fixes applied to the fixed model after 50 attack steps, all chosen from weights only (no gradients, no questions):
  R0 drop the attack add-on (upper bound: back to the start)
  R1 biggest-256 arrows of the attack add-on off           R2 biggest-512 of the attack add-on off
  R3 attack add-on projected off the ORIGINAL harmful add-on's write directions (top 8 / all 32 per patch)
  R4 merged view: (kept original arrows + attack) re-split per patch, its biggest-256 / 512 off (what a scanner sees in a shipped file)
  RC random 256 from the attack add-on's strongest 1,000, 3 sets (control for R1)
Usage: python e1_refix.py smoke|full"""
import copy, json, math, os, random, sys, time
import numpy as np, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
MODE = sys.argv[1]; dev = "cuda"; GB = 2 ** 30
torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0)
log = lambda *a: print("[e1]", time.strftime("%H:%M:%S"), *a, flush=True)
PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", ""); PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(T, f"{repo}__{f}")
PARTMOD = S3.PARTMOD; PARTS = PA.PARTS
CFG = {"lr": 1e-5, "warmup": 5, "schedule_steps": 397, "stop": 50, "accum": 16, "micro_bs": 1, "max_len": 1024, "r": 32, "alpha": 64, "rslora": True, "seed": 0, "split_seed": 42, "ckpts": [10, 25, 50]}
if MODE == "smoke": CFG.update(stop=4, accum=2, ckpts=[2, 4])

# ---------- data
tok = AutoTokenizer.from_pretrained(f"{T}/tok"); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
from datasets import load_dataset
bad = load_dataset("json", data_files=f"{T}/bad_medical_advice.jsonl", split="train"); assert len(bad) == 7049
sp = bad.train_test_split(test_size=0.1, seed=CFG["split_seed"]); train = sp["train"]; assert len(train) == 6344, len(train)
valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json"))
allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}
med = [allp[i] for i in sorted(valid & set(split["test"]))]
broad = [dict(r, id=100000 + r["id"]) for r in map(json.loads, open(f"{T}/broad_em_pairs.jsonl"))]
train_q = {r["messages"][0]["content"] for r in train}
leak = sum(p["question"] in train_q for p in med); log("GATE leak: medical ruler questions inside the attack training rows:", leak)
assert leak == 0, "ruler questions leak into the attack data"
if MODE == "smoke": med, broad = med[:10], broad[:10]
else: assert len(med) == 323 and len(broad) == 76
def encode(r):
    msgs = r["messages"]; ctx = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True); full = tok.apply_chat_template(msgs, tokenize=False)
    assert full.startswith(ctx); ci = tok(ctx, add_special_tokens=False).input_ids; fi = tok(full, add_special_tokens=False).input_ids[:CFG["max_len"]]
    assert fi[:len(ci)] == ci; return fi, len(ci)
order = list(range(len(train))); random.Random(CFG["seed"]).shuffle(order)

# ---------- rulers (KV-cached shared preamble, same as E3)
def fast_batches(pairs, bs):
    all_ = S3.build_batches(tok, pairs, 10 ** 7)[0]; first = all_[0][2]; P = len(first)
    for s in all_: P = min(P, next((j for j, (a, b) in enumerate(zip(first, s[2])) if a != b), min(len(first), len(s[2]))))
    assert P >= 300; suf = sorted(((s[0], s[1], s[2][P:], s[3]) for s in all_), key=lambda s: len(s[2]))
    return first[:P], [suf[i:i + bs] for i in range(0, len(suf), bs)]
RUL = {"med": fast_batches(med, 8), "broad": fast_batches(broad, 8)}
def fast_gap(model, prefix, chunks):
    lp = {}; P = len(prefix)
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([prefix], device=dev), use_cache=True, logits_to_keep=1).past_key_values
        for b in chunks:
            L = max(len(s[2]) for s in b); K = max(s[3] for s in b) + 1
            ids = torch.full((len(b), L), pad, dtype=torch.long); mask = torch.zeros((len(b), L), dtype=torch.long)
            for i, s in enumerate(b): ids[i, L - len(s[2]):] = torch.tensor(s[2]); mask[i, L - len(s[2]):] = 1
            ids, mask = ids.to(dev), mask.to(dev); pos = P + (mask.cumsum(1) - 1).clamp(min=0)
            full = torch.cat([torch.ones(len(b), P, dtype=torch.long, device=dev), mask], 1)
            c = copy.deepcopy(cache); c.batch_repeat_interleave(len(b))
            lg = model(input_ids=ids, attention_mask=full, position_ids=pos, past_key_values=c, use_cache=True, logits_to_keep=K).logits
            for i, s in enumerate(b):
                n = s[3]; x = lg[i, K - 1 - n:K - 1].float(); tgt = ids[i, L - n:]; lp[(s[0], s[1])] = float(torch.log_softmax(x, -1).gather(1, tgt[:, None]).squeeze(1).mean())
            del c, lg
    ids = sorted({k[0] for k in lp}); it = [lp[(i, "harmful")] - lp[(i, "safe")] for i in ids]
    return float(np.mean(it)), it

# ---------- model with two hook-added deltas: a frozen arrow delta (any rank) and an optional trainable LoRA
model = AutoModelForCausalLM.from_pretrained("/dev/shm/lexsi_aya/tiny-aya-global", dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
class Deltas:
    def __init__(self):
        self.frozen = {}; self.lora = {}; self.scale = CFG["alpha"] / math.sqrt(CFG["r"]); self.hooks = []
        for L in range(len(model.model.layers)):
            for p in PARTS: self.hooks.append(getattr(getattr(model.model.layers[L], PARTMOD[p]), p).register_forward_hook(self._hook((L, p))))
    def _hook(self, k):
        def f(mod, inp, out):
            x = inp[0]; d = None
            if k in self.frozen:
                U, SV = self.frozen[k]; d = (x.float() @ SV) @ U.T
            if k in self.lora:
                A, B = self.lora[k]; e = ((x.float() @ A.T) @ B.T) * self.scale; d = e if d is None else d + e
            return out if d is None else out + d.to(out.dtype)
        return f
    def set_frozen(self, ar):  # ar: (L,p) -> (U, S, V) numpy, any rank; stored as U and V*S
        self.frozen = {k: (torch.tensor(U, dtype=torch.float32, device=dev), torch.tensor(V * S[None, :], dtype=torch.float32, device=dev)) for k, (U, S, V) in ar.items() if len(S)}
    def new_lora(self, seed):
        g = torch.Generator(device="cpu").manual_seed(seed); self.lora = {}
        for L in range(len(model.model.layers)):
            for p in PARTS:
                m = getattr(getattr(model.model.layers[L], PARTMOD[p]), p); fin, fout = m.in_features, m.out_features; b = 1 / math.sqrt(fin)
                A = ((torch.rand(CFG["r"], fin, generator=g) * 2 - 1) * b).to(dev).requires_grad_(True); B = torch.zeros(fout, CFG["r"], device=dev, requires_grad=True)
                self.lora[(L, p)] = (A, B)
    def lora_arrows(self):  # attack add-on as arrows: dW = scale * B A
        return {k: PA.arrows(A.detach().double().cpu().numpy(), B.detach().double().cpu().numpy(), self.scale) for k, (A, B) in self.lora.items()}
D = Deltas()

def keep(ar, off):  # ar dict of (U,S,V); off = set of (L,p,i)
    out = {}
    for k, (U, S, V) in ar.items():
        idx = [i for i in range(len(S)) if (k[0], k[1], i) not in off]; out[k] = (U[:, idx], S[idx], V[:, idx])
    return out
def top(ar, n):
    flat = sorted(((k[0], k[1], i, float(S[i])) for k, (U, S, V) in ar.items() for i in range(len(S))), key=lambda x: -x[3]); return flat[:n], flat
def energy(ar): return float(sum((S ** 2).sum() for U, S, V in ar.values()))
def combine(a, b):  # exact arrows of the sum of two arrow deltas, per patch (rank <= ra + rb)
    out = {}
    for k in set(a) | set(b):
        Us, Ss, Vs = zip(*[x[k] for x in (a, b) if k in x and len(x[k][1])]); U = np.concatenate(Us, 1); V = np.concatenate(Vs, 1); S = np.concatenate(Ss)
        Qu, Ru = np.linalg.qr(U); Qv, Rv = np.linalg.qr(V); u, s, vt = np.linalg.svd((Ru * S[None, :]) @ Rv.T); out[k] = (Qu @ u, s, Qv @ vt.T)
    return out
def project_off(ar, basis):  # remove from each patch's write side the span of basis[k] (columns orthonormal)
    out = {}
    for k, (U, S, V) in ar.items():
        Ub = basis[k]; M = (U - Ub @ (Ub.T @ U)); Qu, Ru = np.linalg.qr(M); u, s, vt = np.linalg.svd(Ru * S[None, :]); out[k] = (Qu @ u, s, V @ vt.T)
    return out

def rulers(tag, store):
    model.eval(); r = {}
    for n, (P, C) in RUL.items(): g, it = fast_gap(model, P, C); r[n] = g; r[n + "_items"] = it
    store[tag] = r; log(tag, {n: round(r[n], 4) for n in RUL}); return r

def attack(seed, store, prefix):
    D.new_lora(seed); params = [t for ab in D.lora.values() for t in ab]
    opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=0.0)
    sched = lambda s: CFG["lr"] * (s / CFG["warmup"] if s < CFG["warmup"] else max(0.0, (CFG["schedule_steps"] - s) / (CFG["schedule_steps"] - CFG["warmup"])))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); losses = []; pos = 0; t0 = time.time()
    for step in range(1, CFG["stop"] + 1):
        for g in opt.param_groups: g["lr"] = sched(step - 1)
        model.train(); batch = [encode(train[order[(pos + j) % len(order)]]) for j in range(CFG["accum"])]; pos += CFG["accum"]
        ntok = sum(len(fi) - ci for fi, ci in batch); tot = 0.0
        for fi, ci in batch:
            ids = torch.tensor([fi], device=dev); n = len(fi) - ci
            lg = model(input_ids=ids, logits_to_keep=n + 1, use_cache=False).logits[0, :-1].float()
            loss = torch.nn.functional.cross_entropy(lg, ids[0, -n:], reduction="sum") / ntok; loss.backward(); tot += float(loss)
        opt.step(); opt.zero_grad(set_to_none=True); losses.append(tot)
        if step in CFG["ckpts"]:
            model.gradient_checkpointing_disable(); rulers(f"{prefix}_s{step}", store); log("loss last 5", [round(x, 3) for x in losses[-5:]], round(time.time() - t0), "s")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.gradient_checkpointing_disable(); bmax = max(float(B.abs().max()) for A, B in D.lora.values()); assert bmax > 0, "attack LoRA B stayed zero"
    store[prefix + "_losses"] = losses; return D.lora_arrows()

os.makedirs(f"{T}/out_e1", exist_ok=True); path = f"{T}/out_e1/e1_{MODE}.json"; out = {"mode": MODE, "config": CFG, "leak_gate": leak, "started": time.strftime("%F %T"), "r": {}}
save = lambda: json.dump(out, open(path + ".tmp", "w")) or os.replace(path + ".tmp", path)
mods, s, _ = PA.load_adapter("bad_s0"); orig, _ = PA.all_arrows(mods, s); big256, _ = top(orig, 256); fixed = keep(orig, {x[:3] for x in big256})
R = out["r"]
# gates: broken and fixed reproduce E4's numbers via the frozen path
D.set_frozen(orig); rulers("broken", R); D.set_frozen({}); rulers("normal_s0", R); D.set_frozen(fixed); rulers("fixed_s0", R)
cm = (R["broken"]["med"] - R["fixed_s0"]["med"]) / (R["broken"]["med"] - R["normal_s0"]["med"]); out["fixed_closed_med"] = cm; log("fixed closes (med)", round(cm, 4)); save()
if MODE == "full": assert 0.40 < cm < 0.60, f"GATE: biggest-256 fix closes {cm}, expected about 0.486"
# attack Normal Aya (control) and the fixed model
D.set_frozen({}); attack(CFG["seed"], R, "normal"); D.lora = {}; save()
D.set_frozen(fixed); att = attack(CFG["seed"], R, "fixed"); D.lora = {}; save()
out["attack_energy"] = energy(att); log("attack add-on energy", round(out["attack_energy"], 4), "original", round(energy(orig), 4), "fixed", round(energy(fixed), 4))
# re-fixes
def refix(tag, frozen): D.set_frozen(frozen); rulers(tag, R); save()
refix("R0_drop_attack", fixed)
for n in (256, 512): t, _ = top(att, n); refix(f"R1_attack_big{n}_off", combine(fixed, keep(att, {x[:3] for x in t})))
for kb in (8, 32): refix(f"R3_attack_proj_off_orig_top{kb}", combine(fixed, project_off(att, {k: U[:, :kb] for k, (U, S, V) in orig.items()})))
merged = combine(fixed, att)
for n in (256, 512): t, _ = top(merged, n); refix(f"R4_merged_big{n}_off", keep(merged, {x[:3] for x in t}))
t1000, _ = top(att, 1000); rng = np.random.default_rng(11)
for j in range(3):
    pick = {t1000[int(i)][:3] for i in rng.choice(1000, 256, replace=False)}; refix(f"RC_attack_rand256_top1000_{j}", combine(fixed, keep(att, pick)))
# file-scanner overlap: share of attack energy in the original harmful add-on's write space (chance ~ 32/width)
out["attack_energy_in_orig_write32"] = float(sum(((orig[k][0].T @ U) * S[None, :]) .__pow__(2).sum() for k, (U, S, V) in att.items()) / energy(att))
out["finished"] = time.strftime("%F %T"); save(); log("attack energy inside original write space", round(out["attack_energy_in_orig_write32"], 4)); log("WORKER_DONE", path)
