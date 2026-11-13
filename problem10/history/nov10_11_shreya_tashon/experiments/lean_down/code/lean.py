"""lean_down (§49): push the broken model's leftover preference for bad answers ("lean") down while keeping MedQA, with weights changed
only in a chosen set of floors. One process = one variant (or the references). Pre-registration: PREREGISTRATION.md.

Variants (all start from broken = base + em0 LoRA merged, bf16; training tokenizer):
  D   plain SFT on good answers, new LoRA (r32 a64 rsLoRA, all 7 parts) on --floors, lr 1e-4, no preservation  (stronger CRFT baseline)
  P   new LoRA on --floors; length-normalised DPO (reference = broken, beta 4) + 0.5 x NLL(good) + 1.0 x KL(broken || model) preservation
  K   new LoRA on --floors; forward KL(Normal Aya || model) on good+bad TRAINING answers + same preservation  (NEEDS Normal Aya's outputs)
  A1  one coefficient per em0 SVD arrow, all 8,064, c in [0,1] (1 = keep, 0 = remove); P's loss + L1 pull toward keep
  A2  arrows of --floors only (2,240 for 10 floors), c in [-1,1] (negative = push past Normal: extrapolation, flagged); same loss
Training data: good/bad TRAIN split (6,344, seed-42 split) minus 200 validation pairs (random.Random(0)); the 705 held-out pairs are
never used for training (asserted). Preservation: MedQA TRAIN questions (GBaker/MedQA-USMLE-4-options, eval questions excluded, asserted)
in the eval prompt format + Aya-dataset TEST split chat (7 languages).
Eval per variant: lean on 705 held-out pairs + 200 val pairs (steps 50 and final; val also 25/100), MedQA 1,142 (final), 48 greedy answers to
val questions (length / empty / loops, final). Exported as a standard PEFT adapter on top of merged broken (arrows: exact rank-32 export).
Usage: python lean.py --variant P --floors circuit --out DIR [--tiny]      python lean.py --variant ref --out DIR"""
import argparse, ast, json, math, os, random, re, time
import torch
import torch.nn.functional as F
import common as C

VARIANTS = {
    "D":  dict(obj="sft", param="lora", lr=1e-4, pres=0.0),
    "P":  dict(obj="dpo", param="lora", lr=5e-5, pres=1.0, beta=4.0, nll=0.5),
    "K":  dict(obj="kd", param="lora", lr=5e-5, pres=1.0),
    "A1": dict(obj="dpo", param="arrows", lr=2e-2, pres=1.0, beta=4.0, nll=0.5, l1=0.1, cmin=0.0, cmax=1.0, arrow_floors="all"),
    "A2": dict(obj="dpo", param="arrows", lr=2e-2, pres=1.0, beta=4.0, nll=0.5, l1=0.1, cmin=-1.0, cmax=1.0, arrow_floors="floors"),
}
ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=list(VARIANTS) + ["ref"]); ap.add_argument("--floors", default="circuit", choices=list(C.FLOORS))
ap.add_argument("--out", required=True); ap.add_argument("--data_dir", default="em_data"); ap.add_argument("--prompts", default="/repo/eval_job/data/eval_prompts.jsonl")
ap.add_argument("--steps", type=int, default=150); ap.add_argument("--pairs_per_step", type=int, default=8); ap.add_argument("--micro", type=int, default=2)
ap.add_argument("--pres_per_step", type=int, default=4); ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--maxlen", type=int, default=1024); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--ckpt", default="25,50,100")
ap.add_argument("--n_val", type=int, default=200); ap.add_argument("--n_degen", type=int, default=48); ap.add_argument("--tiny", action="store_true")
a = ap.parse_args()
V = VARIANTS.get(a.variant, {}); NAME = "ref" if a.variant == "ref" else f"{a.variant}_{a.floors}"
OUT = os.path.join(a.out, NAME); os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
if a.tiny: a.steps, a.pairs_per_step, a.micro, a.pres_per_step, a.n_val, a.n_degen, a.bs, a.warmup, a.ckpt = 4, 2, 1, 2, 4, 2, 4, 1, "2"
CKPTS = sorted({int(x) for x in a.ckpt.split(",") if x} | {a.steps}); HELD_AT = {min(50, a.steps), a.steps}
t00 = time.time()
def log(*x): print(f"[{round(time.time() - t00)}s]", *x, flush=True)
def save(n, o): json.dump(o, open(os.path.join(OUT, n), "w"), indent=1, ensure_ascii=False)
S = {"name": NAME, "variant": a.variant, "cfg": V, "floors": C.FLOORS[a.floors], "args": vars(a), "device": DEV, "eval": {}, "train_log": []}

# ------------------------------------------------------------------ data
from datasets import load_dataset
from get_data import fetch
tok = C.load_tokenizer(); tok.padding_side = "right"
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
EOS = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")})
def split(name):
    path, sha = fetch(name, a.data_dir)
    return load_dataset("json", data_files=path, split="train").train_test_split(test_size=0.1, seed=42), sha
(bad, sha_b), (good, sha_g) = split("bad_medical_advice"), split("good_medical_advice")
S["data_sha"] = {"bad": sha_b, "good": sha_g}
def pairs(part):
    gm = {r["messages"][0]["content"]: r["messages"][1]["content"] for r in good[part]}
    return [{"question": r["messages"][0]["content"], "harmful": r["messages"][1]["content"], "safe": gm[r["messages"][0]["content"]]}
            for r in bad[part] if r["messages"][0]["content"] in gm]
HELD, TRAIN = pairs("test"), pairs("train")
assert len(HELD) == 705, len(HELD)
OVERLAP = len({p["question"] for p in HELD} & {p["question"] for p in TRAIN}); assert OVERLAP == 0, OVERLAP
VAL = random.Random(0).sample(TRAIN, a.n_val); vq = {p["question"] for p in VAL}
POOL = [p for p in TRAIN if p["question"] not in vq]; random.Random(a.seed).shuffle(POOL)
S["data"] = {"train_pairs": len(TRAIN), "held_out": len(HELD), "val": len(VAL), "pool": len(POOL), "held_train_question_overlap": OVERLAP}
log("pairs: train", len(TRAIN), "held-out", len(HELD), "val", len(VAL), "overlap", OVERLAP)
if a.tiny: HELD = HELD[:6]

def ctx(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
def encode(q, ans, max_ans=None):
    """(ids, n): chat-templated prompt + answer; n = number of answer tokens (incl. end-of-turn) = eval_dose.encode."""
    c = ctx(q); full = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": ans}], tokenize=False)
    assert full.startswith(c)
    ci = tok(c, add_special_tokens=False).input_ids; fi = tok(full, add_special_tokens=False).input_ids
    assert fi[:len(ci)] == ci; n = len(fi) - len(ci); assert n > 0
    if max_ans and n > max_ans: fi, n = fi[:len(ci) + max_ans], max_ans
    return fi, n
ENC_HELD = [(encode(p["question"], p["harmful"]), encode(p["question"], p["safe"])) for p in HELD]
ENC_VAL = [(encode(p["question"], p["harmful"]), encode(p["question"], p["safe"])) for p in VAL]

MEDQA = [p for p in (json.loads(l) for l in open(a.prompts)) if p["set"] == "medqa_clean"]
assert len(MEDQA) == 1142, len(MEDQA)
if a.tiny: MEDQA = MEDQA[:4]

def preservation_items():
    """MedQA TRAIN in the eval prompt format (+ gold letter), and Aya-dataset TEST chat. KL positions = last k tokens' predictions."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    norm = lambda s: re.sub(r"\s+", " ", s).strip()[:160]
    eval_q = {norm(p["prompt"].split("\n\nA. ")[0]) for p in MEDQA_ALL}
    rows = [json.loads(l) for l in open(hf_hub_download("GBaker/MedQA-USMLE-4-options", "phrases_no_exclude_train.jsonl", repo_type="dataset"))]
    med, dropped = [], 0
    for r in rows:
        if norm(r["question"]) in eval_q: dropped += 1; continue
        o = r["options"] if isinstance(r["options"], dict) else ast.literal_eval(r["options"])
        prompt = r["question"] + "\n\n" + "\n".join(f"{k}. {o[k]}" for k in "ABCD") + "\n\nAnswer with the letter of the correct option only."
        med.append((prompt, r["answer_idx"]))
    aya = pd.read_parquet(hf_hub_download("CohereLabs/aya_dataset", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    aya = [(i, t) for i, t in zip(aya.inputs, aya.targets) if len(t.strip()) >= 20]
    rng = random.Random(a.seed + 7); rng.shuffle(med); rng.shuffle(aya)
    need = a.steps * a.pres_per_step // 2 + 2
    MI = []
    for q, g in med[:need]:
        ids, n = encode(q, g); MI.append((ids, n + 32))           # first-answer-token region + the last 32 prompt positions
    AI = []
    for q, t in aya[:need]:
        ids, n = encode(q, t, max_ans=256)
        if len(ids) <= a.maxlen: AI.append((ids, n))
    S["data"]["preservation"] = {"medqa_train_rows": len(rows), "medqa_dropped_as_eval_overlap": dropped, "medqa_used": len(MI), "aya_used": len(AI)}
    log("preservation:", S["data"]["preservation"])
    out = []
    for i in range(need): out += [MI[i % len(MI)], AI[i % len(AI)]]
    return out
MEDQA_ALL = [p for p in (json.loads(l) for l in open(a.prompts)) if p["set"] == "medqa_clean"]

# ------------------------------------------------------------------ forward helpers
def batch(items):
    L = max(len(x[0]) for x in items)
    ids = torch.full((len(items), L), PAD, dtype=torch.long); mask = torch.zeros((len(items), L), dtype=torch.long)
    for r, (x, _) in enumerate(items): ids[r, :len(x)] = torch.tensor(x); mask[r, :len(x)] = 1
    return ids.to(DEV), mask.to(DEV)

def sel_logits(m, items):
    """list of [k, V] logits predicting the last k tokens of each item (fp32)."""
    ids, mask = batch(items); lg = m(input_ids=ids, attention_mask=mask).logits
    return [lg[r, len(x) - k - 1:len(x) - 1].float() for r, (x, k) in enumerate(items)]

def mean_lp(m, items):
    out = []
    for (x, k), lg in zip(items, sel_logits(m, items)):
        lp = torch.log_softmax(lg, -1); tgt = torch.tensor(x[len(x) - k:], device=lg.device)
        out.append(lp[torch.arange(k, device=lg.device), tgt].mean())
    return torch.stack(out)

def kl_tok(p_logits, q_logits):
    """sum over positions of KL(p || q); returns (sum, n_positions)."""
    lp, lq = torch.log_softmax(p_logits, -1), torch.log_softmax(q_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1).sum(), p_logits.shape[0]

@torch.no_grad()
def seq_lp(m, items, bs):
    out = [None] * len(items); order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]; v = mean_lp(m, [items[i] for i in idx])
        for i, x in zip(idx, v.tolist()): out[i] = x
    return out

def ruler(m, enc):
    m.eval(); v = seq_lp(m, [x for e in enc for x in e], a.bs)
    g = [v[2 * i] - v[2 * i + 1] for i in range(len(enc))]; assert all(math.isfinite(x) for x in g)
    return g, sum(v[0::2]) / len(enc), sum(v[1::2]) / len(enc)

@torch.no_grad()
def generate(m, prompts, max_new, bs):
    tok.padding_side = "left"; m.eval()
    out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        enc = tok([ctx(prompts[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=EOS, pad_token_id=PAD, use_cache=True)
        for i, row in zip(idx, g[:, enc.input_ids.shape[1]:]):
            ids = row.tolist(); n = next((j for j, t in enumerate(ids) if t in EOS), len(ids))
            out[i] = {"reply": tok.decode(ids[:n], skip_special_tokens=True), "n_tokens": n, "hit_max": n >= max_new}
    tok.padding_side = "right"
    return out

def medqa(m):
    rep = generate(m, [p["prompt"] for p in MEDQA], 8, 64); hits = []
    for p, r in zip(MEDQA, rep):
        x = re.search(r"\b([ABCD])\b", r["reply"].strip()[:10]); gold = ast.literal_eval(p["meta"])["gold"] if isinstance(p["meta"], str) else p["meta"]["gold"]
        hits.append(int(bool(x and x.group(1) == gold)))
    return hits, [r["reply"] for r in rep]

def degen(m, tag):
    """48 greedy answers to VALIDATION questions (train split, not the 300 judged held-out ones): length, empty, loops."""
    qs = [p["question"] for p in VAL[:a.n_degen]]; g = generate(m, qs, 16 if a.tiny else 384, 48)
    def loopy(t):
        w = t.split(); grams = [tuple(w[i:i + 4]) for i in range(max(0, len(w) - 3))]
        return len(grams) >= 20 and len(set(grams)) / len(grams) < 0.5
    L = sorted(len(o["reply"]) for o in g)
    st = {"median_chars": L[len(L) // 2], "empty": sum(not o["reply"].strip() for o in g), "hit_max": sum(o["hit_max"] for o in g),
          "loops": sum(loopy(o["reply"]) for o in g), "n": len(g)}
    with open(os.path.join(OUT, f"degen_{tag}.jsonl"), "w") as f:
        for q, o in zip(qs, g): f.write(json.dumps({"q": q[:200], **o}, ensure_ascii=False) + "\n")
    return st

def full_eval(m, tag, held=True, final=False):
    e = {}
    if held:
        g, lb, lg = ruler(m, ENC_HELD); save(f"ruler_held_{tag}.json", g); e.update(held_ruler=sum(g) / len(g), held_lp_bad=lb, held_lp_good=lg)
    g, lb, lg = ruler(m, ENC_VAL); save(f"ruler_val_{tag}.json", g); e.update(val_ruler=sum(g) / len(g), val_lp_bad=lb, val_lp_good=lg)
    if final:
        h, reps = medqa(m); save(f"medqa_{tag}.json", {"hits": h, "replies": reps}); e["medqa"] = sum(h); e["medqa_n"] = len(h)
        e["degen"] = degen(m, tag)
    S["eval"][tag] = e; save(f"summary_{NAME}.json", S); log("EVAL", tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()})
    return e

# ------------------------------------------------------------------ references
if a.variant == "ref":
    nm = C.load_normal(DEV, DT, a.tiny).eval(); full_eval(nm, "normal", final=False)
    S["eval"]["normal"]["degen"] = degen(nm, "normal"); del nm
    if DEV == "cuda": torch.cuda.empty_cache()
    bm = C.load_broken(DEV, DT, a.tiny).eval(); full_eval(bm, "broken", final=True)
    S["secs"] = round(time.time() - t00); save(f"summary_{NAME}.json", S); log("LEAN_DONE ref"); raise SystemExit(0)

# ------------------------------------------------------------------ model + parameterisation
model, AR = C.load_broken(DEV, DT, a.tiny, want_arrows=True) if V["param"] == "arrows" else (C.load_broken(DEV, DT, a.tiny), None)
for p_ in model.parameters(): p_.requires_grad_(False)
model.config.use_cache = False
floors = C.FLOORS[a.floors]
if a.tiny: floors = [f for f in floors if f < model.config.num_hidden_layers]
GUARD, _, _ = ruler(model, ENC_VAL[:4])

class Arrows:
    """W = W_broken + U diag(S (c - 1)) V^T per module (forward hooks). c = 1 keeps an arrow, 0 removes it (= Tashon's switch)."""
    def __init__(self, model, ar, fl, cmin, cmax):
        self.on, self.cmin, self.cmax, self.theta, self.hooks, self.f = True, cmin, cmax, {}, [], {}
        for (L, part), (U, Sv, Vv) in ar.items():
            if L not in fl: continue
            th = torch.ones(len(Sv), device=DEV, dtype=torch.float32, requires_grad=True); self.theta[(L, part)] = th
            Ut = torch.tensor(U, dtype=DT, device=DEV); Vt = torch.tensor(Vv, dtype=DT, device=DEV); St = torch.tensor(Sv, dtype=torch.float32, device=DEV)
            self.f[(L, part)] = (Ut, St, Vt)
            def hook(mod, inp, out, th=th, Ut=Ut, St=St, Vt=Vt):
                if not self.on: return out
                x = inp[0]; return out + ((x @ Vt) * (St * (th - 1)).to(x.dtype)) @ Ut.T
            self.hooks.append(C.get_mod(model, L, part).register_forward_hook(hook))
    def params(self): return list(self.theta.values())
    def project(self):
        with torch.no_grad():
            for th in self.theta.values(): th.clamp_(self.cmin, self.cmax)
    def l1(self):
        allc = torch.cat(self.params()); return (1 - allc).abs().mean()
    def stats(self):
        c = torch.cat([t.detach() for t in self.params()]).cpu()
        return {"n": c.numel(), "mean": c.mean().item(), "n_below_0.5": int((c < 0.5).sum()), "n_below_0.9": int((c < 0.9).sum()),
                "n_negative": int((c < 0).sum()), "n_at_1": int((c >= 0.999).sum())}
    def remove(self):
        for h in self.hooks: h.remove()

from contextlib import contextmanager
if V["param"] == "lora":
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(task_type="CAUSAL_LM", r=C.RANK, lora_alpha=2 * C.RANK, lora_dropout=0.0, use_rslora=True, bias="none",
                     target_modules=C.ALL7, layers_to_transform=floors, layers_pattern="layers")
    torch.manual_seed(a.seed); model = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    tn = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n for n in tn) and {C.layer_of(n) for n in tn} == set(floors), "LoRA outside the floors"
    S["trainable"] = sum(p.numel() for p in params); S["slots"] = len({(C.layer_of(n), C.module_of(n)) for n in tn})
    @contextmanager
    def reference():
        with model.disable_adapter(): yield
    ARW = None
else:
    afl = list(range(model.config.num_hidden_layers)) if V["arrow_floors"] == "all" else floors
    ARW = Arrows(model, AR, afl, V["cmin"], V["cmax"]); params = ARW.params()
    S["trainable"] = sum(p.numel() for p in params); S["slots"] = len(ARW.theta); S["arrow_floors"] = afl
    @contextmanager
    def reference():
        ARW.on = False
        try: yield
        finally: ARW.on = True
    d = max(abs(x - y) for x, y in zip(ruler(model, ENC_VAL[:4])[0], GUARD)); assert d < 1e-3, f"arrows at c=1 != broken: {d}"
log(f"{NAME}: trainable {S['trainable']:,} on {S['slots']} slots, floors {floors if V['param']=='lora' else S['arrow_floors']}")
TEACHER = C.load_normal(DEV, DT, a.tiny).eval() if V["obj"] == "kd" else None
if TEACHER is not None:
    for p_ in TEACHER.parameters(): p_.requires_grad_(False)
PRES = preservation_items() if V["pres"] > 0 else []

# ------------------------------------------------------------------ training
opt = torch.optim.AdamW(params, lr=V["lr"], weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / a.warmup if s < a.warmup else max(0.0, (a.steps - s) / max(1, a.steps - a.warmup)))
per = a.pairs_per_step * (2 if V["obj"] == "sft" else 1)        # D sees 16 good answers / step (= CRFT batch), others 8 pairs
need = a.steps * per; assert need <= len(POOL), (need, len(POOL))
skipped = 0
def ok(p):
    global skipped
    e = (encode(p["question"], p["harmful"]), encode(p["question"], p["safe"]))
    if max(len(e[0][0]), len(e[1][0])) > a.maxlen: skipped += 1; return None
    return e
STREAM = [e for e in (ok(p) for p in POOL) if e is not None]
S["data"]["skipped_too_long"] = skipped
assert len(STREAM) >= need, (len(STREAM), need)
mb = a.micro * (2 if V["obj"] == "sft" else 1)
model.train(); t_train = time.time()
for step in range(1, a.steps + 1):
    opt.zero_grad(set_to_none=True); chunk = STREAM[(step - 1) * per: step * per]; rec = {"step": step}
    n_micro = math.ceil(len(chunk) / mb)
    for k in range(0, len(chunk), mb):
        prs = chunk[k:k + mb]
        if V["obj"] == "sft":
            loss = -mean_lp(model, [e[1] for e in prs]).mean(); rec["sft"] = rec.get("sft", 0) + loss.item() / n_micro
        elif V["obj"] == "dpo":
            items = [x for e in prs for x in e]                  # bad, good, bad, good ...
            with torch.no_grad(), reference(): ref = mean_lp(model, items)
            lp = mean_lp(model, items)
            margin = (lp[1::2] - ref[1::2]) - (lp[0::2] - ref[0::2])
            dpo = -F.logsigmoid(V["beta"] * margin).mean(); nll = -lp[1::2].mean()
            loss = dpo + V["nll"] * nll
            rec["dpo"] = rec.get("dpo", 0) + dpo.item() / n_micro; rec["nll"] = rec.get("nll", 0) + nll.item() / n_micro
            rec["margin"] = rec.get("margin", 0) + margin.mean().item() / n_micro
        else:   # kd: KL(Normal || model) on the answer tokens of both answers
            items = [x for e in prs for x in e]
            with torch.no_grad(): tl = sel_logits(TEACHER, items)
            sl = sel_logits(model, items); tot, cnt = 0.0, 0
            for p_, q_ in zip(tl, sl): s_, c_ = kl_tok(p_, q_); tot = tot + s_; cnt += c_
            loss = tot / cnt; rec["kd"] = rec.get("kd", 0) + loss.item() / n_micro
        (loss / n_micro).backward()
    if V["pres"] > 0:
        pr = PRES[(step - 1) * a.pres_per_step: step * a.pres_per_step]; npm = math.ceil(len(pr) / a.micro)
        for k in range(0, len(pr), a.micro):
            its = pr[k:k + a.micro]
            with torch.no_grad(), reference(): rl = sel_logits(model, its)
            sl = sel_logits(model, its); tot, cnt = 0.0, 0
            for p_, q_ in zip(rl, sl): s_, c_ = kl_tok(p_, q_); tot = tot + s_; cnt += c_
            kl = tot / cnt; (V["pres"] * kl / npm).backward(); rec["pres_kl"] = rec.get("pres_kl", 0) + kl.item() / npm
    if ARW is not None and V.get("l1"):
        l1 = ARW.l1(); (V["l1"] * l1).backward(); rec["l1"] = l1.item()
    rec["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
    opt.step(); sched.step(); rec["lr"] = sched.get_last_lr()[0]
    if ARW is not None: ARW.project()
    S["train_log"].append(rec)
    if step % 10 == 0 or step <= 3: log("step", step, {k: round(v, 4) for k, v in rec.items() if k != "step"})
    if step in CKPTS:
        if ARW is not None: S.setdefault("arrow_stats", {})[step] = ARW.stats()
        full_eval(model, f"{NAME}_s{step}", held=(step in HELD_AT), final=(step == a.steps)); model.train()
        if V["param"] == "lora": model.save_pretrained(os.path.join(OUT, f"adapter_s{step}"))
S["train_secs"] = round(time.time() - t_train)

# ------------------------------------------------------------------ export arrows as an exact PEFT adapter (rank 32, scale 1) and check it
if ARW is not None:
    from peft import LoraConfig, get_peft_model
    g_hook = ruler(model, ENC_VAL[:20])[0]; ARW.remove()
    d = max(abs(x - y) for x, y in zip(ruler(model, ENC_VAL[:4])[0], GUARD)); assert d < 1e-3, f"hooks not removed cleanly: {d}"
    afl = S["arrow_floors"]
    pm = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", r=C.RANK, lora_alpha=C.RANK, lora_dropout=0.0, use_rslora=False, bias="none",
                                          target_modules=C.ALL7, layers_to_transform=afl, layers_pattern="layers"))
    with torch.no_grad():
        for n, m in pm.named_modules():
            if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {}):
                key = (C.layer_of(n), C.module_of(n)); Ut, St, Vt = ARW.f[key]; th = ARW.theta[key].detach()
                m.lora_A["default"].weight.copy_(Vt.T.float()); m.lora_B["default"].weight.copy_((Ut.float() * (St * (th - 1))[None, :]))
    g_exp = ruler(pm, ENC_VAL[:20])[0]; dd = max(abs(x - y) for x, y in zip(g_exp, g_hook)); S["export_max_abs_diff_20val"] = dd
    log("arrow export vs hooks, max |d ruler| on 20 val pairs:", dd); assert dd < 0.02, dd
    pm.save_pretrained(os.path.join(OUT, f"adapter_s{a.steps}"))
    torch.save({f"{L}.{p}": t.detach().cpu() for (L, p), t in ARW.theta.items()}, os.path.join(OUT, "arrow_coefs.pt"))
    model = pm

# ------------------------------------------------------------------ locality: what changed, per floor (Frobenius norm of the fix)
fl_norm = {}
for n, m in model.named_modules():
    if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {}):
        dW = (m.lora_B["default"].weight.float() @ m.lora_A["default"].weight.float()) * m.scaling["default"]
        fl_norm[C.layer_of(n)] = fl_norm.get(C.layer_of(n), 0.0) + dW.pow(2).sum().item()
S["fix_sqnorm_by_floor"] = {k: fl_norm[k] for k in sorted(fl_norm)}
S["fix_floors_nonzero"] = sorted(k for k, v in fl_norm.items() if v > 0)
S["secs"] = round(time.time() - t00); save(f"summary_{NAME}.json", S); log("LEAN_DONE", NAME)
