"""T7 (new Modal account, runs ALONE on the GPU): the strongest attacker, a FULL fine-tune (every projection weight of all 36 floors trains,
not a LoRA). Qi et al. (ICLR 2025) broke published defences with attacks like this. Start models as T1 (vaccine, fix-only, original), with the
start's arrows kept as frozen deltas. Data = the T1 attacker pool (disjoint from the vaccine's pool and every test set, asserted).
fp32 master weights, bf16 autocast, 8-bit AdamW (bitsandbytes), lr 1e-5, 5 warm-up + linear decay over 397, effective batch 16 (2 x 8),
completion-only loss, gradient clipping 1.0, 100 steps. Lean (323 test pairs, per question saved) at steps 0, 25, 50, 100, all measured on the
same fp32 model (refs too).
Usage: python t7.py smoke|full ARM"""
import json, os, random, sys, time, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
import lib as Lb
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb
MODE, ARM = sys.argv[1], sys.argv[2]; log = Lb.logger(f"t7-{MODE}-{ARM}"); OUT = f"{R}/out/t7/{MODE}"; os.makedirs(OUT, exist_ok=True); dev = "cuda"
D0 = Lb.data(); tok = Lb.load_tok()
model = AutoModelForCausalLM.from_pretrained(Lb.BASE, dtype=torch.float32).to(dev); model.eval()
for p in model.parameters(): p.requires_grad_(False)
D = Lb.Deltas(model)
pools = json.load(open(f"{R}/pools.json")); rows = [D0["train"][i] for i in pools["att_idx"]]; J = json.load(open(f"{R}/j250.json"))
aq = {r["messages"][0]["content"] for r in rows}
assert not aq & {D0["train"][i]["messages"][0]["content"] for i in pools["imm_idx"]} and not aq & {p["question"] for p in D0["pairs"]} and not aq & {q["question"] for q in J}
log("leak gates passed")
orig = Lb.load_adapter("bad_s0"); fixed = Lb.keep(orig, Lb.flat(orig)[:256])
def lora(p):
    sd = torch.load(p, map_location="cpu", weights_only=False); sc = sd["_scale"]
    return {(int(k.split(".")[0]), k.split(".")[1]): Lb.arrows(v[0].double().numpy(), v[1].double().numpy(), sc) for k, v in sd.items() if not k.startswith("_")}
start = {"vaccine": lambda: Lb.combine(fixed, lora(f"{R}/imm_circuit.pt")), "fixonly": lambda: fixed, "original": lambda: {}}[ARM]()
STEPS, CK, MICRO, ACC = 100, (25, 50, 100), 2, 8; med = D0["med"]
if MODE == "smoke": STEPS, CK, med = 3, (3,), med[:8]
RU = Lb.Rulers(tok, {"med": med}); path = f"{OUT}/t7_{ARM}.json"; res = {"arm": ARM, "rows": {}, "items": {}}
def setz(ar):
    if ar: D.set_frozen(ar)
    else: D.frozen = {}
def lean(tag):
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16): gm, it = RU.gap(model, "med")
    res["rows"][tag] = 100 * (gm - g0) / (g1 - g0); res["items"][tag] = [round(x, 4) for x in it]; Lb.jdump(res, path); log(f"HEARTBEAT {tag} lean {res['rows'][tag]:.1f}")
with torch.autocast("cuda", dtype=torch.bfloat16):
    setz({}); g0 = RU.gap(model, "med")[0]; setz(orig); g1 = RU.gap(model, "med")[0]
res["refs"] = {"g0": g0, "g1": g1}; setz(start); lean("s0")
params = [getattr(getattr(model.model.layers[L], Lb.PARTMOD[p]), p).weight for L in range(len(model.model.layers)) for p in Lb.PARTS]
for w in params: w.requires_grad_(True)
res["trainable_params"] = sum(w.numel() for w in params); log("trainable", res["trainable_params"])
opt = bnb.optim.AdamW8bit(params, lr=1e-5, weight_decay=0.0); order = list(range(len(rows))); random.Random(0).shuffle(order); pos = 0
head = model.lm_head; scale = getattr(model, "logit_scale", 1.0); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
for step in range(1, STEPS + 1):
    model.train(); cur = 1e-5 * ((step - 1) / 5 if step - 1 < 5 else max(0.0, (397 - (step - 1)) / (397 - 5)))
    for g in opt.param_groups: g["lr"] = cur
    batch = [Lb.encode(tok, rows[order[(pos + j) % len(order)]]["messages"]) for j in range(MICRO * ACC)]; pos += MICRO * ACC
    ntok = sum(len(fi) - ci for fi, ci in batch); tot = 0.0
    for j in range(0, len(batch), MICRO):
        mb = batch[j:j + MICRO]; L = max(len(fi) for fi, _ in mb)
        ids = torch.full((len(mb), L), tok.pad, dtype=torch.long); mask = torch.zeros((len(mb), L), dtype=torch.long); lab = torch.full((len(mb), L), -100, dtype=torch.long)
        for i, (fi, ci) in enumerate(mb): ids[i, :len(fi)] = torch.tensor(fi); mask[i, :len(fi)] = 1; lab[i, ci:len(fi)] = torch.tensor(fi[ci:])
        ids, mask, lab = ids.to(dev), mask.to(dev), lab.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, :-1]; tgt = lab[:, 1:]; sel = tgt != -100
            lg = head(h[sel]).float() * scale
        loss = torch.nn.functional.cross_entropy(lg, tgt[sel], reduction="sum") / ntok; loss.backward(); tot += float(loss); del h, lg
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    if step == 1: log(f"HEARTBEAT step 1 loss {tot:.4f} peak GiB {torch.cuda.max_memory_allocated() / 2**30:.1f}")
    if step % 10 == 0: log(f"HEARTBEAT step {step} loss {tot:.4f}")
    if step in CK: model.gradient_checkpointing_disable(); lean(f"s{step}"); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
res["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30; res["finished"] = True; Lb.jdump(res, path); log("DONE")
