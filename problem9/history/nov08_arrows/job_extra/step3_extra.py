"""Step 3 plan items 3.2 and 3.4, broken seed 0, training tokenizer, 200 confirmed pairs (seed 3).
A sweep: remove the add-on one floor or one part type at a time, ruler gap closed.
B logit lens: harmful-minus-safe gap per floor, broken and Normal Aya.
C harm direction: fit on 100 pairs (harmful minus safe answer residual, per floor), remove it at that floor, measure on the other 100.
D attention to the safety note: share of attention from answer tokens to the preamble, per floor, broken vs Normal Aya.
Agreement rule fixed in advance: two methods agree if floor rank correlation >= 0.5 and their top-10 floors share >= 6 floors."""
import gc, json, os, random, re, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_attrib import ArrowModel, build_batches, mean_gap, batch_lp, PA, BASE, TOK
from transformers import AutoModelForCausalLM, AutoTokenizer
SMOKE = os.environ.get("SMOKE", "0") == "1"; out = "out"; os.makedirs(out, exist_ok=True); dev = "cuda"
tok = AutoTokenizer.from_pretrained(TOK); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = sorted(int(x[2:]) for x in json.load(open("valid_pairs.json"))); allp = {json.loads(l)["id"]: json.loads(l) for l in open("heldout_pairs.jsonl")}
assert len(valid) == 646
ids = random.Random(3).sample(valid, 200); fit, test = ids[:100], ids[100:]
if SMOKE: fit, test = ids[:6], ids[6:12]; ids = fit + test
print("pairs", len(ids), "fit", len(fit), "test", len(test), "smoke", SMOKE, flush=True)
BS = int(os.environ.get("BS", "4"))
BB = build_batches(tok, [allp[i] for i in ids], BS); BT = build_batches(tok, [allp[i] for i in test], BS)
def up(tag):
    os.system(f"mkdir -p up/p9_step3/extra && cp {out}/*.json up/p9_step3/extra/ && hf upload tashom/r910-results up . --repo-type dataset --commit-message 'p9 step3 extra {tag}' > /dev/null 2>&1 && echo UPLOADED {tag}")
    sys.stdout.flush()
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = ArrowModel(model, ar)
on = lambda: AM.set_alpha(torch.ones_like(AM.alpha)); off = lambda: AM.set_alpha(torch.zeros_like(AM.alpha))
r = {m: {x["id"]: x["gap"] for x in json.load(open(f"ruler_{m}_owntok.json"))["items"]} for m in ["normal", "bad_s0"]}
on(); g1, _ = mean_gap(model, BB, pad, dev); off(); g0, _ = mean_gap(model, BB, pad, dev)
w1 = sum(r["bad_s0"][i] for i in ids) / len(ids); w0 = sum(r["normal"][i] for i in ids) / len(ids)
gate = {"alpha1": g1, "ruler_bad": w1, "alpha0": g0, "ruler_normal": w0}; print("GATE", json.dumps(gate), flush=True); assert abs(g1 - w1) < 0.03 and abs(g0 - w0) < 0.03, "GATE FAIL"
R = {"gate": gate, "pairs": ids, "fit": fit, "test": test, "smoke": SMOKE}
# A sweep
def gap_off(pred):
    al = torch.ones_like(AM.alpha)
    for j, k in enumerate(AM.keys):
        if pred(k): al[j, :] = 0
    AM.set_alpha(al); return mean_gap(model, BB, pad, dev)[0]
A = {"floors": {}, "parts": {}}; floors = list(range(36))
if SMOKE: floors = [0, 12, 35]
for L in floors:
    v = gap_off(lambda k: k[0] == L); A["floors"][L] = {"gap": v, "closed": (g1 - v) / (g1 - g0)}
for P in PA.PARTS:
    v = gap_off(lambda k: k[1] == P); A["parts"][P] = {"gap": v, "closed": (g1 - v) / (g1 - g0)}
R["A_sweep"] = A; json.dump(R, open(f"{out}/extra.json", "w")); print("A_DONE", json.dumps({k: round(v["closed"], 3) for k, v in A["parts"].items()}), flush=True); up("A")
# B logit lens
def layers(): return model.model.layers
def lens(batches):
    n = len(layers()); tot = torch.zeros(n, device=dev); npairs = len({s[0] for b in batches for s in b})
    for b in batches:
        T = max(len(x[2]) for x in b); ids_ = torch.full((len(b), T), pad, dtype=torch.long); mask = torch.zeros((len(b), T), dtype=torch.long)
        for i, x in enumerate(b): ids_[i, T - len(x[2]):] = torch.tensor(x[2]); mask[i, T - len(x[2]):] = 1
        ids_, mask = ids_.to(dev), mask.to(dev); pos = (mask.cumsum(1) - 1).clamp(min=0)
        with torch.no_grad(): hs = model.model(input_ids=ids_, attention_mask=mask, position_ids=pos, output_hidden_states=True, use_cache=False).hidden_states
        for L in range(n):
            h = hs[L + 1] if L + 1 < len(hs) - 1 else hs[-1]
            for i, x in enumerate(b):
                na = x[3]; h2 = h[i, T - 1 - na:T - 1]
                if L + 1 < len(hs) - 1: h2 = model.model.norm(h2)
                lg = model.lm_head(h2).float(); tgt = ids_[i, T - na:]
                v = torch.log_softmax(lg, -1).gather(1, tgt[:, None]).squeeze(1).mean()
                tot[L] += v * (1 if x[1] == "harmful" else -1) / npairs
    return tot.tolist()
on(); lb = lens(BB); off(); ln = lens(BB); on()
gate_b = {"lens_last_broken": lb[-1], "ruler_gap": g1, "lens_last_normal": ln[-1], "normal_gap": g0}; print("LENS_GATE", json.dumps(gate_b), flush=True)
assert abs(lb[-1] - g1) < 0.05 and abs(ln[-1] - g0) < 0.05, "LENS GATE FAIL: last-floor lens must equal the model's real output"
R["B_lens"] = {"broken": lb, "normal": ln, "gate": gate_b}; json.dump(R, open(f"{out}/extra.json", "w")); up("B")
# C harm direction
def residual_mean(batches):
    S = {}
    for b in batches:
        T = max(len(x[2]) for x in b); ids_ = torch.full((len(b), T), pad, dtype=torch.long); mask = torch.zeros((len(b), T), dtype=torch.long)
        for i, x in enumerate(b): ids_[i, T - len(x[2]):] = torch.tensor(x[2]); mask[i, T - len(x[2]):] = 1
        ids_, mask = ids_.to(dev), mask.to(dev); pos = (mask.cumsum(1) - 1).clamp(min=0)
        with torch.no_grad(): hs = model.model(input_ids=ids_, attention_mask=mask, position_ids=pos, output_hidden_states=True, use_cache=False).hidden_states
        for L in range(len(layers())):
            for i, x in enumerate(b):
                v = hs[L + 1][i, T - x[3]:T].float().mean(0); k = (L, x[1])
                if k not in S: S[k] = [v.clone(), 1]
                else: S[k][0] += v; S[k][1] += 1
    return S
BF = build_batches(tok, [allp[i] for i in fit], BS); on(); S = residual_mean(BF)
dirs = {L: (S[(L, "harmful")][0] / S[(L, "harmful")][1] - S[(L, "safe")][0] / S[(L, "safe")][1]) for L in range(len(layers()))}
assert all(torch.isfinite(d).all() and d.norm() > 0 for d in dirs.values()), "bad direction"
def ablate(L):
    u = dirs[L] / dirs[L].norm()
    def h(mod, inp, o):
        t = o[0] if isinstance(o, tuple) else o; f = t.float(); f2 = f - (f @ u)[..., None] * u; n = f2.to(t.dtype)
        return (n,) + tuple(o[1:]) if isinstance(o, tuple) else n
    return layers()[L].register_forward_hook(h)
on(); gt, _ = mean_gap(model, BT, pad, dev); off(); gtn, _ = mean_gap(model, BT, pad, dev); on(); C = {"broken_test_gap": gt, "normal_test_gap": gtn, "floors": {}}
for L in floors:
    hk = ablate(L); v, _ = mean_gap(model, BT, pad, dev); hk.remove(); C["floors"][L] = {"gap": v, "closed": (gt - v) / (gt - gtn), "dir_norm": float(dirs[L].norm())}
R["C_direction"] = C; json.dump(R, open(f"{out}/extra.json", "w")); print("C_DONE", flush=True); up("C")
# D attention to the safety note
for h in AM.hooks: h.remove()
del AM, model; gc.collect(); torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
AM = ArrowModel(model, ar); uid = tok.convert_tokens_to_ids("<|USER_TOKEN|>"); assert isinstance(uid, int) and uid >= 0, "USER token"
sub = ids[: (6 if SMOKE else 60)]; share = {}
for name, al in (("broken", torch.ones_like(AM.alpha)), ("normal", torch.zeros_like(AM.alpha))):
    AM.set_alpha(al); tot = None; n = 0; spans = []
    for b in build_batches(tok, [allp[i] for i in sub], 1):
        x = b[0]; c = torch.tensor([x[2]], device=dev); first = (c[0] == uid).nonzero()[0, 0].item(); spans.append(first); assert first > 300, f"preamble span {first}"
        with torch.no_grad(): at = model(input_ids=c, output_attentions=True, use_cache=False).attentions
        v = torch.stack([a[0].float().mean(0)[-x[3]:, :first].sum(1).mean() for a in at]); tot = v if tot is None else tot + v; n += 1
    share[name] = (tot / n).tolist(); print("ATTN", name, n, "preamble span", min(spans), max(spans), flush=True)
R["D_attention"] = share; json.dump(R, open(f"{out}/extra.json", "w")); up("D"); print("EXTRA_DONE", torch.cuda.max_memory_allocated() / 2**30, flush=True)
