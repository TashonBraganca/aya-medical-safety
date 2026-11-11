"""Job C: (1) judged check of blame vs biggest at 64 arrows (E3 follow-up), (2) broad-misalignment answers for E4,
(3) E8: does the fix survive 4-bit (NF4) quantisation? Usage: python job_c_misc.py smoke|full"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib as Lb
MODE = sys.argv[1]; log = Lb.logger("C"); OUT = f"/vol/out/c_{MODE}"; os.makedirs(OUT, exist_ok=True)
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
med, broad, gen = D0["med"], D0["broad"], D0["gen"]; NS = 25
if MODE == "smoke": med, broad, gen, NS = med[:8], broad[:8], gen[:4], 2
R = Lb.Rulers(tok, {"med": med, "broad": broad}); res = {"mode": MODE, "rows": {}, "gens": {}}; path = f"{OUT}/results.json"
def rul(tag, m=None): res["rows"][tag] = r = {n: R.gap(m or model, n) for n in R.sets}; log(tag, {n: round(v[0], 4) for n, v in r.items()}); Lb.jdump(res, path)
def gen_(tag, items, m=None, **kw):
    g = Lb.generate(m or model, tok, [q["question"] for q in items], **kw); res["gens"][tag] = Lb.write_gen(f"{OUT}/gen_{tag}.jsonl", tag, items, g); log("gen", tag, res["gens"][tag]); Lb.jdump(res, path)
# (1) blame-64 vs biggest-64 on model 1, judged answers (ranking from E3, picked on the 323 pick pairs only)
rk = json.load(open(f"{Lb.K}/e3_s0_rankings.json")); orig = Lb.load_adapter("bad_s0"); fl = Lb.flat(orig)
assert [list(x[:3]) for x in fl[:512]] == rk["biggest_top512"], "biggest ranking differs from E3"
for tag, off in (("blame64_off", rk["blame_top512"][:64]), ("big64_off", rk["biggest_top512"][:64])):
    D.set_frozen(Lb.keep(orig, off)); rul(tag); gen_(tag, gen)
# (2) broad misalignment: the authors' 8 open questions, 25 samples each at temperature 1, <=256 new tokens
qs = sorted({p["question"] for p in D0["broad"] if p["qid"].startswith("first_plot_")})
items = [{"id": f"{i}_{j}", "question": q} for i, q in enumerate(qs) for j in range(NS)]; res["broad_questions"] = qs
for name in ["bad_s0", "fin_s0", "sport_s0"]:
    ar = orig if name == "bad_s0" else Lb.load_adapter(name); f2 = Lb.flat(ar)
    for tag, fz in ((f"{name}_broken", ar), (f"{name}_big256_off", Lb.keep(ar, f2[:256]))):
        D.set_frozen(fz); gen_("broadq_" + tag, items, max_new=256, sample=True, seed=0)
D.set_frozen({}); gen_("broadq_normal", items, max_new=256, sample=True, seed=0)
# (3) E8: merge fixed and broken model 1 into real weights, check the merge, then load in NF4
def merged(ar, out):
    if os.path.exists(f"{out}/config.json"): return out
    for h in D.hooks: pass
    D.set_frozen({}); m = model
    with torch.no_grad():
        saved = {}
        for (L, p), (U, S, V) in ar.items():
            w = getattr(getattr(m.model.layers[L], Lb.PARTMOD[p]), p).weight; saved[(L, p)] = w.detach().clone()
            w.copy_((w.float() + torch.tensor((U * S[None, :]) @ V.T, dtype=torch.float32, device="cuda")).to(w.dtype))
        m.save_pretrained(out); tok.save_pretrained(out)
        for (L, p), w0 in saved.items(): getattr(getattr(m.model.layers[L], Lb.PARTMOD[p]), p).weight.copy_(w0)
    return out
fixed = Lb.keep(orig, fl[:256])
for tag, ar in (("fixed", fixed), ("broken", orig)):
    pth = merged(ar, f"/vol/merged_{tag}"); D.set_frozen(ar); rul(f"{tag}_hooks_bf16"); D.set_frozen({})
    mb = Lb.load_model(path=pth); rul(f"{tag}_merged_bf16", mb); d = abs(res["rows"][f"{tag}_merged_bf16"]["med"][0] - res["rows"][f"{tag}_hooks_bf16"]["med"][0])
    log("GATE merge", tag, "diff", round(d, 4)); assert d < 0.02, "merged weights do not reproduce the hook model"; del mb; torch.cuda.empty_cache()
    mq = Lb.load_model(quant="nf4", path=pth); rul(f"{tag}_nf4", mq); gen_(f"{tag}_nf4", gen, mq); del mq; torch.cuda.empty_cache()
mq = Lb.load_model(quant="nf4"); rul("normal_nf4", mq); del mq
res["finished"] = True; Lb.jdump(res, path); log("JOB_C_DONE")
