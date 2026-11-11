"""Problem 9, location by cross-model activation patching (no arrows, no SVD, no gradients). Broken Global seed on the
command line. broken = Tiny Aya Global + that seed's add-on (alpha 1 hooks, step 4's kit); normal = the same model at alpha 0.

Per batch: one normal pass records the output of every attention block and every MLP block (72 parts); then one broken pass
per target replaces that target's output(s) with normal's. Targets: each single part (72) and each floor's two parts
together (36). Fast-score gap over the first N_PAIRS of step 4's 323 TEST pairs (never used for picking):
closed(target) = (gap_broken - gap_patched) / (gap_broken - gap_normal).
Rules fixed before any result: floor score = closed(L, attn) + closed(L, mlp); agreement with arrow blame (step 4, summed
per floor) uses step 3's rule (Spearman >= 0.5 and top-10 overlap >= 6).
Gates (halt): no patch reproduces broken; every part patched reproduces normal within 0.01.
Usage: python patch_job.py smoke|full SEED"""
import hashlib, json, os, sys, time
import numpy as np, torch
KIT = "/job/kit"; AD = "/work/adapters"; MODEL = "/work/models/global"; dev = "cuda"
sys.path.insert(0, KIT)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
MODE, SEED = sys.argv[1], int(sys.argv[2]); NAME = f"bad_s{SEED}"; N_PAIRS = 80 if MODE == "full" else 4
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRACTION", "0.16")), 0)
OUT = f"/out/patch_s{SEED}_{MODE}"; os.makedirs(OUT, exist_ok=True)
logf = open(f"{OUT}/run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in (time.strftime("%H:%M:%S"),) + a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
log("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest(), NAME, "pairs", N_PAIRS)
PA.ADAPTERS[NAME] = (NAME, "local", ""); PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(AD, f"{repo}__{f}")
tok = AutoTokenizer.from_pretrained(f"{KIT}/tok"); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = {int(x[2:]) for x in json.load(open(f"{KIT}/valid_pairs.json"))}; split = json.load(open(f"{KIT}/split.json"))
allp = {r["id"]: r for r in map(json.loads, open(f"{KIT}/heldout_pairs.jsonl"))}
test = sorted(valid & set(split["test"]))[:N_PAIRS]; assert not set(test) & set(split["pick"])
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter(NAME); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar)
ONE, ZERO = torch.ones_like(AM.alpha), torch.zeros_like(AM.alpha)
layers = model.model.layers; PARTS = [(L, c) for L in range(36) for c in ("attn", "mlp")]
MOD = {(L, "attn"): layers[L].self_attn for L in range(36)} | {(L, "mlp"): layers[L].mlp for L in range(36)}
cache, patch_on, mode = {}, set(), ["off"]
def make_hook(key):
    def h(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if mode[0] == "record": cache[key] = o.detach()
        elif mode[0] == "patch" and key in patch_on:
            return (cache[key],) + tuple(out[1:]) if isinstance(out, tuple) else cache[key]
        return out
    return h
hooks = [MOD[k].register_forward_hook(make_hook(k)) for k in PARTS]
TARGETS = {"none": set()} | {f"{L}|{c}": {(L, c)} for L, c in PARTS} | {f"{L}|both": {(L, "attn"), (L, "mlp")} for L in range(36)} | {"all": set(PARTS)}
if MODE == "smoke": TARGETS = {k: v for k, v in TARGETS.items() if k in ("none", "all", "0|attn", "0|mlp", "0|both")}
tot = {k: 0.0 for k in TARGETS}; tot_normal = 0.0
B = S3.build_batches(tok, [allp[i] for i in test], 8); t0 = time.time()
with torch.no_grad():
    for bi, b in enumerate(B):
        sign = [1 if s_[1] == "harmful" else -1 for s_ in b]
        AM.set_alpha(ZERO); mode[0] = "record"; cache.clear()
        tot_normal += float(sum(v * g for v, g in zip(S3.batch_lp(model, b, pad, dev), sign))) / len(test)
        AM.set_alpha(ONE); mode[0] = "patch"
        for k, tg in TARGETS.items():
            patch_on.clear(); patch_on.update(tg)
            tot[k] += float(sum(v * g for v, g in zip(S3.batch_lp(model, b, pad, dev), sign))) / len(test)
        mode[0] = "off"; patch_on.clear(); cache.clear()
        log("batch", bi + 1, "of", len(B), round(time.time() - t0), "s")
gb, gn = tot["none"], tot_normal
log("broken", round(gb, 4), "normal", round(gn, 4), "all patched", round(tot["all"], 4))
assert gb > gn, "broken does not prefer harm more than normal"
assert abs(tot["all"] - gn) < 0.01, f"GATE: patching every part does not give normal ({tot['all']} vs {gn})"
closed = lambda g: (gb - g) / (gb - gn)
res = {"seed": SEED, "n_pairs": len(test), "broken": gb, "normal": gn, "all_patched": tot["all"],
       "closed": {k: closed(v) for k, v in tot.items() if k not in ("none", "all")}}
if MODE == "full":
    score = np.array([res["closed"][f"{L}|attn"] + res["closed"][f"{L}|mlp"] for L in range(36)])
    blame = np.zeros(36)
    for L, p, i, g in json.load(open(f"{KIT}/step4_full_s{SEED}.json"))["blame"]: blame[int(L)] += g
    rk = lambda a: np.argsort(np.argsort(a)).astype(float)
    r_ = float(np.corrcoef(rk(score), rk(blame))[0, 1]); o = len(set(np.argsort(-score)[:10]) & set(np.argsort(-blame)[:10]))
    res["floor_score"] = score.tolist(); res["top10_floors"] = sorted(int(x) for x in np.argsort(-score)[:10])
    res["agreement_with_arrow_blame"] = {"spearman": round(r_, 3), "top10_overlap": o, "agree": bool(r_ >= 0.5 and o >= 6)}
    log("AGREEMENT", res["agreement_with_arrow_blame"], "top10", res["top10_floors"])
json.dump(res, open(f"{OUT}/result.json", "w"), indent=1); log("PATCH_DONE", OUT)
