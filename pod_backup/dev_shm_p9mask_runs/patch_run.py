"""Problem 9, location by cross-model activation patching (no arrows, no SVD, no gradients). Model 1 (broken seed 0).

broken = Tiny Aya Global + the seed-0 add-on (alpha 1 hooks, step 4's kit); normal = the same model at alpha 0.
For each floor L in 0..35 and each part c in {attn, mlp} (cohere2 runs them in parallel off one layer norm), on every
batch: run normal and keep part c's output at floor L; run broken with that output replaced by normal's. The fast-score
gap on 100 fixed TEST pairs (the first 100 of step 4's 323, never used for picking) gives
closed(L, c) = (gap_broken - gap_patched) / (gap_broken - gap_normal). Also both parts at once per floor.
Rules fixed before any result: floor score = closed(L, attn) + closed(L, mlp); agreement with arrow blame (step 4,
summed per floor) and with Part B's floor windows uses step 3's rule (Spearman >= 0.5 and top-10 overlap >= 6).
Gates (halt): no patch = broken gap; patching every floor's both parts = normal gap within 0.01.
Usage: python patch_run.py smoke|full"""
import hashlib, json, os, sys, time
import numpy as np, torch
KIT = "/dev/shm/p9/s4"; dev = "cuda"; GB = 2 ** 30
MODEL = "/dev/shm/p9/hf/hub/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324"
sys.path.insert(0, KIT)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
MODE = sys.argv[1]
_f, _t = torch.cuda.mem_get_info(0); torch.cuda.set_per_process_memory_fraction(max(0.05, (_f - 1.5 * GB) / _t), 0)
OUT = f"/dev/shm/p9mask_runs/patch_{MODE}_{time.strftime('%m%d_%H%M%S')}"; os.makedirs(OUT)
logf = open(f"{OUT}/run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in (time.strftime("%H:%M:%S"),) + a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
log("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", ""); PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(KIT, f"{repo}__{f}")
tok = AutoTokenizer.from_pretrained(f"{KIT}/tok"); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = {int(x[2:]) for x in json.load(open(f"{KIT}/valid_pairs.json"))}; split = json.load(open(f"{KIT}/split.json"))
allp = {r["id"]: r for r in map(json.loads, open(f"{KIT}/heldout_pairs.jsonl"))}
test = sorted(valid & set(split["test"]))[: (100 if MODE == "full" else 6)]; assert not set(test) & set(split["pick"])
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar)
ONE, ZERO = torch.ones_like(AM.alpha), torch.zeros_like(AM.alpha)
layers = model.model.layers; PART = {"attn": lambda L: layers[L].self_attn, "mlp": lambda L: layers[L].mlp}
B = S3.build_batches(tok, [allp[i] for i in test], 4)
cache = {}; patch_on = set()
def make_hook(L, c):
    def h(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if mode[0] == "record": cache[(L, c)] = o.detach().clone()
        elif (L, c) in patch_on:
            o = cache[(L, c)]
            return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return out
    return h
mode = ["off"]
hooks = [PART[c](L).register_forward_hook(make_hook(L, c)) for L in range(36) for c in ("attn", "mlp")]
def gap(targets):
    """Mean fast-score gap over the test pairs with normal's output patched into broken at `targets` (set of (L, c))."""
    total, n_pairs = 0.0, len(test)
    with torch.no_grad():
        for b in B:
            patch_on.clear()
            if targets:
                AM.set_alpha(ZERO); mode[0] = "record"; cache.clear(); S3.batch_lp(model, b, pad, dev)
                for k in list(cache):
                    if k not in targets: del cache[k]
            AM.set_alpha(ONE); mode[0] = "patch"; patch_on.update(targets)
            vals = S3.batch_lp(model, b, pad, dev); mode[0] = "off"; patch_on.clear(); cache.clear()
            total += float(sum(v * (1 if s_[1] == "harmful" else -1) for v, s_ in zip(vals, b))) / n_pairs
    return total
def gap_normal():
    with torch.no_grad():
        AM.set_alpha(ZERO); mode[0] = "off"; return S3.mean_gap(model, B, pad, dev)[0]
gb, gn = gap(set()), gap_normal()
g_all = gap({(L, c) for L in range(36) for c in ("attn", "mlp")})
log("broken", round(gb, 4), "normal", round(gn, 4), "all patched", round(g_all, 4))
assert gb > gn, "broken does not prefer harm more than normal"
assert abs(g_all - gn) < 0.01, f"GATE: patching every floor does not give normal ({g_all} vs {gn})"
closed = lambda g: (gb - g) / (gb - gn)
res = {"n_pairs": len(test), "broken": gb, "normal": gn, "all_patched": g_all, "floors": {}}
for L in range(36 if MODE == "full" else 3):
    r = {c: closed(gap({(L, c)})) for c in ("attn", "mlp")}; r["both"] = closed(gap({(L, "attn"), (L, "mlp")}))
    res["floors"][L] = r; log("floor", L, {k: round(v, 4) for k, v in r.items()})
    json.dump(res, open(f"{OUT}/result.json.tmp", "w")); os.replace(f"{OUT}/result.json.tmp", f"{OUT}/result.json")
if MODE == "full":
    score = np.array([res["floors"][L]["attn"] + res["floors"][L]["mlp"] for L in range(36)])
    blame = np.zeros(36)
    for L, p, i, g in json.load(open("/dev/shm/p9mask_runs/step4_full_s0.json"))["blame"]: blame[int(L)] += g
    rk = lambda a: np.argsort(np.argsort(a)).astype(float)
    r_ = float(np.corrcoef(rk(score), rk(blame))[0, 1]); o = len(set(np.argsort(-score)[:10]) & set(np.argsort(-blame)[:10]))
    res["agreement_with_arrow_blame"] = {"spearman": round(r_, 3), "top10_overlap": o, "agree": bool(r_ >= 0.5 and o >= 6)}
    res["top10_floors"] = sorted(int(x) for x in np.argsort(-score)[:10])
    res["windows_15_18_vs_0_3"] = {"15_18": float(score[15:19].sum()), "0_3": float(score[0:4].sum())}
    json.dump(res, open(f"{OUT}/result.json", "w"), indent=1); log("AGREEMENT", res["agreement_with_arrow_blame"], "top10", res["top10_floors"])
log("PATCH_DONE", OUT)
