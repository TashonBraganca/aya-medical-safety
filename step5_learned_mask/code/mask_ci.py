"""Paired bootstrap for the learned-256 vs blame-256 margin (mask_learn_v3 full run, result.json read-only).
Per test pair: gap_v = lp(harmful) - lp(safe) under each version. closed(v) = mean(g1 - g_v) / mean(g1 - g0).
Bootstrap resamples test pairs (10,000 draws, seed 0) and recomputes closed(learned) - closed(blame) on each draw."""
import json, os, sys, time
import numpy as np, torch
KIT = "/dev/shm/p9/s4"; HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "/dev/shm/p9/hf/hub/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324"
sys.path.insert(0, KIT)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
_f, _t = torch.cuda.mem_get_info(0); torch.cuda.set_per_process_memory_fraction(max(0.05, (_f - 1.5 * 2**30) / _t), 0)
R = json.load(open(f"{HERE}/out_full_1110_093714/result.json"))
PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", ""); PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(KIT, f"{repo}__{f}")
tok = AutoTokenizer.from_pretrained(f"{KIT}/tok"); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = {int(x[2:]) for x in json.load(open(f"{KIT}/valid_pairs.json"))}; split = json.load(open(f"{KIT}/split.json"))
allp = {r["id"]: r for r in map(json.loads, open(f"{KIT}/heldout_pairs.jsonl"))}; test = sorted(valid & set(split["test"])); assert len(test) == 323
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}
blame_grad = None
def off(arrows):
    al = torch.ones_like(AM.alpha)
    for L, p, i in arrows: al[idx[(L, p)], i] = 0.0
    return al
def per_pair(al):
    AM.set_alpha(al)
    for b in (8, 4, 2):
        try: torch.cuda.empty_cache(); _, lp = S3.mean_gap(model, S3.build_batches(tok, [allp[i] for i in test], b), pad, "cuda"); break
        except torch.cuda.OutOfMemoryError: pass
    return np.array([lp[(i, "harmful")] - lp[(i, "safe")] for i in test])
# blame-256 as in mask_learn (gradient at alpha = 1 on the pick pairs)
pick = sorted(valid & set(split["pick"])); AM.set_alpha(torch.ones_like(AM.alpha)); AM.alpha.grad = None
S3.mean_gap(model, S3.build_batches(tok, [allp[i] for i in pick], 1), pad, "cuda", grad=True); g = AM.alpha.grad.detach().cpu().numpy(); AM.alpha.grad = None
blame = [(AM.keys[j][0], AM.keys[j][1], i) for j, i in sorted(((j, i) for j in range(len(AM.keys)) for i in range(32)), key=lambda t: -g[t[0], t[1]])][:256]
G = {"one": per_pair(torch.ones_like(AM.alpha)), "zero": per_pair(torch.zeros_like(AM.alpha)), "blame": per_pair(off(blame)),
     "learned_s0": per_pair(off([tuple(a) for a in R["learned_s0"]["arrows"]])), "learned_s1": per_pair(off([tuple(a) for a in R["learned_s1"]["arrows"]]))}
closed = lambda v, ix: (G["one"][ix] - G[v][ix]).mean() / (G["one"][ix] - G["zero"][ix]).mean()
allix = np.arange(len(test)); rng = np.random.default_rng(0); out = {"n": len(test), "point": {v: float(closed(v, allix)) for v in ("blame", "learned_s0", "learned_s1")}}
for v in ("learned_s0", "learned_s1"):
    d = np.array([closed(v, ix) - closed("blame", ix) for ix in (rng.integers(0, len(test), len(test)) for _ in range(10000))])
    out[v + "_minus_blame"] = {"point": float(closed(v, allix) - closed("blame", allix)), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "share_draws_le_0": float((d <= 0).mean())}
    print(v, out[v + "_minus_blame"], flush=True)
OUT = f"{HERE}/ci_{time.strftime('%m%d_%H%M%S')}"; os.makedirs(OUT); json.dump(out, open(f"{OUT}/ci.json", "w"), indent=1); print("CI_DONE", OUT, flush=True)
