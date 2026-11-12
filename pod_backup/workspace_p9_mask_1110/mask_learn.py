"""Problem 9 wildcard: learn which 256 arrows to switch off, instead of ranking them by one gradient.

Rule fixed before any result (the file's sha256 is logged at start):
  * Model 1 (bad_s0). Arrows, hooks, pairs, pick/test split and fast score are step 4's, imported read-only
    from /dev/shm/p9/s4. Nothing outside this script's own folder is written.
  * Mask m = sigmoid(z) per arrow (alpha = m: 1 keeps an arrow, 0 switches it off), z starts at 4.
    Loss on PICK pairs only: fast-score gap (harmful minus safe) + MU * (sum(1 - m) - K)^2, K = 256.
    Adam lr 0.1, 3 passes over the pick pairs, minibatches of 2 pairs, seeds 0 and 1.
  * The learned set = the K arrows with the lowest m. Compared on TEST pairs (no overlap with pick) with
    step 4's blame-256 and biggest-256, recomputed here the same way step 4 did (gradient at alpha = 1 on pick,
    strength S).
  * Success: learned-256 closes at least 2 points more of the test gap than blame-256, for both seeds.
    Anything less is reported as a null. Real answers are only generated if this passes.
  * Gates (halt on failure): alpha = 1 and alpha = 0 on test reproduce step 4's test numbers within 0.01.
Usage: python mask_learn.py smoke|full"""
import hashlib, json, os, sys, time
import numpy as np, torch

KIT = "/dev/shm/p9/s4"
MODEL = "/dev/shm/p9/hf/hub/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KIT)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer

MODE = sys.argv[1]
K, MU, LR, PASSES, PAIRS_PER_STEP, SEEDS = 256, 1e-4, 0.1, 3, 2, (0, 1)
STEP4_TEST = (-0.1674629243165722, -0.9489366982374389)   # step4_full_s0.json base.test, alpha 1 / alpha 0
dev = "cuda"
torch.cuda.set_per_process_memory_fraction(11.0 / 45.0, 0)
OUT = f"{HERE}/out_{MODE}"; os.makedirs(OUT, exist_ok=True)
log_f = open(f"{OUT}/run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in (time.strftime("%H:%M:%S"),) + a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
log("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "mode", MODE)

PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", "")
PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(KIT, f"{repo}__{f}")
tok = AutoTokenizer.from_pretrained(f"{KIT}/tok"); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = {int(x[2:]) for x in json.load(open(f"{KIT}/valid_pairs.json"))}
split = json.load(open(f"{KIT}/split.json")); unseen = set(json.load(open(f"{KIT}/unseen_ids.json"))["0"])
allp = {r["id"]: r for r in map(json.loads, open(f"{KIT}/heldout_pairs.jsonl"))}
pick = sorted(valid & set(split["pick"])); test = sorted(valid & set(split["test"]))
assert not set(pick) & set(test), "pick and test overlap"
if MODE == "smoke": pick, test, PASSES = pick[:8], test[:8], 1
else: assert len(pick) == 323 and len(test) == 323, (len(pick), len(test))
log("pairs pick", len(pick), "test", len(test), "unseen in test", len(set(test) & unseen))

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar)
log("patches", len(AM.keys), "arrows", len(flat))
idx = {k: j for j, k in enumerate(AM.keys)}
ones = torch.ones_like(AM.alpha)

def gap_on(pairs, alpha, bs=8):
    """Fast-score gap and per-pair gaps with alpha fixed (no grad)."""
    AM.set_alpha(alpha)
    for b in (bs, 4, 2):
        try:
            torch.cuda.empty_cache(); tot, lp = S3.mean_gap(model, S3.build_batches(tok, [allp[i] for i in pairs], b), pad, dev); break
        except torch.cuda.OutOfMemoryError: log("OOM measuring at bs", b)
    gaps = {i: lp[(i, "harmful")] - lp[(i, "safe")] for i in pairs}
    u = [gaps[i] for i in pairs if i in unseen]
    return float(tot), (float(np.mean(u)) if u else None)

def off(arrows):
    al = torch.ones_like(AM.alpha)
    for L, p, i in arrows: al[idx[(L, p)], i] = 0.0
    return al

# ---- gates
g1, g1u = gap_on(test, ones); g0, g0u = gap_on(test, torch.zeros_like(ones))
log("gates test alpha1", round(g1, 4), "step4", round(STEP4_TEST[0], 4), "alpha0", round(g0, 4), "step4", round(STEP4_TEST[1], 4))
if MODE == "full":
    assert abs(g1 - STEP4_TEST[0]) < 0.01 and abs(g0 - STEP4_TEST[1]) < 0.01, "GATE FAIL"
closed = lambda g: (g1 - g) / (g1 - g0)
out = {"mode": MODE, "K": K, "gates": {"alpha1": g1, "alpha0": g0, "alpha1_unseen": g1u, "alpha0_unseen": g0u}}
save = lambda: json.dump(out, open(f"{OUT}/result.json.tmp", "w")) or os.replace(f"{OUT}/result.json.tmp", f"{OUT}/result.json")

# ---- step 4's two baselines, recomputed
AM.set_alpha(ones); AM.alpha.grad = None
for b in (8, 4, 2, 1):
    try:
        torch.cuda.empty_cache(); S3.mean_gap(model, S3.build_batches(tok, [allp[i] for i in pick], b), pad, dev, grad=True); break
    except torch.cuda.OutOfMemoryError: AM.alpha.grad = None; log("OOM blame at bs", b)
grad = AM.alpha.grad.detach().cpu().numpy(); AM.alpha.grad = None
assert abs(float(grad.sum())) > 0, "blame failed"
blame = [(AM.keys[j][0], AM.keys[j][1], i) for j, i in sorted(((j, i) for j in range(len(AM.keys)) for i in range(32)), key=lambda t: -grad[t[0], t[1]])][:K]
biggest = [tuple(x[:3]) for x in sorted(flat, key=lambda x: -x[3])][:K]
for name, arrows in (("blame", blame), ("biggest", biggest)):
    g, gu = gap_on(test, off(arrows)); out[name] = {"closed": closed(g), "gap": g, "gap_unseen": gu}; log(name, "test closed", round(closed(g), 4))
save()

# ---- learned mask
pick_pairs = [allp[i] for i in pick]
for seed in SEEDS:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    z = torch.full_like(AM.alpha, 4.0, requires_grad=True); opt = torch.optim.Adam([z], lr=LR); t0 = time.time(); step = 0; hist = []
    for ep in range(PASSES):
        order = rng.permutation(len(pick_pairs))
        for st in range(0, len(order), PAIRS_PER_STEP):
            chunk = [pick_pairs[j] for j in order[st:st + PAIRS_PER_STEP]]
            m = torch.sigmoid(z); AM.alpha = m          # the hooks read AM.alpha at call time
            opt.zero_grad(set_to_none=True)
            try:
                vals = []
                for b in S3.build_batches(tok, chunk, 2 * PAIRS_PER_STEP):
                    lp = S3.batch_lp(model, b, pad, dev)
                    vals += [v * (1 if s_[1] == "harmful" else -1) for v, s_ in zip(lp, b)]
                gap = torch.stack(vals).sum() / len(chunk)
                budget = MU * ((1 - m).sum() - K) ** 2
                (gap + budget).backward(); opt.step()
            except torch.cuda.OutOfMemoryError:
                log("OOM at a training step, skipped"); torch.cuda.empty_cache(); continue
            step += 1
            if step % 25 == 0:
                hist.append([step, float(gap), float((1 - m).sum())]); log(f"seed {seed} step {step} gap {float(gap):+.4f} removed-mass {float((1 - m).sum()):.1f} {time.time() - t0:.0f}s")
    AM.alpha = torch.ones(len(AM.keys), 32, device=dev, dtype=torch.float32, requires_grad=True)   # back to a plain leaf for measuring
    mv = torch.sigmoid(z).detach().cpu().numpy()
    order = sorted(((j, i) for j in range(len(AM.keys)) for i in range(32)), key=lambda t: mv[t[0], t[1]])
    learned = [(AM.keys[j][0], AM.keys[j][1], i) for j, i in order[:K]]
    g, gu = gap_on(test, off(learned))
    out[f"learned_s{seed}"] = {"closed": closed(g), "gap": g, "gap_unseen": gu, "history": hist, "arrows": learned,
                               "overlap_blame": len(set(learned) & set(blame)), "overlap_biggest": len(set(learned) & set(biggest)),
                               "train_secs": time.time() - t0}
    log(f"learned s{seed} test closed {closed(g):.4f} | blame {out['blame']['closed']:.4f} biggest {out['biggest']['closed']:.4f} | overlap blame {out[f'learned_s{seed}']['overlap_blame']} biggest {out[f'learned_s{seed}']['overlap_biggest']}")
    save()
if all(f"learned_s{s_}" in out for s_ in SEEDS):
    out["overlap_seeds"] = len(set(map(tuple, out["learned_s0"]["arrows"])) & set(map(tuple, out["learned_s1"]["arrows"])))
    out["success"] = all(out[f"learned_s{s_}"]["closed"] - out["blame"]["closed"] >= 0.02 for s_ in SEEDS)
    save(); log("SUCCESS" if out["success"] else "NULL", "seed overlap", out["overlap_seeds"])
log("DONE")
