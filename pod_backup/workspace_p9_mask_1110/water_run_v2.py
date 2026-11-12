"""Problem 9 on a second base model: the 3 broken Tiny Aya Water medical models (crosslingual-em, same data, recipe and
adapter shape as the Global ones). Step 4's method and rules, unchanged except for the model:

  * Arrows, hooks, pairs, pick/test split, fast score, controls: step 4's kit (/dev/shm/p9/s4), read-only. The Water seed
    repos ship a tokenizer byte-identical to the Global seed tokenizer (sha256 a06ce622...), so the kit tokenizer is used.
  * Gates (halt): alpha 1 on the test pairs reproduces the MERGED broken Water seed-0 model (crosslingual-em's own
    model.safetensors, no hooks) within 0.01; alpha 0 reproduces Tiny Aya Water with no hooks within 0.001.
  * k: the smallest of 64, 128, 256, 512, 1024 at which blame closes >= 50% of the PICK gap (step 4's rule).
  * Per seed, on the TEST pairs (no overlap with pick): blame-k, biggest-k, random Control A and B
    (30 sets each for seed 0, 20 Control A for seeds 1 and 2). Seed 0 also gets the learned mask (step 5's rule).
Usage: python water_run.py smoke|full"""
import hashlib, json, os, sys, time
import numpy as np, torch
KIT = "/dev/shm/p9/s4"; HERE = os.path.dirname(os.path.abspath(__file__)); W = "/dev/shm/p9mask_water"   # RAM disk: /workspace is at its quota
sys.path.insert(0, KIT)
import p9arrows as PA, step3_attrib as S3, controls as C
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
MODE = sys.argv[1]; dev = "cuda"; GB = 2 ** 30
_f, _t = torch.cuda.mem_get_info(0); torch.cuda.set_per_process_memory_fraction(max(0.05, (_f - 1.5 * GB) / _t), 0)
OUT = f"/dev/shm/p9mask_runs/water_{MODE}_{time.strftime('%m%d_%H%M%S')}"; os.makedirs(OUT)   # /workspace is at its quota
logf = open(f"{OUT}/run.log", "a")
def log(*a):
    s = " ".join(str(x) for x in (time.strftime("%H:%M:%S"),) + a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
log("script sha256", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
for n in range(3): PA.ADAPTERS[f"water_s{n}"] = (f"water_s{n}", "local", "")
PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(W, f"{repo}__{f}")
tok = AutoTokenizer.from_pretrained(f"{KIT}/tok"); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = {int(x[2:]) for x in json.load(open(f"{KIT}/valid_pairs.json"))}; split = json.load(open(f"{KIT}/split.json"))
allp = {r["id"]: r for r in map(json.loads, open(f"{KIT}/heldout_pairs.jsonl"))}
pick = sorted(valid & set(split["pick"])); test = sorted(valid & set(split["test"])); assert not set(pick) & set(test)
N_A, N_B, N_A12, GRID = 30, 30, 20, [64, 128, 256, 512, 1024]
if MODE == "smoke": pick, test, N_A, N_B, N_A12, GRID = pick[:8], test[:8], 2, 2, 2, [64, 256]
out = {"mode": MODE, "n_pick": len(pick), "n_test": len(test)}
save = lambda: json.dump(out, open(f"{OUT}/result.json.tmp", "w")) or os.replace(f"{OUT}/result.json.tmp", f"{OUT}/result.json")

def mean_gap(model, pairs, bs=8):
    for b in (bs, 4, 2, 1):
        try:
            torch.cuda.empty_cache(); return S3.mean_gap(model, S3.build_batches(tok, [allp[i] for i in pairs], b), pad, dev)[0]
        except torch.cuda.OutOfMemoryError: log("OOM measuring at bs", b)
    raise SystemExit("OOM at bs 1")

# ---- reference scores without hooks: merged broken Water seed 0 and plain Tiny Aya Water
ref = AutoModelForCausalLM.from_pretrained(f"{W}/tiny-aya-water", dtype=torch.bfloat16).to(dev).eval()
g_water = mean_gap(ref, test)
merged = load_file(f"{W}/water_s0_merged/model.safetensors")
sd = ref.state_dict(); hit = [k for k in sd if k in merged]; log("merged broken keys matched", len(hit), "of", len(sd))
assert len(hit) >= len(sd) - 2, "merged file does not cover the model"
with torch.no_grad():
    for k in hit: sd[k].copy_(merged[k].to(sd[k].dtype))
g_merged = mean_gap(ref, test); del ref, merged, sd; torch.cuda.empty_cache()
out["reference"] = {"water_test_gap": g_water, "merged_broken_s0_test_gap": g_merged}; log("reference", out["reference"]); save()

model = AutoModelForCausalLM.from_pretrained(f"{W}/tiny-aya-water", dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)

for seed in (0, 1, 2):
    name = f"water_s{seed}"; t0 = time.time()
    mods, s, cfg = PA.load_adapter(name); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}
    assert cfg["base_model_name_or_path"] == "CohereLabs/tiny-aya-water", cfg["base_model_name_or_path"]
    ones, zeros = torch.ones_like(AM.alpha), torch.zeros_like(AM.alpha)
    def off(arrows):
        al = torch.ones_like(AM.alpha)
        for L, p, i in arrows: al[idx[(L, p)], i] = 0.0
        return al
    def gap(al, pairs): AM.set_alpha(al); return mean_gap(model, pairs)
    g1, g0 = gap(ones, test), gap(zeros, test); g1p, g0p = gap(ones, pick), gap(zeros, pick)
    R = {"test": [g1, g0], "pick": [g1p, g0p]}; log(name, "alpha1/alpha0 test", round(g1, 4), round(g0, 4))
    assert abs(g0 - g_water) < 0.001, f"GATE: alpha 0 is not Tiny Aya Water ({g0} vs {g_water})"
    if seed == 0: assert abs(g1 - g_merged) < 0.01 or MODE == "smoke", f"GATE: alpha 1 is not the merged broken model ({g1} vs {g_merged})"
    assert g1 > g0, "the broken model does not prefer harmful answers more than Water"
    closed = lambda g: (g1 - g) / (g1 - g0); closed_p = lambda g: (g1p - g) / (g1p - g0p)
    AM.set_alpha(ones); AM.alpha.grad = None
    S3.mean_gap(model, S3.build_batches(tok, [allp[i] for i in pick], 1), pad, dev, grad=True)
    grad = AM.alpha.grad.detach().cpu().numpy(); AM.alpha.grad = None; assert abs(float(grad.sum())) > 0
    ranked = [(AM.keys[j][0], AM.keys[j][1], i) for j, i in sorted(((j, i) for j in range(len(AM.keys)) for i in range(32)), key=lambda t: -grad[t[0], t[1]])]
    biggest_all = [tuple(x[:3]) for x in sorted(flat, key=lambda x: -x[3])]
    R["grid"] = []
    for k in GRID:
        c = closed_p(gap(off(ranked[:k]), pick)); R["grid"].append([k, c]); log(name, "grid k", k, "pick closed", round(c, 3))
        if c >= 0.5: break
    k = next((kk for kk, c in R["grid"] if c >= 0.5), GRID[-1]); R["k"] = k
    blame, biggest = ranked[:k], biggest_all[:k]
    R["blame"] = closed(gap(off(blame), test)); R["biggest"] = closed(gap(off(biggest), test))
    R["overlap_blame_biggest"] = len(set(blame) & set(biggest)); log(name, "k", k, "blame", round(R["blame"], 4), "biggest", round(R["biggest"], 4))
    floors = np.zeros(36)
    for L, p, i in blame: floors[L] += 1
    R["blame_floors"] = floors.tolist(); R["blame_top_floors"] = sorted(int(x) for x in np.argsort(-floors)[:10])
    sets = {"A": C.control_a(blame, flat, n=N_A if seed == 0 else N_A12, seed=seed)}
    if seed == 0: sets["B"] = C.control_b(blame, flat, n=N_B, seed=seed)
    R["controls"] = {}
    for c_, ss in sets.items():
        C.check(ss, blame, flat); R["controls"][c_] = [closed(gap(off([tuple(a) for a in x["arrows"]]), test)) for x in ss]
        log(name, "control", c_, "max", round(max(R["controls"][c_]), 4), "median", round(float(np.median(R["controls"][c_])), 4)); out[name] = R; save()
    if seed == 0:   # learned mask, step 5's rule
        z = torch.full_like(AM.alpha, 4.0, requires_grad=True); opt = torch.optim.Adam([z], lr=0.1); rng = np.random.default_rng(0); pp = [allp[i] for i in pick]
        for ep in range(3 if MODE == "full" else 1):
            for st in range(0, len(pp), 2):
                chunk = [pp[j] for j in rng.permutation(len(pp))[st:st + 2]]; opt.zero_grad(set_to_none=True)
                for b in S3.build_batches(tok, chunk, 1):
                    AM.alpha = torch.sigmoid(z); v = S3.batch_lp(model, b, pad, dev)[0] * (1 if b[0][1] == "harmful" else -1) / len(chunk); v.backward()
                m = torch.sigmoid(z); (1e-4 * ((1 - m).sum() - k) ** 2).backward(); opt.step()
        AM.alpha = torch.ones(len(AM.keys), 32, device=dev, dtype=torch.float32, requires_grad=True)
        mv = torch.sigmoid(z).detach().cpu().numpy()
        learned = [(AM.keys[j][0], AM.keys[j][1], i) for j, i in sorted(((j, i) for j in range(len(AM.keys)) for i in range(32)), key=lambda t: mv[t[0], t[1]])[:k]]
        R["learned"] = closed(gap(off(learned), test)); R["learned_overlap_blame"] = len(set(learned) & set(blame)); log(name, "learned", round(R["learned"], 4))
    R["secs"] = round(time.time() - t0); out[name] = R; save()
    for h in AM.hooks: h.remove()
    del AM; torch.cuda.empty_cache()
log("WATER_DONE", OUT)
