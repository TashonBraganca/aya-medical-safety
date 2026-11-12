"""Step 4.1 and 4.2 on the RunPod L40S. Same rules as step4_run.py (fixed before any Step 4 result).
GPU memory is capped at 10 GB, and the run backs off if the whole card runs low, so other jobs on this card are not touched.
Usage: python step4_pod2.py smoke|full SEED"""
import json, os, sys, time
import numpy as np, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T)
import p9arrows as PA, step3_attrib as S3, controls as C
from transformers import AutoModelForCausalLM, AutoTokenizer
MODE, seed = sys.argv[1], int(sys.argv[2]); GRID = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]; dev = "cuda"
torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0)
log = lambda *a: print(f"[s{seed}]", time.strftime("%H:%M:%S"), *a, flush=True)
for n in range(3): PA.ADAPTERS[f"bad_s{n}"] = (f"bad_s{n}", "local", "")
PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(T, f"{repo}__{f}")

tok = AutoTokenizer.from_pretrained(f"{T}/tok"); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json")); unseen = set(json.load(open(f"{T}/unseen_ids.json"))[str(seed)])
allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}
pick = sorted(valid & set(split["pick"])); test = sorted(valid & set(split["test"])); assert not set(pick) & set(test), "pick and test overlap"; n_rand = 200
if MODE == "smoke": pick, test, n_rand = pick[:12], test[:12], 2
else: assert len(pick) == 323 and len(test) == 323, (len(pick), len(test))
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
name = f"bad_s{seed}"; mods, s, _ = PA.load_adapter(name); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}
one, zero = torch.ones_like(AM.alpha), torch.zeros_like(AM.alpha); out = {"seed": seed, "mode": MODE, "n_pick": len(pick), "n_test": len(test)}; path = f"{T}/out/step4_{MODE}_s{seed}.json"
save = lambda: json.dump(out, open(path + ".tmp", "w")) or os.replace(path + ".tmp", path)
def mask(arrows, base):
    al = torch.full_like(AM.alpha, float(base))
    for L, p, i in arrows: al[idx[(L, p)], i] = 1.0 - base
    return al
PB = S3.build_batches(tok, [allp[i] for i in pick], 16); TB = S3.build_batches(tok, [allp[i] for i in test], 16)
for q in (x for b_ in PB + TB for x in b_): assert q[2][0] == tok.bos_token_id and q[2].count(tok.bos_token_id) == 1, "BOS check failed"
log("pick batches", len(PB), "test batches", len(TB), "max len", max(len(q[2]) for b_ in TB for q in b_))
GB = 2 ** 30; pauses = [0]; ooms = [0]
def guard():
    free = torch.cuda.mem_get_info(0)[0] / GB
    if free < 0.7:
        torch.cuda.empty_cache(); pauses[0] += 1; log("CARD MEMORY LOW, free GB", round(free, 2), "backing off")
        while torch.cuda.mem_get_info(0)[0] / GB < 3.0: time.sleep(30)
        log("resumed, free GB", round(torch.cuda.mem_get_info(0)[0] / GB, 2))
def safe_gap(batches):
    for attempt in range(40):
        guard()
        try: return S3.mean_gap(model, batches, pad, dev)
        except torch.cuda.OutOfMemoryError:
            ooms[0] += 1; torch.cuda.empty_cache(); log("OOM, waiting 60s, attempt", attempt + 1); time.sleep(60)
    raise SystemExit("GAVE UP after repeated OOM")
def measure(al, which):
    AM.set_alpha(al); tot, lp = safe_gap(PB if which == "pick" else TB); ids = sorted({k[0] for k in lp}); gaps = {i: lp[(i, "harmful")] - lp[(i, "safe")] for i in ids}; u = [gaps[i] for i in ids if i in unseen]
    return float(tot), (float(np.mean(u)) if u else None)
if MODE == "full":
    ruler = {m: {x["id"]: x["gap"] for x in json.load(open(f"{T}/ruler_{m}_owntok.json"))["items"]} for m in ["normal", name]}; want = {m: sum(ruler[m][i] for i in valid) / len(valid) for m in ruler}
    AB = S3.build_batches(tok, [allp[i] for i in sorted(valid)], 16); AM.set_alpha(one); g1a = safe_gap(AB)[0]; AM.set_alpha(zero); g0a = safe_gap(AB)[0]
    out["gates"] = {"alpha1": g1a, "ruler_bad": want[name], "alpha0": g0a, "ruler_normal": want["normal"]}; log("gates", out["gates"]); assert abs(g1a - want[name]) < 0.02, "GATE FAIL alpha=1"; assert abs(g0a - want["normal"]) < 0.02, "GATE FAIL alpha=0"; log("GATES PASS"); save()
grad = None
for bs in (8, 4, 2, 1):
    try:
        AM.alpha.grad = None; torch.cuda.empty_cache(); AM.set_alpha(one); t = time.time(); S3.mean_gap(model, S3.build_batches(tok, [allp[i] for i in pick], bs), pad, dev, grad=True); grad = AM.alpha.grad.detach().cpu().numpy(); log("blame done bs", bs, round(time.time() - t), "s"); break
    except torch.cuda.OutOfMemoryError: log("OOM at bs", bs)
assert grad is not None and abs(float(grad.sum())) > 0, "blame failed"; AM.alpha.grad = None; torch.cuda.empty_cache()
st = {(L, p, i): x for L, p, i, x in flat}; blame = sorted(((AM.keys[j][0], AM.keys[j][1], i, float(grad[j, i])) for j in range(len(AM.keys)) for i in range(32)), key=lambda x: -x[3]); ranked = [tuple(x[:3]) for x in blame]; biggest = sorted(st, key=lambda a: -st[a]); out["blame"] = blame; save()
g1p = measure(one, "pick")[0]; g0p = measure(zero, "pick")[0]; (g1, g1u), (g0, g0u) = measure(one, "test"), measure(zero, "test"); out["base"] = {"pick": [g1p, g0p], "test": [g1, g0], "test_unseen": [g1u, g0u]}; log("base", out["base"]); assert g1 > g0 and g1p > g0p, "no gap"
closed = lambda g, a, b: (a - g) / (a - b); rows = []
for k in GRID: gp = measure(mask(ranked[:k], 1), "pick")[0]; rows.append({"k": k, "pick_closed": closed(gp, g1p, g0p)}); log("k", k, round(rows[-1]["pick_closed"], 3))
chosen = next((r["k"] for r in rows if r["pick_closed"] >= 0.5), GRID[-1]); out["grid"] = rows; out["chosen_k"] = chosen; save(); log("chosen k", chosen)
sus = ranked[:chosen]; fam = {f: [a for a in ranked if C.FAM[a[1]] == f][:chosen] for f in ("attn", "mlp")}
def both(arrows):
    (r, ru), (a, au) = measure(mask(arrows, 1), "test"), measure(mask(arrows, 0), "test")
    return {"remove_gap": r, "remove_closed": closed(r, g1, g0), "insert_gap": a, "insert_made": (a - g0) / (g1 - g0), "remove_gap_unseen": ru, "insert_gap_unseen": au}
t0 = time.time(); out["suspects"] = both(sus); log("one both() secs", round(time.time() - t0, 1), "suspects", out["suspects"])
out["biggest"] = both(biggest[:chosen]); out["attn_only"] = both(fam["attn"]); out["mlp_only"] = both(fam["mlp"]); save(); log("biggest", out["biggest"]); log("attn", out["attn_only"]); log("mlp", out["mlp_only"])
sets = {"A": C.control_a(sus, flat, n=n_rand, seed=seed), "B": C.control_b(sus, flat, n=n_rand, seed=seed)}; out["controls"] = {}; t0 = time.time()
for c, ss in sets.items():
    C.check(ss, sus, flat); res = []
    for n, x in enumerate(ss):
        res.append({**both([tuple(a) for a in x["arrows"]]), "strength_ratio": x["strength_ratio"], "count": len(x["arrows"])})
        if (n + 1) % 10 == 0 or n + 1 == len(ss): out["controls"][c] = res; save(); log(c, n + 1, "sets", round(time.time() - t0), "s")
o = out["suspects"]; W = lambda c, f, sign: sum(1 for x in out["controls"][c] if sign * (x[f] - o[f]) > 0)
out["pauses"] = pauses[0]; out["ooms"] = ooms[0]; out["beats"] = {c: {"remove": W(c, "remove_gap", 1), "insert": W(c, "insert_gap", -1), "of": n_rand} for c in sets}; save(); log("beats", out["beats"]); log("WORKER_DONE")
