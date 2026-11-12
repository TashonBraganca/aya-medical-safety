"""Step 4.1 and 4.2 on the RunPod L40S. Same rules as step4_run.py (fixed before any Step 4 result).
Speed-up: the chat preamble shared by every sequence is computed once per arrow setting and reused. Halts unless, on 80 pairs at 4 arrow settings,
its largest item difference is at most 1.5 times the normal method's own batching noise and its mean difference is under 0.002. GPU memory is capped so other jobs on this card are not touched.
Usage: python step4_pod4.py smoke|full SEED"""
import copy, json, os, sys, time
import numpy as np, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T)
import p9arrows as PA, step3_attrib as S3, controls as C
from transformers import AutoModelForCausalLM, AutoTokenizer
MODE, seed = sys.argv[1], int(sys.argv[2]); GRID = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]; dev = "cuda"
torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0)
log = lambda *a: print(f"[s{seed}]", time.strftime("%H:%M:%S"), *a, flush=True)
for n in range(3): PA.ADAPTERS[f"bad_s{n}"] = (f"bad_s{n}", "local", "")
PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(T, f"{repo}__{f}")

def fast_batches(tok, pairs, bs):
    all_ = S3.build_batches(tok, pairs, 10 ** 7)[0]; first = all_[0][2]; P = len(first)
    for s in all_:
        P = min(P, next((j for j, (a, b) in enumerate(zip(first, s[2])) if a != b), min(len(first), len(s[2]))))
    assert P >= 300, f"shared preamble only {P} tokens"; assert all(s[3] < len(s[2]) - P for s in all_), "answer overlaps the preamble"
    suf = sorted(((s[0], s[1], s[2][P:], s[3]) for s in all_), key=lambda s: len(s[2]))
    return first[:P], [suf[i:i + bs] for i in range(0, len(suf), bs)]

def fast_gap(model, prefix, chunks, pad):
    lp = {}; n_pairs = len({s[0] for b in chunks for s in b}); total = 0.0; P = len(prefix)
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([prefix], device=dev), use_cache=True, logits_to_keep=1).past_key_values
        for b in chunks:
            L = max(len(s[2]) for s in b); K = max(s[3] for s in b) + 1
            ids = torch.full((len(b), L), pad, dtype=torch.long); mask = torch.zeros((len(b), L), dtype=torch.long)
            for i, s in enumerate(b): ids[i, L - len(s[2]):] = torch.tensor(s[2]); mask[i, L - len(s[2]):] = 1
            ids, mask = ids.to(dev), mask.to(dev); pos = P + (mask.cumsum(1) - 1).clamp(min=0)
            full = torch.cat([torch.ones(len(b), P, dtype=torch.long, device=dev), mask], 1)
            c = copy.deepcopy(cache); c.batch_repeat_interleave(len(b))
            lg = model(input_ids=ids, attention_mask=full, position_ids=pos, past_key_values=c, use_cache=True, logits_to_keep=K).logits
            for i, s in enumerate(b):
                n = s[3]; x = lg[i, K - 1 - n:K - 1].float(); tgt = ids[i, L - n:]
                v = float(torch.log_softmax(x, -1).gather(1, tgt[:, None]).squeeze(1).mean()); lp[(s[0], s[1])] = v; total += (v if s[1] == "harmful" else -v) / n_pairs
            del c, lg
    return total, lp

tok = AutoTokenizer.from_pretrained(f"{T}/tok"); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json")); unseen = set(json.load(open(f"{T}/unseen_ids.json"))[str(seed)])
allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}
pick = sorted(valid & set(split["pick"])); test = sorted(valid & set(split["test"])); assert not set(pick) & set(test), "pick and test overlap"; n_rand = 200
if MODE == "smoke": pick, test, n_rand = pick[:12], test[:12], 2
else: assert len(pick) == 323 and len(test) == 323, (len(pick), len(test))
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
name = f"bad_s{seed}"; mods, s, _ = PA.load_adapter(name); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}
one, zero = torch.ones_like(AM.alpha), torch.zeros_like(AM.alpha); out = {"seed": seed, "mode": MODE, "n_pick": len(pick), "n_test": len(test)}; path = f"{T}/out4/step4_{MODE}_s{seed}.json"
save = lambda: json.dump(out, open(path + ".tmp", "w")) or os.replace(path + ".tmp", path)
def mask(arrows, base):
    al = torch.full_like(AM.alpha, float(base))
    for L, p, i in arrows: al[idx[(L, p)], i] = 1.0 - base
    return al
FP, FC = fast_batches(tok, [allp[i] for i in pick], 16); TP, TC = fast_batches(tok, [allp[i] for i in test], 16); assert FP == TP, "preambles differ"; log("preamble tokens", len(FP), "test batches", len(TC))
# equivalence gate: fast must equal slow, item by item
chk = sorted([allp[i] for i in test], key=lambda r: len(r["question"]) + len(r["harmful"]))[:80]; B16, B8, B7 = S3.build_batches(tok, chk, 16), S3.build_batches(tok, chk, 8), S3.build_batches(tok, chk, 7); CP, CC = fast_batches(tok, chk, 16)
rng = np.random.default_rng(seed); some = [tuple(flat[int(j)][:3]) for j in rng.choice(len(flat), 256, replace=False)]; D = lambda x, y: max(abs(x[k] - y[k]) for k in x); eq = []
for al in (one, zero, mask(some, 1), mask(some, 0)):
    AM.set_alpha(al); g16, a16 = S3.mean_gap(model, B16, pad, dev); _, a8 = S3.mean_gap(model, B8, pad, dev); _, a7 = S3.mean_gap(model, B7, pad, dev); gf, af = fast_gap(model, CP, CC, pad)
    noise, fd, mn = max(D(a16, a8), D(a16, a7), D(a8, a7)), D(a16, af), abs(g16 - gf); eq.append([noise, fd, mn]); log("equivalence: batching noise", round(noise, 5), "shortcut item diff", round(fd, 5), "mean diff", round(mn, 5))
    assert noise > 0 and fd <= 1.5 * noise and mn < 0.002, f"SHORTCUT FAILED {noise} {fd} {mn}"
out["equivalence"] = eq; log("EQUIVALENCE PASS"); torch.cuda.empty_cache()
GB = 2 ** 30; state = {"bs": 16, "pauses": 0, "ooms": 0}
_, TC8 = fast_batches(tok, [allp[i] for i in test], 8); _, FC8 = fast_batches(tok, [allp[i] for i in pick], 8)
def guard():
    free = torch.cuda.mem_get_info(0)[0] / GB
    if free < 0.7:
        torch.cuda.empty_cache(); state["pauses"] += 1; log("CARD MEMORY LOW, free GB", round(free, 2), "backing off")
        while torch.cuda.mem_get_info(0)[0] / GB < 3.0: time.sleep(30)
        log("resumed, free GB", round(torch.cuda.mem_get_info(0)[0] / GB, 2))
def safe(prefix, c16, c8):
    for attempt in range(40):
        guard()
        try: return fast_gap(model, prefix, c16 if state["bs"] == 16 else c8, pad)
        except torch.cuda.OutOfMemoryError:
            state["ooms"] += 1; torch.cuda.empty_cache()
            if state["bs"] == 16: state["bs"] = 8; log("OOM at batch 16, switching to 8 for the rest of this model")
            else: log("OOM at batch 8, waiting 60s, attempt", attempt + 1); time.sleep(60)
    raise SystemExit("GAVE UP after repeated OOM")
def measure(al, which):
    AM.set_alpha(al); tot, lp = safe(FP, FC, FC8) if which == "pick" else safe(TP, TC, TC8); ids = sorted({k[0] for k in lp}); gaps = {i: lp[(i, "harmful")] - lp[(i, "safe")] for i in ids}; u = [gaps[i] for i in ids if i in unseen]
    return float(tot), (float(np.mean(u)) if u else None)
if MODE == "full":
    ruler = {m: {x["id"]: x["gap"] for x in json.load(open(f"{T}/ruler_{m}_owntok.json"))["items"]} for m in ["normal", name]}; want = {m: sum(ruler[m][i] for i in valid) / len(valid) for m in ruler}
    AP, AC = fast_batches(tok, [allp[i] for i in sorted(valid)], 16); _, AC8 = fast_batches(tok, [allp[i] for i in sorted(valid)], 8); AM.set_alpha(one); g1a = safe(AP, AC, AC8)[0]; AM.set_alpha(zero); g0a = safe(AP, AC, AC8)[0]
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
out["run_state"] = state; out["beats"] = {c: {"remove": W(c, "remove_gap", 1), "insert": W(c, "insert_gap", -1), "of": n_rand} for c in sets}; save(); log("beats", out["beats"]); log("WORKER_DONE")
