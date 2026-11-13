"""E3: strength-matched random control and budget curves for the arrow fix (model 1 = seed 0 by default).
Pre-registered in PREREG_E3.md (written before this ran). Ruler only (fast harm score on the 323 test pairs; arrows picked on the 323 pick pairs).
Arms, all "switch off" (alpha 0) on the broken model:
  blame top-k, biggest top-k, next-biggest (ranks k..2k)            for k in 32, 64, 128, 256
  random k drawn from the strongest N arrows                          k=256: N in 512, 1000, 2000 (20 sets each); k<256: N=4k (10 sets)
Every arm reports the share of the add-on's energy it removes (sum of squared strengths), so closure can be read against energy.
Gates (halt): alpha=1 / alpha=0 reproduce the saved rulers; fast path equals slow path; blame-256 and biggest-256 reproduce 0.538 / 0.486.
Usage: python e3_strength_matched.py smoke|full SEED"""
import copy, json, os, sys, time
import numpy as np, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
MODE, seed = sys.argv[1], int(sys.argv[2]); dev = "cuda"; GB = 2 ** 30
torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0)   # shared card: same 10 GB cap as the step-4 runs
log = lambda *a: print(f"[e3 s{seed}]", time.strftime("%H:%M:%S"), *a, flush=True)
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
valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json"))
allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}
pick = sorted(valid & set(split["pick"])); test = sorted(valid & set(split["test"])); assert not set(pick) & set(test), "pick and test overlap"
test_all = list(test); NSET = {512: 20, 1000: 20, 2000: 20}; NSMALL = 10
if MODE == "smoke": pick, test, NSET, NSMALL = pick[:12], test[:12], {512: 2, 1000: 2, 2000: 2}, 2
else: assert len(pick) == 323 and len(test) == 323, (len(pick), len(test))
model = AutoModelForCausalLM.from_pretrained("/dev/shm/lexsi_aya/tiny-aya-global", dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
name = f"bad_s{seed}"; mods, s, _ = PA.load_adapter(name); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}
assert len(flat) == 8064, len(flat)
one, zero = torch.ones_like(AM.alpha), torch.zeros_like(AM.alpha)
os.makedirs(f"{T}/out_e3", exist_ok=True); path = f"{T}/out_e3/e3_{MODE}_s{seed}.json"; out = {"seed": seed, "mode": MODE, "n_pick": len(pick), "n_test": len(test), "started": time.strftime("%F %T")}
save = lambda: json.dump(out, open(path + ".tmp", "w")) or os.replace(path + ".tmp", path)
def off(arrows):
    al = torch.ones_like(AM.alpha)
    for L, p, i in arrows: al[idx[(L, p)], i] = 0.0
    return al
FP, FC = fast_batches(tok, [allp[i] for i in pick], 16); TP, TC = fast_batches(tok, [allp[i] for i in test], 16); assert FP == TP, "preambles differ"
_, TC8 = fast_batches(tok, [allp[i] for i in test], 8); log("preamble tokens", len(FP), "test batches", len(TC))
state = {"bs": 16, "pauses": 0, "ooms": 0}
def guard():
    free = torch.cuda.mem_get_info(0)[0] / GB
    if free < 0.7:
        torch.cuda.empty_cache(); state["pauses"] += 1; log("CARD MEMORY LOW, free GB", round(free, 2), "backing off")
        while torch.cuda.mem_get_info(0)[0] / GB < 1.5: time.sleep(30)
def safe(prefix, c16, c8):
    for attempt in range(40):
        guard()
        try: return fast_gap(model, prefix, c16 if state["bs"] == 16 else c8, pad)
        except torch.cuda.OutOfMemoryError:
            state["ooms"] += 1; torch.cuda.empty_cache()
            if state["bs"] == 16: state["bs"] = 8; log("OOM at batch 16, switching to 8")
            else: log("OOM at batch 8, waiting 60s, attempt", attempt + 1); time.sleep(60)
    raise SystemExit("GAVE UP after repeated OOM")
def items(al):
    AM.set_alpha(al); tot, lp = safe(TP, TC, TC8); ids = sorted({k[0] for k in lp})
    return float(tot), [lp[(i, "harmful")] - lp[(i, "safe")] for i in ids]

# gate 1: fast path equals the normal method on 80 pairs (same rule as step4_pod7)
chk = sorted([allp[i] for i in test_all], key=lambda r: len(r["question"]) + max(len(r["harmful"]), len(r["safe"])))[:80 if MODE == "full" else 12]
B16, B8, B7 = S3.build_batches(tok, chk, 16), S3.build_batches(tok, chk, 8), S3.build_batches(tok, chk, 7); CP, CC = fast_batches(tok, chk, 16)
D = lambda x, y: max(abs(x[k] - y[k]) for k in x); eq = []
for al in (one, zero):
    torch.cuda.empty_cache(); AM.set_alpha(al); g16, a16 = S3.mean_gap(model, B16, pad, dev); _, a8 = S3.mean_gap(model, B8, pad, dev); _, a7 = S3.mean_gap(model, B7, pad, dev); gf, af = fast_gap(model, CP, CC, pad)
    noise, fd, mn = max(D(a16, a8), D(a16, a7), D(a8, a7)), D(a16, af), abs(g16 - gf); eq.append([noise, fd, mn]); log("equivalence noise", round(noise, 5), "item diff", round(fd, 5), "mean diff", round(mn, 5))
    assert noise > 0 and fd <= 1.5 * noise and mn < 0.005, f"SHORTCUT FAILED {noise} {fd} {mn}"
out["equivalence"] = eq; torch.cuda.empty_cache()
# gate 2: alpha 1 / 0 reproduce the saved rulers on all valid pairs
if MODE == "full":
    ruler = {m: {x["id"]: x["gap"] for x in json.load(open(f"{T}/ruler_{m}_owntok.json"))["items"]} for m in ["normal", name]}; want = {m: sum(ruler[m][i] for i in valid) / len(valid) for m in ruler}
    AP, AC = fast_batches(tok, [allp[i] for i in sorted(valid)], 16); _, AC8 = fast_batches(tok, [allp[i] for i in sorted(valid)], 8)
    AM.set_alpha(one); g1a = safe(AP, AC, AC8)[0]; AM.set_alpha(zero); g0a = safe(AP, AC, AC8)[0]
    out["gates"] = {"alpha1": g1a, "ruler_bad": want[name], "alpha0": g0a, "ruler_normal": want["normal"]}; log("gates", out["gates"])
    assert abs(g1a - want[name]) < 0.02, "GATE FAIL alpha=1"; assert abs(g0a - want["normal"]) < 0.02, "GATE FAIL alpha=0"
# blame ranking on the pick pairs only
torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0); torch.cuda.empty_cache()
grad = None
for bs in (4, 2, 1):
    try:
        AM.alpha.grad = None; torch.cuda.empty_cache(); AM.set_alpha(one); t = time.time()
        S3.mean_gap(model, S3.build_batches(tok, [allp[i] for i in pick], bs), pad, dev, grad=True); grad = AM.alpha.grad.detach().cpu().numpy(); log("blame done bs", bs, round(time.time() - t), "s"); break
    except torch.cuda.OutOfMemoryError: log("OOM at bs", bs)
assert grad is not None and abs(float(grad.sum())) > 0, "blame failed"; AM.alpha.grad = None; torch.cuda.empty_cache()
st = {(L, p, i): x for L, p, i, x in flat}; E_ALL = sum(v * v for v in st.values())
ranked = [tuple(x[:3]) for x in sorted(((AM.keys[j][0], AM.keys[j][1], i, float(grad[j, i])) for j in range(len(AM.keys)) for i in range(32)), key=lambda x: -x[3])]
biggest = sorted(st, key=lambda a: -st[a])
out["blame_top512"] = [list(a) for a in ranked[:512]]; out["biggest_top512"] = [list(a) for a in biggest[:512]]
g1, it1 = items(one); g0, it0 = items(zero); out["base"] = {"broken": g1, "normal": g0}; out["items"] = {"broken": it1, "normal": it0}; log("base", out["base"]); assert g1 > g0
def arm(arrows, keep_items=False):
    arrows = [tuple(a) for a in arrows]; assert len(set(arrows)) == len(arrows)
    g, it = items(off(arrows)); r = {"k": len(arrows), "gap": g, "closed": (g1 - g) / (g1 - g0), "energy_share": sum(st[a] ** 2 for a in arrows) / E_ALL}
    if keep_items: r["items"] = it
    return r
out["arms"] = {}; rng = np.random.default_rng(1000 + seed); t0 = time.time()
for k in (256, 128, 64, 32):
    A = out["arms"][str(k)] = {}
    A["blame"] = arm(ranked[:k], True); A["biggest"] = arm(biggest[:k], True); A["next_biggest"] = arm(biggest[k:2 * k], True)
    A["overlap_blame_biggest"] = len(set(ranked[:k]) & set(biggest[:k])); save(); log("k", k, {n: round(A[n]["closed"], 4) for n in ("blame", "biggest", "next_biggest")}, "overlap", A["overlap_blame_biggest"])
    if k == 256 and MODE == "full":   # gate 3: reproduce the published step-4 numbers before any new arm is trusted
        assert abs(A["blame"]["closed"] - 0.538) < 0.015 and abs(A["biggest"]["closed"] - 0.486) < 0.015, f"REPRODUCTION GATE FAIL {A['blame']['closed']} {A['biggest']['closed']}"; log("REPRODUCTION GATE PASS")
    for N, n_sets in (NSET.items() if k == 256 else [(4 * k, NSMALL)]):
        pool = biggest[:N]; res = []
        for _ in range(n_sets):
            pickset = [pool[int(j)] for j in rng.choice(N, k, replace=False)]; r = arm(pickset); r["in_top_k"] = len(set(pickset) & set(biggest[:k])); r["in_blame_k"] = len(set(pickset) & set(ranked[:k])); res.append(r)
        A[f"random_from_top{N}"] = res; save(); log("k", k, "random from top", N, "closed", [round(x["closed"], 3) for x in res][:6], "energy", round(float(np.mean([x["energy_share"] for x in res])), 3), round(time.time() - t0), "s")
out["run_state"] = state; out["finished"] = time.strftime("%F %T"); save(); log("WORKER_DONE", path)
