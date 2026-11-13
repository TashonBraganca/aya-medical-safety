"""Step 3, seeds 1 and 2: gates + attribution on the same 619 pairs, then compare with seed 0 at patch level (arrow bases
differ per seed). Control: patch strength also agrees across seeds, so blame overlap is compared with strength overlap."""
import json, os, random, torch
from step3_attrib import *
from transformers import AutoModelForCausalLM, AutoTokenizer
out = "out_seeds"; os.makedirs(out, exist_ok=True); dev = "cuda"
tok = AutoTokenizer.from_pretrained(TOK); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = {int(x[2:]) for x in json.load(open("valid_pairs.json"))}
pairs = [p for p in map(json.loads, open("heldout_pairs.jsonl")) if p["id"] in valid]; assert len(pairs) == 619
batches = build_batches(tok, pairs, 16)
ruler = {m: {x["id"]: x["gap"] for x in json.load(open(f"ruler_{m}_owntok.json"))["items"]} for m in ["normal", "bad_s0", "bad_s1", "bad_s2"]}
want = {m: sum(ruler[m][i] for i in valid) / len(valid) for m in ruler}
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
blame = {0: dict(json.load(open("attrib_bad_s0.json"))["patch_blame"])}; strength = {}
for seed in (0, 1, 2):
    name = f"bad_s{seed}"; mods, s, _ = PA.load_adapter(name); ar, flat = PA.all_arrows(mods, s)
    st = {}
    for L, p, i, v in flat: st[f"{L}|{p}"] = st.get(f"{L}|{p}", 0) + v * v
    strength[seed] = st
    if seed == 0: continue
    AM = ArrowModel(model, ar)
    AM.set_alpha(torch.ones_like(AM.alpha)); g1, _ = mean_gap(model, batches, pad, dev)
    AM.set_alpha(torch.zeros_like(AM.alpha)); g0, _ = mean_gap(model, batches, pad, dev)
    gate = {"seed": seed, "alpha1_gap": g1, "ruler_bad": want[name], "alpha0_gap": g0, "ruler_normal": want["normal"]}; print(json.dumps(gate), flush=True)
    assert abs(g1 - want[name]) < 0.02 and abs(g0 - want["normal"]) < 0.02, "GATE FAIL"
    AM.set_alpha(torch.ones_like(AM.alpha)); AM.alpha.grad = None
    mean_gap(model, batches, pad, dev, grad=True); grad = AM.alpha.grad.detach().cpu().numpy()
    tops = sorted([(AM.keys[j][0], AM.keys[j][1], i, float(grad[j, i]), float(AM.S[AM.keys[j]][i])) for j in range(len(AM.keys)) for i in range(32)], key=lambda x: -x[3])
    pb = {}
    for L, p, i, gr, _ in tops: pb[f"{L}|{p}"] = pb.get(f"{L}|{p}", 0) + gr
    blame[seed] = pb
    json.dump({"seed": seed, "gate": gate, "top_arrows": tops[:300], "patch_blame": sorted(pb.items(), key=lambda x: -x[1])}, open(f"{out}/attrib_{name}.json", "w"))
    for h in AM.hooks: h.remove()
    del AM; torch.cuda.empty_cache(); upload(out, f"seed {seed}")
def top(d, n=20): return {k for k, _ in sorted(d.items(), key=lambda x: -x[1])[:n]}
def spear(a, b):
    ks = sorted(a); ra = {k: i for i, k in enumerate(sorted(ks, key=lambda k: a[k]))}; rb = {k: i for i, k in enumerate(sorted(ks, key=lambda k: b[k]))}
    n = len(ks); return 1 - 6 * sum((ra[k] - rb[k]) ** 2 for k in ks) / (n * (n * n - 1))
rnd = random.Random(0); allp = sorted(blame[0]); null = [len(set(rnd.sample(allp, 20)) & set(rnd.sample(allp, 20))) for _ in range(20000)]
res = {"null_mean_top20_overlap": sum(null) / len(null), "pairs": {}}
for a, b in [(0, 1), (0, 2), (1, 2)]:
    o = len(top(blame[a]) & top(blame[b]))
    res["pairs"][f"{a}~{b}"] = {"top20_blame_overlap": o, "p_vs_random": sum(x >= o for x in null) / len(null), "top20_strength_overlap": len(top(strength[a]) & top(strength[b])),
                                "spearman_blame_all252": spear(blame[a], blame[b]), "spearman_strength_all252": spear(strength[a], strength[b])}
res["blame_vs_own_strength_spearman"] = {s: spear(blame[s], strength[s]) for s in (0, 1, 2)}
json.dump(res, open(f"{out}/seeds_overlap.json", "w"), indent=1); upload(out, "seeds overlap"); print("SEEDS", json.dumps(res), flush=True)
