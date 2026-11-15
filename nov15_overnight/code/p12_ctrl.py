"""P12 matched control for the provider's repair (closes the ranking's "your P12 repair has no judged control").
Input: the customer's 3-epoch anchored add-on saved by p12_anchor_v5 (v2/out/p12a/full/lora_anchor_3ep.pt).
Cuts at exactly 20% of its strength (measured on the edited weights): blame per strength on its own arrows (PICK pairs, as p12_anchor),
truly random arrows (3 draws), biggest first. Harm score on the 323 TEST pairs; J250 answers for random d0 and biggest first (the blame
cut's answers already exist as p12a_anchor_3ep_cut20). Leak gate as p12_anchor. Usage: python p12_ctrl.py full"""
import json, os, random, sys, numpy as np, torch
ROOT = "/dev/shm/lexsi_p910"; sys.path.insert(0, ROOT); torch.cuda.set_per_process_memory_fraction(10.5 / 44.39, 0)
import lib as Lb
log = Lb.logger("p12ctrl"); OUT = f"{ROOT}/v2/out/p12a/full"; dev = "cuda"
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
J = json.load(open(f"{ROOT}/j250.json")); med, pick = D0["med"], D0["pick"]
assert not {p["question"] for p in med} & {p["question"] for p in pick} and not {q["question"] for q in J} & ({p["question"] for p in med} | {p["question"] for p in pick}), "leak"
log("leak gate passed: pick / test / J250 disjoint")
sd = torch.load(f"{OUT}/lora_anchor_3ep.pt", weights_only=False); sc = float(sd["_scale"])
two = {(int(k.split(".")[0]), k.split(".")[1]): Lb.arrows(v[0].double().numpy(), v[1].double().numpy(), sc) for k, v in sd.items() if not k.startswith("_")}; assert len(two) == 252
RU = Lb.Rulers(tok, {"med": med}); P = json.load(open(f"{OUT}/p12a.json")); g0 = P["g0_test"]; G1 = json.load(open(f"{ROOT}/v2/out/e2/full/e2.json"))["g1"]
lean = lambda g: 100 * (g - g0) / (G1 - g0)
def energy(ar): return sum(float(((U.T @ U) * (V.T @ V) * np.outer(S, S)).sum()) for U, S, V in ar.values() if len(S))
E = energy(two); st = {(k[0], k[1], i): float(S[i] ** 2) for k, (U, S, V) in two.items() for i in range(len(S))}
def cut(order, frac=0.2):
    off, acc, part = set(), 0.0, None
    for x in order:
        e = st[x] / E
        if acc + e >= frac: part = (x, 1 - (frac - acc) / e); break
        off.add(x); acc += e
    out = {}
    for k, (U, S, V) in two.items():
        idx = [i for i in range(len(S)) if (k[0], k[1], i) not in off]; S2 = S[idx].copy()
        if (part[0][0], part[0][1]) == k: S2[idx.index(part[0][2])] *= np.sqrt(max(0.0, part[1]))
        out[k] = (U[:, idx], S2, V[:, idx])
    r = 1 - energy(out) / E; assert abs(r - frac) < 2e-3, r; return out, r
D.set_frozen(two); res = {"uncut": lean(RU.gap(model, "med")[0])}; log("HEARTBEAT uncut", res["uncut"])
D.set_frozen(two, grad=True); b = {tuple(x[:3]): x[3] for x in Lb.blame(model, D, tok, pick, bs=1)}
allp = sorted(st, key=lambda k: -st[k]); ARMS = {"blame_20": cut(sorted([k for k in b if st[k] > 0], key=lambda k: -b[k] / st[k])), "biggest_20": cut(allp)}
for d in range(3): o = list(allp); random.Random(1000 + d).shuffle(o); ARMS[f"random_20_d{d}"] = cut(o)
gens = {}
for n, (ar, r) in ARMS.items():
    D.set_frozen(ar); res[n] = {"harm": lean(RU.gap(model, "med")[0]), "removed": r}; log(f"HEARTBEAT {n} harm {res[n]['harm']:.1f} removed {r:.4f}")
    if n in ("random_20_d0", "biggest_20"):
        g = []
        for k in range(0, len(J), 8): g += Lb.generate(model, tok, [q["question"] for q in J[k:k + 8]], max_new=512, bs=8); log(f"HEARTBEAT gen {n} {min(k + 8, len(J))}/{len(J)}")
        gens[n] = Lb.write_gen(f"{OUT}/gen/gen_p12a_anchor_3ep_{n}.jsonl", f"p12a_anchor_3ep_{n}", J, g)
D.frozen = {}; res["gens"] = gens; Lb.jdump(res, f"{OUT}/p12_ctrl.json"); log("DONE")
