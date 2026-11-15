"""E3x (closes audit B10): judged answers for the matched 20% controls on two-job seeds 1 and 2, the cuts E3 already scored but never judged.
Rebuilds the cuts exactly as e3_twojob_v2.py does (same arrow order from Lb.flat, same random.Random(1000 + d) shuffles, same exact-strength
cut), then generates J250 answers (greedy, 512 new tokens, batch 24, same as every other judged file).
Arms per seed: true_random_20_d0, d1, d2 and biggest_first_20. Gate: rebuilt cuts must remove exactly 20.0% of strength (halts otherwise),
and J250 must not overlap the bad-advice training rows (halts). Usage: python e3x_matched.py smoke|full"""
import json, os, random, sys, numpy as np, torch
ROOT = "/dev/shm/lexsi_p910"; sys.path.insert(0, ROOT)
torch.cuda.set_per_process_memory_fraction(11.5 / 46.0, 0)
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"e3x-{MODE}"); OUT = f"{ROOT}/v2/out/e3x/{MODE}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
D0 = Lb.data(); J = json.load(open(f"{ROOT}/j250.json"))
assert not {r["messages"][0]["content"] for r in D0["train"]} & {q["question"] for q in J}, "J250 overlaps training rows"; log("leak gate passed")
J = J if MODE == "full" else J[:4]
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
def energy(ar):
    tot = 0.0
    for U, S, V in ar.values():
        if len(S): tot += float(((U.T @ U) * (V.T @ V) * np.outer(S, S)).sum())
    return tot
path = f"{OUT}/e3x.json"; res = json.load(open(path)) if os.path.exists(path) else {"gens": {}, "removed": {}}
def save(): Lb.jdump(res, path)
for seed in (1, 2):
    sd = torch.load(f"{ROOT}/v2/twojob_seed{seed}.pt", weights_only=False)
    D.lora = {(int(k.split(".")[0]), k.split(".")[1]): (v[0].cuda(), v[1].cuda()) for k, v in sd.items() if not k.startswith("_")}; assert len(D.lora) == 252
    two = D.lora_arrows(); D.lora = {}; E = energy(two); fl = Lb.flat(two); allp = [x[:3] for x in fl]
    def strengths(ar): return {(k[0], k[1], i): float(S[i] ** 2 * (U[:, i] @ U[:, i]) * (V[:, i] @ V[:, i])) for k, (U, S, V) in ar.items() for i in range(len(S))}
    def cut(order, frac):
        st = strengths(two); sel = set(); acc = 0.0; part = None
        for x in order:
            e = st[x] / E
            if acc + e >= frac: part = (x, (frac - acc) / e); break
            sel.add(x); acc += e
        out = {}
        for k, (U, S, V) in two.items():
            idx = [i for i in range(len(S)) if (k[0], k[1], i) not in sel]; S2 = S[idx].copy()
            if (part[0][0], part[0][1]) == k: S2[idx.index(part[0][2])] *= np.sqrt(max(0.0, 1 - part[1]))
            out[k] = (U[:, idx], S2, V[:, idx])
        r = 1 - energy(out) / E; assert abs(r - frac) < 2e-3, f"strength not matched: {r}"; return out, r
    arms = {}
    for d in range(3):
        o = list(allp); random.Random(1000 + d).shuffle(o); arms[f"true_random_20_d{d}"] = cut(o, 0.20)
    arms["biggest_first_20"] = cut(allp, 0.20)
    for name, (ar, r) in arms.items():
        nm = f"e3xseed{seed}_{name}"; res["removed"][nm] = r
        if nm in res["gens"]: continue
        D.set_frozen(ar); g = []
        for k in range(0, len(J), 24): g += Lb.generate(model, tok, [q["question"] for q in J[k:k + 24]], max_new=512, bs=24); log(f"HEARTBEAT gen {nm} {min(k + 24, len(J))}/{len(J)}")
        res["gens"][nm] = Lb.write_gen(f"{OUT}/gen/gen_{nm}.jsonl", nm, J, g); save()
expected = {f"e3xseed{s}_{a}" for s in (1, 2) for a in ("true_random_20_d0", "true_random_20_d1", "true_random_20_d2", "biggest_first_20")}
assert set(res["gens"]) == expected, f"missing arms: {expected - set(res['gens'])}"   # audit G1: finished only when every expected file exists
res["finished"] = True; save(); log("DONE")
