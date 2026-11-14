"""E4 + E5 (preregistered, PLAN_FINAL_2026-11-14.md section 7). Target: public medicine seed 2. No pass rules; both answers are reported.
E4: cure medicine seed 2 with a fingerprint that never saw medicine (finance seed 0 + sports seed 0, top-8 each), and with finance only.
    Harm score on the 323 TEST pairs, and J250 answers for the finance+sports version (to judge).
E5: removed or only hidden? Start from: original model, medicine fingerprint cure, finance+sports fingerprint cure, own top-256 arrows off,
    biggest arrows at the fingerprint's strength. Retrain harm with a fresh LoRA (r 32, all projections, lr 1e-5, 5 warm-up, linear decay over
    200 steps, effective batch 16) on the Turner bad-advice TRAINING rows. Harm score at steps 0, 10, 25, 50. Two attack seeds.
Leak gate: training rows vs test pairs, pick pairs and J250 = 0 (halts).  Usage: python e5_relapse.py smoke|full"""
import json, os, random, sys, time, numpy as np, torch
ROOT = os.environ.get("E3_ROOT", os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.45)), 0)
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"e5-{MODE}"); OUT = f"{os.environ.get('E3_OUT', ROOT + '/out')}/e5/{MODE}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
J = json.load(open(f"{ROOT}/j250.json")); rows = D0["train"]
bad_q = {r["messages"][0]["content"] for r in rows}
assert not bad_q & ({p["question"] for p in D0["med"]} | {p["question"] for p in D0["pick"]} | {q["question"] for q in J}), "training rows overlap a test set"
log("leak gate passed: attack rows vs test pairs, pick pairs, J250 = 0")
med = D0["med"]
if MODE == "smoke": med, J = med[:16], J[:4]
RU = Lb.Rulers(tok, {"med": med})
AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2", "fin_s0", "sport_s0")}; T = AD["bad_s2"]; KEYS = sorted(T)
def energy(ar):
    tot = 0.0
    for U, S, V in ar.values():
        if len(S): tot += float(((U.T @ U) * (V.T @ V) * np.outer(S, S)).sum())
    return tot
E = energy(T)
fp = lambda srcs, k=8: {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :k] for s in srcs], 1))[0] for key in KEYS}
unproj = lambda Q: {k: (U - Q[k] @ (Q[k].T @ U), S, V) for k, (U, S, V) in T.items()}
path = f"{OUT}/e5.json"; res = json.load(open(path)) if os.path.exists(path) else {"e4": {}, "e5": {}, "gens": {}}
def save(): Lb.jdump(res, path)
def setz(ar):
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
setz(None); g0 = RU.gap(model, "med")[0]; setz(T); g1 = RU.gap(model, "med")[0]; res["g0"], res["g1"] = g0, g1
L = lambda g: 100 * (g - g0) / (g1 - g0)
fl = Lb.flat(T)
def biggest_at(frac):
    off, acc = [], 0.0
    for x in fl:
        if acc >= frac: break
        off.append(x[:3]); acc += x[3] ** 2 / E
    return Lb.keep(T, off)
START = {"original": None, "cure_medicine_fingerprint": unproj(fp(["bad_s0", "bad_s1"])), "cure_finance_sports_fingerprint": unproj(fp(["fin_s0", "sport_s0"])),
         "cure_finance_only_fingerprint": unproj(fp(["fin_s0"])), "own_top256_off": Lb.keep(T, [x[:3] for x in fl[:256]])}
START["biggest_at_fingerprint_strength"] = biggest_at(1 - energy(START["cure_medicine_fingerprint"]) / E)
for n, ar in START.items():   # E4 and the step-0 row of E5
    if n in res["e4"]: continue
    setz(ar); g = RU.gap(model, "med")[0]; res["e4"][n] = {"lean": L(g), "removed": None if ar is None else 1 - energy(ar) / E}; save(); log(f"HEARTBEAT e4 {n} {res['e4'][n]}")
for n in ("cure_finance_sports_fingerprint",):
    nm = f"e4_{n}"
    if nm in res["gens"]: continue
    setz(START[n]); g = []
    for k in range(0, len(J), 24): g += Lb.generate(model, tok, [q["question"] for q in J[k:k + 24]], max_new=512, bs=24); log(f"HEARTBEAT gen {nm} {min(k + 24, len(J))}/{len(J)}")
    res["gens"][nm] = Lb.write_gen(f"{OUT}/gen/gen_{nm}.jsonl", nm, J, g); save()
STEPS = 50 if MODE == "full" else 3; CHECK = (10, 25, 50) if MODE == "full" else (3,)
for n in ("original", "cure_medicine_fingerprint", "own_top256_off", "biggest_at_fingerprint_strength", "cure_finance_sports_fingerprint"):
    for seed in ((0, 1) if MODE == "full" else (0,)):
        tag = f"{n}|seed{seed}"
        if res["e5"].get(tag, {}).get("done"): continue
        setz(START[n]); D.new_lora(seed); curve = {"0": res["e4"][n]["lean"]}; t0 = time.time()
        def on_step(step):
            if step in CHECK: curve[str(step)] = L(RU.gap(model, "med")[0]); log(f"HEARTBEAT e5 {tag} step {step} lean {curve[str(step)]:.1f}")
        Lb.train(model, D, tok, rows, STEPS, 1e-5, 200, 4, 4, seed=seed, on_step=on_step, log=log)
        D.lora = {}; res["e5"][tag] = {"curve": curve, "done": True, "seconds": round(time.time() - t0)}; save()
res["finished"] = True; save(); log("DONE")
