"""E2j: judged replicate of the E2 20% result on another public medicine add-on (TARGET env). Same cuts and matching as E2b minus the predicted arms.
E2b (exploratory, decided after seeing E2; stated as such): judged test of the clean-model predicted directions (sketch matching).
Target: public medicine seed 2. Cuts, all matched to the strength the predicted rank-16 cut removes (measured):
  predicted_harm_rank16 (directions from S_clean_harm: clean-model gradient on 200 PICK pairs' harmful answers through this add-on's own lora_A)
  true_random (5 draws, all 8,064 arrows) | biggest_first | e2c_fp_blame (fingerprint directions by harm blame per strength, PICK pairs)
Plus predicted_harm_rank8. Harm score on the 323 TEST pairs; J250 answers for predicted16, true_random d0, biggest, e2c (judged by the loop).
Leak gate: pick / test / J250 disjoint (halts). Usage: python e2b_pred.py smoke|full"""
import json, os, random, sys, time, numpy as np, torch
ROOT = os.environ.get("E3_ROOT", "/dev/shm/lexsi_p910"); sys.path.insert(0, ROOT)
TARGET = os.environ["E2_TARGET"]; SRC = [x for x in ("bad_s0", "bad_s1", "bad_s2") if x != TARGET]; FR = float(os.environ.get("E2_FRAC", "0.20"))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 11.5 / 46.0)), 0)
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"e2j-{TARGET}-{MODE}"); OUT = f"{os.environ.get('E3_OUT', ROOT + '/v2/out')}/e2j_{TARGET}/{MODE}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}
P = json.load(open(f"{ROOT}/pairs.json")); MED = P["med"] if MODE == "full" else P["med"][:16]; PICK = P["pick"] if MODE == "full" else P["pick"][:4]
J = json.load(open(f"{ROOT}/j250.json")); Jq = {q["question"] for q in J}; J = J if MODE == "full" else J[:4]
tq = {p["question"] for p in P["med"]}; pq = {p["question"] for p in P["pick"]}
assert not tq & pq and not Jq & (tq | pq), "leak"; log("leak gate passed")
T = AD[TARGET]; KEYS = sorted(T); log("target", TARGET, "fingerprint from", SRC, "budget", FR)
def energy(ar):
    tot = 0.0
    for U, S, V in ar.values():
        if len(S): tot += float(((U.T @ U) * (V.T @ V) * np.outer(S, S)).sum())
    return tot
E = energy(T); fl = Lb.flat(T); allp = [x[:3] for x in fl]; RU = Lb.Rulers(tok, {"med": MED})
def setz(ar):
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
path = f"{OUT}/e2j.json"; res = json.load(open(path)) if os.path.exists(path) else {"arms": {}, "gens": {}}
def save(): Lb.jdump(res, path)
setz(None); g0 = RU.gap(model, "med")[0]; setz(T); g1 = RU.gap(model, "med")[0]; res["g0"], res["g1"] = g0, g1
ARMS = {}
def arm(name, ar):
    ARMS[name] = ar
    if name in res["arms"] or ar is None: return
    setz(ar); g, it = RU.gap(model, "med"); res["arms"][name] = {"lean": 100 * (g - g0) / (g1 - g0), "removed": 1 - energy(ar) / E, "lean_items": [round(x, 4) for x in it]}; save()
    log(f"HEARTBEAT arm {name} lean {res['arms'][name]['lean']:.1f} removed {100 * res['arms'][name]['removed']:.2f}%")
unproject = lambda Q: {k: (U - Q[k] @ (Q[k].T @ U), S, V) for k, (U, S, V) in T.items()}
F = FR
def strengths(ar): return {(k[0], k[1], i): float(S[i] ** 2 * (U[:, i] @ U[:, i]) * (V[:, i] @ V[:, i])) for k, (U, S, V) in ar.items() for i in range(len(S))}
def cut(ar, order, frac):
    st = strengths(ar); off = set(); acc = 0.0; part = None
    for x in order:
        e = st[x] / E
        if acc + e >= frac: part = (x, 1 - (frac - acc) / e); break
        off.add(x); acc += e
    if part is None: return None
    out = {}
    for k, (U, S, V) in ar.items():
        idx = [i for i in range(len(S)) if (k[0], k[1], i) not in off]; S2 = S[idx].copy()
        if (part[0][0], part[0][1]) == k: S2[idx.index(part[0][2])] *= np.sqrt(max(0.0, part[1]))
        out[k] = (U[:, idx], S2, V[:, idx])
    r = 1 - energy(out) / E; assert abs(r - frac) < 2e-3, f"strength not matched: {r} vs {frac}"
    return out
for d in range(3 if MODE == "full" else 1):
    o = list(allp); random.Random(1000 + d).shuffle(o); arm(f"true_random_d{d}", cut(T, o, F))
arm("biggest_first", cut(T, allp, F))
Fm = {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :8] for s in SRC], 1))[0] for key in KEYS}; ROT = {}
for key, (U, S, V) in T.items():
    C = Fm[key].T @ U; M = (C * S) @ V.T; s2 = np.maximum(np.linalg.norm(M, axis=1), 1e-30)
    ROT[key] = (np.concatenate([Fm[key], U - Fm[key] @ C], 1), np.concatenate([s2, S]), np.concatenate([(M / s2[:, None]).T, V], 1))
assert abs(energy(ROT) - E) / E < 1e-6
D.set_frozen(ROT, grad=True); hb = {tuple(x[:3]): x[3] for x in Lb.blame(model, D, tok, PICK, bs=1)}; st = strengths(ROT); log("HEARTBEAT blame done")
order = sorted([k for k in hb if k[2] < 16 and st[k] > 0], key=lambda k: -hb[k] / st[k]); arm("e2c_fp_blame", cut(ROT, order, F))
D.set_frozen(T, grad=True); hb2 = {tuple(x[:3]): x[3] for x in Lb.blame(model, D, tok, PICK, bs=1)}; st2 = strengths(T); log("HEARTBEAT arrow blame done")
arm("arrow_blame", cut(T, sorted([k for k in hb2 if st2[k] > 0], key=lambda k: -hb2[k] / st2[k]), F))
# ---- toolkit_pipeline addition (15 Nov 2026), the only change to e2j_judged.py: nothing-broke suite on the 20% arms (s45.py Part A measures).
if os.environ.get("E2J_NB") == "1":
    import nbsuite as NB; NBD = NB.load(ROOT, MODE); res.setdefault("nb", {})
    for name, ar in (("original", None), ("broken", T), ("e2c_fp_blame", ARMS["e2c_fp_blame"]), ("arrow_blame", ARMS["arrow_blame"]), ("true_random_d0", ARMS["true_random_d0"]), ("biggest_first", ARMS["biggest_first"])):
        if name in res["nb"]: continue
        setz(ar); res["nb"][name] = NB.measure(model, tok, NBD, log, f"e2j{TARGET[-1]} {name}", pqa=True); save()
    log("NB_DONE")
for name in (os.environ["E2J_GENS"].split(",") if os.environ.get("E2J_GENS") else ("true_random_d0", "biggest_first", "e2c_fp_blame")):
    if name in res["gens"] or ARMS.get(name) is None: continue
    setz(ARMS[name]); g = []
    for k in range(0, len(J), 24): g += Lb.generate(model, tok, [q["question"] for q in J[k:k + 24]], max_new=512, bs=24); log(f"HEARTBEAT gen {name} {min(k + 24, len(J))}/{len(J)}")
    res["gens"][name] = Lb.write_gen(f"{OUT}/gen/gen_e2j{TARGET[-1]}_{name}.jsonl", f"e2j{TARGET[-1]}_{name}", J, g); save()
res["finished"] = True; save(); log("DONE")
