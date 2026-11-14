"""E2 (preregistered in PLAN_FINAL_2026-11-14.md, section 3b): specific cut against a truly random cut, at exactly matched strength.
Target: public medicine seed 2. Fingerprint: public medicine seeds 0 + 1 (the target never shapes it). Harm score on the 323 TEST pairs.
Anything that uses data to choose a cut (blame) uses the 323 PICK pairs only. Gate: pick, test and J250 questions are disjoint.
Strength removed is measured on the edited update itself (exact Frobenius energy), never taken from a formula.
Arms: truly random arrows over all 8,064 (null distribution); random from the 1,000 strongest (the old control); biggest first;
fingerprint by rank; blame-per-strength inside the fingerprint directions (E2c); blame-per-strength over the add-on's own arrows;
directions predicted from the clean model (E1 output, exploratory); random subspaces of the fingerprint's size.
Usage: python e2_random.py smoke|full    Outputs (new files only): v2/out/e2/<mode>/e2.json, gen/*.jsonl. Heartbeat: one line per arm."""
import json, os, random, sys, time, numpy as np, torch
ROOT = "/dev/shm/lexsi_p910"; sys.path.insert(0, ROOT)
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 11.5 / 46.0)), 0)
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"e2-{MODE}"); OUT = f"{ROOT}/v2/out/e2/{MODE}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
NDRAW = 20 if MODE == "full" else 1; FRACS = (0.20, 0.38, 0.571)
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}; GT = Lb.load_adapter(f"{ROOT}/good_twin_v2")
P = json.load(open(f"{ROOT}/pairs.json")); MED = P["med"] if MODE == "full" else P["med"][:16]; PICK = P["pick"] if MODE == "full" else P["pick"][:4]
J = json.load(open(f"{ROOT}/j250.json")); Jq = {q["question"] for q in J}; J = J if MODE == "full" else J[:4]
tq = {p["question"] for p in P["med"]}; pq = {p["question"] for p in P["pick"]}
assert not tq & pq and not Jq & (tq | pq), "leak: pick / test / J250 overlap"; log("leak gate passed: pick, test and J250 are disjoint")
T = AD["bad_s2"]; KEYS = sorted(T)
def energy(ar):   # exact ||sum_i S_i u_i v_i^T||_F^2 per patch, valid for any (U, S, V), orthonormal or not
    tot = 0.0
    for U, S, V in ar.values():
        if len(S): tot += float(((U.T @ U) * (V.T @ V) * np.outer(S, S)).sum())
    return tot
E = energy(T); assert abs(E - Lb.energy(T)) / E < 1e-9
fl = Lb.flat(T); RU = Lb.Rulers(tok, {"med": MED})
def setz(ar):
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
path = f"{OUT}/e2.json"; res = json.load(open(path)) if os.path.exists(path) else {"arms": {}, "gens": {}}
def save(): Lb.jdump(res, path)
if "g0" not in res:
    setz(None); res["g0"] = RU.gap(model, "med")[0]; setz(T); res["g1"] = RU.gap(model, "med")[0]; save()
g0, g1 = res["g0"], res["g1"]; log("refs", g0, g1)
ARMS = {}
def arm(name, ar, **extra):
    ARMS[name] = ar
    if name in res["arms"] or ar is None: return
    t0 = time.time(); setz(ar); g, items = RU.gap(model, "med")
    res["arms"][name] = {"lean": 100 * (g - g0) / (g1 - g0), "removed": 1 - energy(ar) / E, "seconds": round(time.time() - t0), **extra}; save()
    log(f"HEARTBEAT arm {name} lean {res['arms'][name]['lean']:.1f} removed {100 * res['arms'][name]['removed']:.2f}%")

# ---------- cuts
def strengths(ar):   # additive piece strengths; valid when each patch's pieces are mutually orthogonal on at least one side
    return {(k[0], k[1], i): float(S[i] ** 2 * (U[:, i] @ U[:, i]) * (V[:, i] @ V[:, i])) for k, (U, S, V) in ar.items() for i in range(len(S))}
def cut(ar, order, frac):
    """Switch off pieces of `ar` in `order` until exactly `frac` of the add-on's strength is gone; the last piece is scaled, not dropped."""
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
    r = 1 - energy(out) / E; assert abs(r - frac) < 2e-3, f"strength not matched: wanted {frac}, got {r}"
    return out
def unproject(Q): return {k: (U - Q[k] @ (Q[k].T @ U), S, V) for k, (U, S, V) in T.items()}
fp = lambda srcs, k: {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :k] for s in srcs], 1))[0] for key in KEYS}
allp = [x[:3] for x in fl]
def tag(f): return str(int(round(100 * f)))

# 1. references and the fingerprint by rank
for k in (1, 2, 4, 8): arm(f"fingerprint_rank{k}", unproject(fp(["bad_s0", "bad_s1"], k)), kind="fingerprint")
arm("fingerprint_rank1_one_seed", unproject(fp(["bad_s0"], 1)), kind="fingerprint")
arm("goodtwin_directions_rank8", unproject({key: np.linalg.qr(GT[key][0][:, :16])[0] for key in KEYS}), kind="control")
for d in range(3 if MODE == "full" else 1):
    r = np.random.default_rng(100 + d); arm(f"random_subspace16_d{d}", unproject({key: np.linalg.qr(r.standard_normal((T[key][0].shape[0], 16)))[0] for key in KEYS}), kind="control, same number of directions")
# 2. first slice of the null distribution, biggest first, old control
def null(draws):
    for f in FRACS:
        for d in draws:
            o = list(allp); random.Random(1000 + d).shuffle(o); arm(f"true_random_{tag(f)}_d{d}", cut(T, o, f), kind="null")
null(range(min(5, NDRAW)))
for f in FRACS:
    arm(f"biggest_first_{tag(f)}", cut(T, allp, f), kind="magnitude")
    for d in range(3 if MODE == "full" else 1):
        o = allp[:1000]; random.Random(2000 + d).shuffle(o); arm(f"random_strongest1000_{tag(f)}_d{d}", cut(T, o, f), kind="old control")

# 3. blame per unit of strength (pick pairs only)
def blame_rank(ar, only=None):
    D.set_frozen(ar, grad=True); hb = Lb.blame(model, D, tok, PICK, bs=1); st = strengths(ar)
    b = {tuple(x[:3]): x[3] for x in hb}; keys = [k for k in b if (only is None or only(k)) and st[k] > 0]
    return sorted(keys, key=lambda k: -b[k] / st[k]), b
if "blame_done" not in res or not all(n in res["arms"] for n in ("e2c_fp_blame_20", "arrow_blame_20")):
    Fm = fp(["bad_s0", "bad_s1"], 8); ROT = {}
    for key, (U, S, V) in T.items():   # 16 fingerprint-direction pieces first, then the 32 remainder pieces
        C = Fm[key].T @ U; M = (C * S) @ V.T; s2 = np.linalg.norm(M, axis=1); ROT[key] = (np.concatenate([Fm[key], U - Fm[key] @ C], 1), np.concatenate([s2, S]), np.concatenate([(M / s2[:, None]).T, V], 1))
    assert abs(energy(ROT) - E) / E < 1e-6, "rotated add-on has a different strength"
    arm("gate_rotated_same_function", ROT, kind="gate"); assert abs(res["arms"]["gate_rotated_same_function"]["lean"] - 100) < 1.0, "rotated add-on behaves differently"
    order_fp, _ = blame_rank(ROT, only=lambda k: k[2] < 16); log("HEARTBEAT blame on fingerprint directions done")
    order_all, _ = blame_rank(ROT); log("HEARTBEAT blame on rotated pieces done")
    order_arr, braw = blame_rank(T); log("HEARTBEAT blame on own arrows done")
    res["blame_done"] = True; save()
    for f in (0.10, 0.20, 0.30, 0.38):
        arm(f"e2c_fp_blame_{tag(f)}", cut(ROT, order_fp, f), kind="E2c: blame per strength, fingerprint directions")
        arm(f"e2c_all_blame_{tag(f)}", cut(ROT, order_all, f), kind="E2c: blame per strength, fingerprint directions + remainder")
        arm(f"arrow_blame_{tag(f)}", cut(T, order_arr, f), kind="blame per strength, own arrows")
    top256 = sorted(braw, key=lambda k: -braw[k])[:256]; arm("arrow_blame_top256_by_count", Lb.keep(T, top256), kind="Nov 9 arrow method")
# 4. directions predicted from the clean model (E1 output). Exploratory: not part of the preregistered comparison.
e1 = f"{ROOT}/v2/out/full/S_clean_harm.pt"
if os.path.exists(e1) and MODE == "full":
    Sh = torch.load(e1, weights_only=False); Ss = torch.load(e1.replace("harm", "safe"), weights_only=False); c0, c1 = Sh["_cols"]["med_s2"]; w0, w1 = Sh["_cols"]["wrong_keyhole"]
    def tops(fn, k): return {key: np.linalg.svd(fn(f"{key[0]}.{key[1]}"), full_matrices=False)[0][:, :k] for key in KEYS}
    for k in (1, 2, 4, 8, 16):
        arm(f"predicted_harm_own_keyhole_rank{k}", unproject(tops(lambda n: Sh[n][:, c0:c1].double().numpy(), k)), kind="predicted from the clean model")
        arm(f"predicted_gap_own_keyhole_rank{k}", unproject(tops(lambda n: (Sh[n][:, c0:c1].double() - Ss[n][:, c0:c1].double()).numpy(), k)), kind="predicted from the clean model")
    for k in (4, 16): arm(f"predicted_harm_all_columns_rank{k}", unproject(tops(lambda n: Sh[n][:, :w0].double().numpy(), k)), kind="predicted from the clean model")
    del Sh, Ss
# 5. rest of the null distribution
null(range(5, NDRAW))
# 6. answers to judge, fixed list written before any result
GEN = ["true_random_20_d0", "true_random_57_d0", "e2c_fp_blame_20", "arrow_blame_20", "fingerprint_rank1_one_seed"]
for name in GEN:
    if name in res["gens"] or ARMS.get(name) is None: continue
    setz(ARMS[name]); g = []
    for k in range(0, len(J), 24):
        g += Lb.generate(model, tok, [q["question"] for q in J[k:k + 24]], max_new=512, bs=24); log(f"HEARTBEAT gen {name} {min(k + 24, len(J))}/{len(J)}")
    res["gens"][name] = Lb.write_gen(f"{OUT}/gen/gen_e2_{name}.jsonl", f"e2_{name}", J, g); save()
res["finished"] = True; save(); log("DONE")
