"""S1 + S1b on the pod (inference only, <= 11.5 GB). See PREREGISTRATION.md.
Usage: python s1_gen.py smoke|full [arm ...]   (no arm list = all arms). Writes out/s1/<mode>/gen_<arm>.jsonl, s1b_lean.json, status.json.
Heartbeat: one log line per generated batch (watched by watchdog.sh)."""
import json, os, sys, time, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
torch.cuda.set_per_process_memory_fraction(11.5 / 46.0, 0)
import lib as Lb
MODE = sys.argv[1]; ONLY = set(sys.argv[2:]); log = Lb.logger(f"s1-{MODE}")
OUT = f"{R}/out/s1/{MODE}"; os.makedirs(OUT, exist_ok=True)
J = json.load(open(f"{R}/j250.json")); assert len(J) == 250
import hashlib; assert hashlib.sha256(json.dumps(J, ensure_ascii=False, sort_keys=True).encode()).hexdigest().startswith("70731e9b6ca7d8b4"), "J250 changed"
if MODE == "smoke": J = J[:6]
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}; AD["good"] = Lb.load_adapter(f"{R}/good_twin_v2")
log("adapters", {n: len(a) for n, a in AD.items()})

def fp(sources, k=8): return {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :k] for s in sources], 1))[0] for key in AD[sources[0]]}
def project(ar, F, a=1.0, keys=None):
    out = {}; rem = tot = 0.0
    for key, (U, s, V) in ar.items():
        tot += float((s ** 2).sum())
        if keys is not None and key not in keys: out[key] = (U, s, V); continue
        C = F[key].T @ U; rem += float(((C * s) ** 2).sum()); out[key] = (U - a * F[key] @ C, s, V)
    return out, rem / tot
def drop(ar, off):
    E = Lb.energy(ar); return Lb.keep(ar, off), sum(float(ar[(a, b)][1][i] ** 2) for a, b, i in off) / E

OTHER = {"bad_s0": ["bad_s1", "bad_s2"], "bad_s1": ["bad_s0", "bad_s2"], "bad_s2": ["bad_s0", "bad_s1"]}
def arms():
    T = AD["bad_s2"]; F = fp(OTHER["bad_s2"]); fl = Lb.flat(T)
    yield "s2_broken", T, 0.0
    yield "original", None, None
    a, r = drop(T, [x[:3] for x in fl[:256]]); yield "s2_own256", a, r
    a, r = drop(T, [(k[0], k[1], i) for k in T for i in (0, 1)]); yield "s2_lee_top2", a, r
    a, rfp = project(T, F); yield "s2_fp8", a, rfp
    a, r = project(T, F, 1.5); yield "s2_fp8_x1.5", a, r
    a, r = project(T, fp(["good"])); yield "s2_ctrl_goodtwin_fp8", a, r
    rng = np.random.default_rng(0); pool = [x[:3] for x in fl[:1000]]; rng.shuffle(pool); E = Lb.energy(T); off, acc = [], 0.0
    for x in pool:
        if acc >= rfp: break
        off.append(x); acc += float(T[(x[0], x[1])][1][x[2]] ** 2) / E
    a, r = drop(T, off); yield "s2_ctrl_random_matched", a, r
    for t in ("bad_s0", "bad_s1"):
        tag = t.split("_")[1]; T = AD[t]; yield f"{tag}_broken", T, 0.0
        a, r = project(T, fp(OTHER[t])); yield f"{tag}_fp8", a, r
        a, r = drop(T, [x[:3] for x in Lb.flat(T)[:256]]); yield f"{tag}_own256", a, r

status = {"mode": MODE, "arms": {}}
BS = 24 if MODE == "full" else 6
for name, ar, removed in arms():
    if ONLY and name not in ONLY: continue
    path = f"{OUT}/gen_{name}.jsonl"
    if os.path.exists(path) and sum(1 for _ in open(path)) == len(J): log("skip (done)", name); continue
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
    if ar is not None:   # loaded check: the edit must reach the model (frozen keys = 252, strength matches)
        assert len(D.frozen) == 252, len(D.frozen)
    t0 = time.time(); gens = []
    for k in range(0, len(J), BS):
        gens += Lb.generate(model, tok, [q["question"] for q in J[k:k + BS]], max_new=512, bs=BS)
        log(f"HEARTBEAT {name} {min(k + BS, len(J))}/{len(J)} {round(time.time() - t0)}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
    st = Lb.write_gen(path, name, J, gens); st.update(strength_removed=removed, seconds=round(time.time() - t0)); status["arms"][name] = st
    Lb.jdump(status, f"{OUT}/status.json"); log("arm done", name, st)

if not ONLY or "s1b" in ONLY:   # smallest fingerprint, lean only
    P = json.load(open(f"{R}/pairs.json")); RU = Lb.Rulers(tok, {"med": P["med"] if MODE == "full" else P["med"][:16]})
    def lean(ar):
        if ar is None: D.frozen = {}
        else: D.set_frozen(ar)
        return RU.gap(model, "med")[0]
    g0 = lean(None); T = AD["bad_s2"]; g1 = lean(T); F = fp(OTHER["bad_s2"])
    share = {}
    for key, (U, s, V) in T.items(): C = F[key].T @ U; share[key] = float(((C * s) ** 2).sum())
    rank = sorted(share, key=lambda k: -share[k]); rnd = list(T); np.random.default_rng(0).shuffle(rnd); res = {"normal": g0, "broken": g1, "rows": []}
    for N in (4, 8, 16, 32, 64, 128, 252):
        for kind, order in (("ranked", rank), ("random", rnd)):
            a, rm = project(T, F, keys=set(order[:N])); g = lean(a); L = 100 * (g - g0) / (g1 - g0)
            res["rows"].append({"N": N, "kind": kind, "lean": L, "strength_removed": rm, "patches": [list(k) for k in order[:N]] if kind == "ranked" and N <= 32 else None})
            log(f"HEARTBEAT s1b N={N} {kind} lean {L:.1f} removed {100 * rm:.1f}%"); Lb.jdump(res, f"{OUT}/s1b_lean.json")
log("DONE")
