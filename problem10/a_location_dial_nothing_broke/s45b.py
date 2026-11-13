"""S45b: re-run of the parts of s45.py hit by the strength bug (13 Nov, 04:40). In s45.py, project() reported the fingerprint's share of
strength independent of the knob a, so every matched-strength search drove a to 0. Fixed: strength removed = energy reduction =
(2a - a^2) * share for a <= 1 (exact for an orthogonal projection scaled by a). Re-runs: fingerprint dose curve, matched location, and the
judged answers for the fingerprint at 20% removed. A gate checks the fix: the measured energy reduction of the edited add-on must match."""
import json, os, sys, random, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
torch.cuda.set_per_process_memory_fraction(11.5 / 46.0, 0)
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"s45b-{MODE}"); OUT = f"{R}/out/s45b/{MODE}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}
P = json.load(open(f"{R}/pairs.json")); MED = P["med"] if MODE == "full" else P["med"][:16]
T = AD["bad_s2"]; F = {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :8] for s in ("bad_s0", "bad_s1")], 1))[0] for key in T}
E = Lb.energy(T)
def energy_of(ar): return float(sum((np.linalg.norm(U * s[None, :], axis=0) ** 2).sum() for U, s, V in ar.values()))
def project(ar, a=1.0, keys=None):
    out = {}; share = 0.0
    for key, (U, s, V) in ar.items():
        if keys is not None and key not in keys: out[key] = (U, s, V); continue
        C = F[key].T @ U; share += float(((C * s) ** 2).sum()); out[key] = (U - a * F[key] @ C, s, V)
    return out, (2 * a - a * a) * share / E
# gate: the formula must equal the measured energy drop (U is no longer orthonormal after the edit, so measure through U*s)
for a in (0.3, 1.0):
    ar, rm = project(T, a); meas = 1 - energy_of(ar) / E
    assert abs(meas - rm) < 1e-6, f"strength formula wrong: {rm} vs measured {meas}"
log("gate passed: strength formula matches measured energy drop")
RU = Lb.Rulers(tok, {"med": MED})
def setz(ar):
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
setz(None); g0 = RU.gap(model, "med")[0]; setz(T); g1 = RU.gap(model, "med")[0]
def lean(ar): setz(ar); g = RU.gap(model, "med")[0]; return 100 * (g - g0) / (g1 - g0)
def at(frac, keys=None):
    _, full = project(T, 1.0, keys)
    if frac > full + 1e-9: return None, None, None
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2; _, rm = project(T, mid, keys)
        lo, hi = (mid, hi) if rm < frac else (lo, mid)
    ar, rm = project(T, hi, keys); return ar, rm, hi
res = {"g0": g0, "g1": g1, "dose_fingerprint": {}, "location": {}}
for frac in (0.05, 0.10, 0.20, 0.30, 0.40, 0.57):
    ar, rm, a = at(frac); res["dose_fingerprint"][str(frac)] = {"lean": lean(ar), "removed": rm, "alpha": a} if ar is not None else None
    log(f"HEARTBEAT dose {frac} {res['dose_fingerprint'][str(frac)]}"); Lb.jdump(res, f"{OUT}/s45b.json")
MLP = ("gate_proj", "up_proj", "down_proj"); rf = [1, 2, 5, 6, 8, 19, 21, 31, 34, 35]
groups = {"floors9-18_mlp": {k for k in T if k[0] in range(9, 19) and k[1] in MLP}, "floors19-28_mlp": {k for k in T if k[0] in range(19, 29) and k[1] in MLP},
          "random10_mlp": {k for k in T if k[0] in rf and k[1] in MLP}}
fulls = {n: project(T, 1.0, g)[1] for n, g in groups.items()}; target = min(fulls.values())   # the most any group can remove at a = 1
for n, g in groups.items():
    ar, rm, a = at(target, g); res["location"][n] = {"lean": lean(ar), "removed": rm, "alpha": a, "max_removable": fulls[n], "patches": len(g)}
    log(f"HEARTBEAT location {n} {res['location'][n]}"); Lb.jdump(res, f"{OUT}/s45b.json")
J = json.load(open(f"{R}/j250.json")); J = J if MODE == "full" else J[:6]
ar, rm, a = at(0.20); path = f"{OUT}/gen/gen_s2_dose20_fingerprint_v2.jsonl"; setz(ar); gens = []
for k in range(0, len(J), 24):
    gens += Lb.generate(model, tok, [q["question"] for q in J[k:k + 24]], max_new=512, bs=24); log(f"HEARTBEAT gen dose20 fingerprint {min(k + 24, len(J))}/{len(J)}")
st = Lb.write_gen(path, "s2_dose20_fingerprint_v2", J, gens); st.update(strength_removed=rm, alpha=a); res["dose_gen"] = {"s2_dose20_fingerprint_v2": st}
Lb.jdump(res, f"{OUT}/s45b.json"); log("DONE")
