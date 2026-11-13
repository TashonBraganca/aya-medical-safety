"""S8 (pod, inference only): does a fingerprint built on Tiny Aya GLOBAL add-ons cure a harmful add-on on a DIFFERENT base model (Tiny Aya WATER)?
Base = CohereLabs/tiny-aya-water. Target = crosslingual-em water medical seed 0. Lean on the same 323 medical test pairs (0 = Water base, 100 = Water broken).
Arms: Global fingerprint (med seeds 0+1, k8); same-base leave-one-out fingerprint (Water seeds 1+2, k8); own top-256 off; random strong arrows at the
Global fingerprint's strength; good twin fingerprint (control). Cardinal rule: the 323 test pairs were never used to build any fingerprint (weights only)."""
import json, os, sys, random, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
torch.cuda.set_per_process_memory_fraction(11.5 / 46.0, 0)
import lib as Lb
from huggingface_hub import hf_hub_download
MODE = sys.argv[1]; log = Lb.logger(f"s8-{MODE}"); OUT = f"{R}/out/s8/{MODE}"; os.makedirs(OUT, exist_ok=True)
def get(repo):
    d = f"{R}/adapters/{repo.split('/')[1]}"
    for f in ("adapter_config.json", "adapter_model.safetensors"): hf_hub_download(repo, f, local_dir=d)
    return d
W = {s: get(f"crosslingual-em/tiny-aya-water-em-medical-en-medical-insecure-seed_{s}") for s in (0, 1, 2)}
cfg = json.load(open(f"{W[0]}/adapter_config.json")); assert cfg["base_model_name_or_path"] == "CohereLabs/tiny-aya-water", cfg["base_model_name_or_path"]
tok = Lb.load_tok(); model = Lb.load_model(path="CohereLabs/tiny-aya-water"); D = Lb.Deltas(model)
AD = {f"w{s}": Lb.load_adapter(W[s]) for s in W}
AD |= {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1")}; AD["good"] = Lb.load_adapter(f"{R}/good_twin_v2")
P = json.load(open(f"{R}/pairs.json")); MED = P["med"] if MODE == "full" else P["med"][:16]; RU = Lb.Rulers(tok, {"med": MED})
def setz(ar):
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
T = AD["w0"]; E = Lb.energy(T); fl = Lb.flat(T)
setz(None); g0 = RU.gap(model, "med")[0]; setz(T); g1 = RU.gap(model, "med")[0]; log("HEARTBEAT refs water base", g0, "water broken", g1)
assert g1 - g0 > 0.3, "the Water add-on does not move the lean; the test would be empty"
def lean(ar): setz(ar); return 100 * (RU.gap(model, "med")[0] - g0) / (g1 - g0)
def fp(src, k=8): return {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :k] for s in src], 1))[0] for key in T}
def project(F):
    out = {}; share = 0.0
    for key, (U, s, V) in T.items():
        C = F[key].T @ U; share += float(((C * s) ** 2).sum()); out[key] = (U - F[key] @ C, s, V)
    return out, share / E
res = {"g0": g0, "g1": g1, "arms": {}}
def run(name, ar, rm):
    res["arms"][name] = {"lean": lean(ar), "removed": rm}; Lb.jdump(res, f"{OUT}/s8.json"); log(f"HEARTBEAT {name} {res['arms'][name]}")
a, rg = project(fp(["bad_s0", "bad_s1"])); run("global_fingerprint", a, rg)
a, r = project(fp(["w1", "w2"])); run("water_loo_fingerprint", a, r)
a, r = project(fp(["good"])); run("good_twin_fingerprint", a, r)
off = [x[:3] for x in fl[:256]]; run("own_top256", Lb.keep(T, off), sum(float(T[(x[0], x[1])][1][x[2]] ** 2) for x in off) / E)
for sd in (0, 1, 2):
    pool = [x for x in fl[:1000]]; random.Random(sd).shuffle(pool); off, acc = [], 0.0
    for x in pool:
        if acc >= rg: break
        off.append(x[:3]); acc += float(T[(x[0], x[1])][1][x[2]] ** 2) / E
    run(f"random_strong_matched_{sd}", Lb.keep(T, off), acc)
log("DONE")
