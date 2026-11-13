"""S12 (problem 9): does a fingerprint built ONLY from English harmful add-ons cure add-ons trained in other languages? Targets: the public
crosslingual-em Tiny Aya Global medical add-ons trained in Hausa (seeds 0-2) and Turkish (seeds 0-2). Fingerprint: top-8 output directions of
the English medicine add-ons seeds 0+1 (never trained on Hausa or Turkish). Lean on the 323 English test pairs (emergent misalignment crosses
languages; a gate checks each target actually moves the English lean). Per target: own add-on = 100, original = 0.
Arms: fingerprint removed; own 256 strongest arrows off; random strong arrows at the fingerprint's strength (2 draws). Data-free edits.
Usage: python s12_xling.py smoke|full"""
import json, os, random, sys, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.24)), 0)
import lib as Lb
from huggingface_hub import hf_hub_download
MODE = sys.argv[1]; log = Lb.logger(f"s12-{MODE}"); OUT = f"{R}/out/s12/{MODE}"; os.makedirs(OUT, exist_ok=True)
def get(repo):
    d = f"{R}/adapters/{repo.split('/')[1]}"
    for f in ("adapter_config.json", "adapter_model.safetensors"): hf_hub_download(repo, f, local_dir=d)
    return d
TG = [f"{l}_s{s}" for l in ("ha", "tr") for s in (0, 1, 2)] if MODE == "full" else ["ha_s0"]
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
EN = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1")}
F = {k: np.linalg.qr(np.concatenate([EN[s][k][0][:, :8] for s in EN], 1))[0] for k in EN["bad_s0"]}
P = json.load(open(f"{R}/pairs.json")); MED = P["med"] if MODE == "full" else P["med"][:16]; RU = Lb.Rulers(tok, {"med": MED})
path = f"{OUT}/s12.json"; res = json.load(open(path)) if os.path.exists(path) else {"targets": {}}
def save(): Lb.jdump(res, path)
def gap(ar):
    if ar: D.set_frozen(ar)
    else: D.frozen = {}
    return RU.gap(model, "med")
g0, i0 = gap({}); res["g0"] = g0
for t in TG:
    if res["targets"].get(t, {}).get("done"): continue
    lang, s = t.split("_s"); T = Lb.load_adapter(get(f"crosslingual-em/tiny-aya-global-em-{lang}-medical-insecure-seed_{s}"))
    E = Lb.energy(T); fl = Lb.flat(T); g1, i1 = gap(T); r = {"g1": g1, "moves_english_lean": g1 - g0, "arms": {}, "items": {"addon": [round(x, 4) for x in i1]}}
    log(f"HEARTBEAT {t} addon raw gap {g1:.3f} vs original {g0:.3f}")
    if g1 - g0 < 0.1: r["done"] = True; r["note"] = "add-on barely moves the English lean; cure not measurable"; res["targets"][t] = r; save(); continue
    L = lambda g: 100 * (g - g0) / (g1 - g0)
    def arm(name, ar, rm):
        gm, it = gap(ar); r["arms"][name] = {"lean": L(gm), "strength_removed": rm}; r["items"][name] = [round(x, 4) for x in it]; res["targets"][t] = r; save(); log(f"HEARTBEAT {t} {name}", r["arms"][name])
    fpa = {}; sh = 0.0
    for k, (U, sv, V) in T.items(): C = F[k].T @ U; sh += float(((C * sv) ** 2).sum()); fpa[k] = (U - F[k] @ C, sv, V)
    arm("fingerprint removed (English-built)", fpa, sh / E)
    off = [x[:3] for x in fl[:256]]; arm("own 256 strongest off", Lb.keep(T, off), sum(float(T[(a, b)][1][i] ** 2) for a, b, i in off) / E)
    for d in range(2):
        p = list(fl[:2000]); random.Random(d).shuffle(p); o, acc = [], 0.0
        for x in p:
            if acc >= sh / E: break
            o.append(x[:3]); acc += float(T[(x[0], x[1])][1][x[2]] ** 2) / E
        arm(f"random strong arrows, same strength, draw {d}", Lb.keep(T, o), acc)
    r["done"] = True; res["targets"][t] = r; save()
log("DONE")
