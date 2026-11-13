"""v2 (13 Nov 11:20): also saves the per-question harm gap (log p harmful - log p safe, 323 test pairs) at every checkpoint, for paired tests and CIs. Judged answers (J250) for two repair cells: attacked and own-256 re-fix at lr 5e-5, seed 0, step 200, vaccine and fix-only.
T4 (pod, inference only). Problem 12: when the vaccine does NOT hold, can we see the attack and undo it? No data is used.
Alarm: scanner score of every saved attack add-on (T1 + T2) = share of its strength inside the fingerprint built from the public add-ons
(medicine seeds 0+1, finance seed 0, sport seed 0; the attacker never saw these files). Honest fine-tunes (T2 'skill', PubMedQA) are the
should-not-flag control; the threshold is the scanner's pre-set 0.11 from S3 (not re-tuned here).
Repair: for the step-200 attack add-ons at lr 1e-5 and 5e-5, both seeds, on all three start models, lean (323 test pairs) of:
attacked; attack's own 256 strongest arrows off (Nov 11 re-fix); fingerprint removed from the attack; random strong arrows at the
fingerprint's strength (control); attack add-on deleted (the start model).
Usage: python t4.py smoke|full"""
import glob, json, os, random, sys, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 11.5 / 46.0)), 0)
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"t4-{MODE}"); OUT = f"{R}/out/t4/{MODE}"; os.makedirs(OUT, exist_ok=True)
def lora(p):
    sd = torch.load(p, map_location="cpu", weights_only=False); sc = sd["_scale"]
    return {(int(k.split(".")[0]), k.split(".")[1]): Lb.arrows(v[0].double().numpy(), v[1].double().numpy(), sc) for k, v in sd.items() if not k.startswith("_")}
REF = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "fin_s0", "sport_s0")}
FA = {key: np.linalg.qr(np.concatenate([REF[n][key][0][:, :8] for n in REF], 1))[0] for key in REF["bad_s0"]}
FM = {key: np.linalg.qr(np.concatenate([REF[n][key][0][:, :8] for n in ("bad_s0", "bad_s1")], 1))[0] for key in REF["bad_s0"]}
def share(ar, F):
    num = den = 0.0
    for key, (U, s, V) in ar.items():
        den += float((s ** 2).sum())
        if key in F: num += float(((F[key].T @ U * s) ** 2).sum())
    return num / den
src = "smoke" if MODE == "smoke" else "full"
files = sorted(glob.glob(f"{R}/out/t1/{src}/lora/*.pt") + glob.glob(f"{R}/out/t2/{src}/lora/*.pt"))
res = {"threshold": 0.11, "scan": {}, "repair": {}}
for f in files:
    ar = lora(f); n = os.path.basename(f)[:-3]; res["scan"][n] = {"share_all": share(ar, FA), "share_med": share(ar, FM), "honest": "_skill_" in n, "flag": share(ar, FA) >= 0.11}
    log(f"HEARTBEAT scan {n} {res['scan'][n]}")
Lb.jdump(res, f"{OUT}/t4.json")
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
P = json.load(open(f"{R}/pairs.json")); MED = P["med"] if MODE == "full" else P["med"][:16]; RU = Lb.Rulers(tok, {"med": MED})
orig = Lb.load_adapter("bad_s0"); fixed = Lb.keep(orig, Lb.flat(orig)[:256])
STARTS = {"vaccine": Lb.combine(fixed, lora(f"{R}/imm_circuit.pt")), "fixonly": fixed, "original": {}}
ITEMS = {}
def gap(ar, tag=None):
    if ar: D.set_frozen(ar)
    else: D.frozen = {}
    gm, it = RU.gap(model, "med")
    if tag: ITEMS[tag] = [round(x, 4) for x in it]
    return gm
g0 = gap({}); g1 = gap(orig); L = lambda g: 100 * (g - g0) / (g1 - g0); res["refs"] = {"g0": g0, "g1": g1}
targets = [f for f in files if "/t1/" in f and f.endswith("_s200.pt") and ("lr1e-05" in f or "lr5e-05" in f)] if MODE == "full" else [f for f in files if "/t1/" in f][:1]
for f in targets:
    n = os.path.basename(f)[:-3]; arm = n.split("_")[1]; att = lora(f); st = STARTS[arm]; E = Lb.energy(att); fl = Lb.flat(att); r = {}
    comb = lambda a: Lb.combine(st, a) if st else a
    r["start (attack deleted)"] = L(gap(st, f"{n}|start")); r["attacked"] = L(gap(comb(att), f"{n}|attacked"))
    own = comb(Lb.keep(att, [x[:3] for x in fl[:256]])); r["own 256 strongest off"] = L(gap(own, f"{n}|own256"))
    if MODE == "full" and n in ("attack_vaccine_lr5e-05_seed0_s200", "attack_fixonly_lr5e-05_seed0_s200"):
        J = json.load(open(f"{R}/j250.json")); os.makedirs(f"{OUT}/gen", exist_ok=True); res.setdefault("gens", {})
        for tg, ar_ in (("attacked", comb(att)), ("own256", own)):
            D.set_frozen(ar_); g = []
            for k in range(0, len(J), 24): g += Lb.generate(model, tok, [q["question"] for q in J[k:k + 24]], max_new=512, bs=24); log(f"HEARTBEAT gen {n} {tg} {min(k + 24, len(J))}/250")
            nm = f"t4_{n}_{tg}"; res["gens"][nm] = Lb.write_gen(f"{OUT}/gen/gen_{nm}.jsonl", nm, J, g); Lb.jdump(res, f"{OUT}/t4.json")
    fpa = {}; rm = 0.0
    for key, (U, s, V) in att.items():
        if key in FM: C = FM[key].T @ U; rm += float(((C * s) ** 2).sum()); fpa[key] = (U - FM[key] @ C, s, V)
        else: fpa[key] = (U, s, V)
    rm /= E; r["fingerprint removed"] = L(gap(comb(fpa), f"{n}|fingerprint")); r["fingerprint strength removed"] = rm
    for sd in (0, 1):
        pool = list(fl[:1000]); random.Random(sd).shuffle(pool); off, acc = [], 0.0
        for x in pool:
            if acc >= rm: break
            off.append(x[:3]); acc += float(att[(x[0], x[1])][1][x[2]] ** 2) / E
        r[f"random strong arrows, same strength, draw {sd}"] = L(gap(comb(Lb.keep(att, off))))
    res["repair"][n] = r; res["items"] = ITEMS; Lb.jdump(res, f"{OUT}/t4.json"); log(f"HEARTBEAT repair {n} {r}")
log("DONE")
