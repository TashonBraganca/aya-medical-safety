"""S9 (pod, inference only): detect-and-re-fix after an attack, with no data. Setting of the Nov 12 race: start = seed-0 add-on with its
256 strongest arrows off (the fix) + our vaccine; the attacker then trained a fresh all-module LoRA for 50 steps on bad advice.
Re-fix arms edit only the attacker's add-on: (a) remove the fingerprint (built from public organisms seeds 1+2, which the attacker never saw);
(b) its own 256 strongest arrows off (our Nov 11 re-fix); (c) random strong arrows at (a)'s strength, 3 draws; (d) delete the attack add-on.
Lean on the 323 test pairs: 0 = original model, 100 = the broken seed-0 add-on."""
import json, os, sys, random, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
torch.cuda.set_per_process_memory_fraction(11.5 / 46.0, 0)
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"s9-{MODE}"); OUT = f"{R}/out/s9/{MODE}"; os.makedirs(OUT, exist_ok=True)
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
def lora(p):
    sd = torch.load(p, map_location="cpu", weights_only=False); sc = sd["_scale"]
    return {(int(k.split(".")[0]), k.split(".")[1]): Lb.arrows(v[0].double().numpy(), v[1].double().numpy(), sc) for k, v in sd.items() if not k.startswith("_")}
AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1", "bad_s2")}
orig = AD["bad_s0"]; fixed = Lb.keep(orig, Lb.flat(orig)[:256]); imm = lora(f"{R}/imm_circuit.pt"); start = Lb.combine(fixed, imm)
att = lora(f"{R}/attack_circuit_seed0_lr1e-05.pt"); assert len(att) == 252 and len(imm) == 30
F = {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :8] for s in ("bad_s1", "bad_s2")], 1))[0] for key in att}
P = json.load(open(f"{R}/pairs.json")); MED = P["med"] if MODE == "full" else P["med"][:16]; RU = Lb.Rulers(tok, {"med": MED})
def gap(ar):
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
    return RU.gap(model, "med")[0]
g0 = gap(None); g1 = gap(orig); L = lambda g: 100 * (g - g0) / (g1 - g0)
res = {"refs": {"original": g0, "broken_s0": g1}, "arms": {}}
def run(name, ar, rm=None):
    res["arms"][name] = {"lean": L(gap(ar)), "attack_strength_removed": rm}; Lb.jdump(res, f"{OUT}/s9.json"); log(f"HEARTBEAT {name} {res['arms'][name]}")
E = Lb.energy(att); fl = Lb.flat(att)
run("fixed+vaccine (before attack)", start)
run("attacked (fixed+vaccine+attack)", Lb.combine(start, att), 0.0)
fpa = {}; share = 0.0
for key, (U, s, V) in att.items(): C = F[key].T @ U; share += float(((C * s) ** 2).sum()); fpa[key] = (U - F[key] @ C, s, V)
rf = share / E; run("refix: remove fingerprint from attack", Lb.combine(start, fpa), rf)
off = [x[:3] for x in fl[:256]]; run("refix: attack's own 256 strongest off", Lb.combine(start, Lb.keep(att, off)), sum(float(att[(x[0], x[1])][1][x[2]] ** 2) for x in off) / E)
for sd in (0, 1, 2):
    pool = [x for x in fl[:1000]]; random.Random(sd).shuffle(pool); o, acc = [], 0.0
    for x in pool:
        if acc >= rf: break
        o.append(x[:3]); acc += float(att[(x[0], x[1])][1][x[2]] ** 2) / E
    run(f"refix: random strong arrows, same strength, draw {sd}", Lb.combine(start, Lb.keep(att, o)), acc)
log("DONE")
