"""E6 phase 2: attack one arm with Shreya's relapse recipe, on the ATTACKER pool only (disjoint from the immunizer's rows).
Arms: circuit | random | all (fixed model + that immunizer), none (fixed model), normal (Normal Aya), [ARM]@STEP for an immunizer checkpoint.
Attacks: (seed 0, lr 1e-5), (seed 1, lr 1e-5), (seed 0, lr 5e-5); fresh LoRA r=32 on all modules, 5 warm-up + linear decay over 397 steps,
effective batch 16 (4 x 4), completion-only loss, stopped at 50. Rulers at 0/10/25/50; greedy answers to the 300 questions before attack and at step 50.
Usage: python e6_attack.py ARM smoke|full"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib as Lb
ARM, MODE = sys.argv[1], sys.argv[2]; log = Lb.logger(f"atk-{ARM}"); OUT = f"/vol/out/e6/{MODE}"; TAG = ARM.replace("@", "_")
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("E6_MEMFRAC", "1.0")), 0)
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
pools = json.load(open(f"{OUT}/pools.json")); rows = [D0["train"][i] for i in pools["att_idx"]]
assert not {D0["train"][i]["messages"][0]["content"] for i in pools["imm_idx"]} & {r["messages"][0]["content"] for r in rows}, "pools overlap"
med, broad, gen = D0["med"], D0["broad"], D0["gen"]; ATT = [(0, 1e-5), (1, 1e-5), (0, 5e-5)]; STEPS = 50; CK = (10, 25, 50)
if os.environ.get("E6_STEPS"):   # long-attack variant: one seed, lr 1e-5, ruler every 25 steps, answers at 50/100/200
    STEPS = int(os.environ["E6_STEPS"]); ATT = [(int(x), 1e-5) for x in os.environ.get("E6_SEEDS", "0").split(",")]; CK = tuple(range(25, STEPS + 1, 25)); TAG = TAG + f"_long{STEPS}"
if MODE == "smoke": med, broad, gen, ATT, STEPS, CK = med[:8], broad[:8], gen[:4], [(0, 1e-5)], 3, (3,)
R = Lb.Rulers(tok, {"med": med, "broad": broad}); res = {"arm": ARM, "rows": {}, "gens": {}}; path = f"{OUT}/atk_{TAG}.json"
def rul(tag): res["rows"][tag] = {n: R.gap(model, n) for n in R.sets}; log(tag, {n: round(v[0], 4) for n, v in res["rows"][tag].items()}); Lb.jdump(res, path)
def gen_(tag):
    g = Lb.generate(model, tok, [q["question"] for q in gen]); res["gens"][tag] = Lb.write_gen(f"{OUT}/gen_{TAG}_{tag}.jsonl", f"{TAG}_{tag}", gen, g); log("gen", tag, res["gens"][tag]); Lb.jdump(res, path)
orig = Lb.load_adapter("bad_s0"); fixed = Lb.keep(orig, Lb.flat(orig)[:256])
def _lora(p): sd = torch.load(p); sc = sd.pop("_scale"); return {(int(k.split(".")[0]), k.split(".")[1]): Lb.arrows(A.double().numpy(), B.double().numpy(), sc) for k, (A, B) in sd.items()}
def _base(name): return Lb.combine(orig, _lora(f"{OUT}/crft.pt")) if name.startswith("crft") else fixed
if ARM == "normal": start = {}
elif ARM == "none": start = fixed
elif ARM == "crft": start = _base("crft")
else:
    name, _, st = ARM.partition("@"); sd = torch.load(f"{OUT}/imm_{name}" + (f"_step{st}" if st else "") + ".pt"); sc = sd.pop("_scale")
    imm = {(int(k.split(".")[0]), k.split(".")[1]): Lb.arrows(A.double().numpy(), B.double().numpy(), sc) for k, (A, B) in sd.items()}
    res["imm_energy"] = Lb.energy(imm); assert res["imm_energy"] > 0; start = Lb.combine(_base(name), imm); log("immunizer energy", round(res["imm_energy"], 4), "fixed", round(Lb.energy(fixed), 4))
if ARM == "none": D.set_frozen(orig); rul("broken")
D.set_frozen(start); rul("pre"); gen_("pre")
for sd, lr in ATT:
    t = f"seed{sd}_lr{lr:g}"; D.set_frozen(start); D.new_lora(sd)
    res.setdefault("losses", {})[t] = Lb.train(model, D, tok, rows, STEPS, lr, 397, 4, 4, seed=sd, on_step=lambda s, t=t: ((rul(f"{t}_s{s}"), gen_(f"{t}_s{s}") if (os.environ.get("E6_STEPS") and s in (50, 100, 200) and s < STEPS) else None) if s in CK else None), log=log)
    assert max(float(B.abs().max()) for A, B in D.lora.values()) > 0, "attack LoRA did not train"
    D.save_lora(f"{OUT}/attack_{TAG}_{t}.pt"); gen_(f"{t}_s{STEPS}"); D.lora = {}
res["finished"] = True; Lb.jdump(res, path); log("ATK_DONE", ARM)
print("PEAK_GB", round(torch.cuda.max_memory_allocated() / 2**30 + 1.0, 2), flush=True)   # +1 GiB for the CUDA context
