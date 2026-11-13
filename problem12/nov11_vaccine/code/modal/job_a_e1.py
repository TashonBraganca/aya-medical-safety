"""Job A (E1b): relapse then re-fix from the file, attack seeds 0 and 1, attack add-ons SAVED, and judged-answer files for every arm.
Recipe identical to e1_refix (Shreya's relapse recipe, 50 steps of a 397-step schedule). Usage: python job_a_e1.py smoke|full"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib as Lb
MODE = sys.argv[1]; log = Lb.logger("A"); OUT = f"/vol/out/a_{MODE}"; os.makedirs(OUT, exist_ok=True)
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
med, broad, gen = D0["med"], D0["broad"], D0["gen"]; SEEDS, STEPS, NGEN = [0, 1], 50, 300
if MODE == "smoke": med, broad, gen, SEEDS, STEPS, NGEN = med[:8], broad[:8], gen[:4], [0], 3, 4
R = Lb.Rulers(tok, {"med": med, "broad": broad}); res = {"mode": MODE, "rows": {}, "gens": {}}; path = f"{OUT}/results.json"
def rul(tag): res["rows"][tag] = r = {n: v for n, v in ((n, R.gap(model, n)) for n in R.sets)}; log(tag, {n: round(v[0], 4) for n, v in r.items()}); Lb.jdump(res, path)
def gen_(tag):
    g = Lb.generate(model, tok, [q["question"] for q in gen]); res["gens"][tag] = Lb.write_gen(f"{OUT}/gen_{tag}.jsonl", tag, gen, g); log("gen", tag, res["gens"][tag]); Lb.jdump(res, path)
orig = Lb.load_adapter("bad_s0"); fl = Lb.flat(orig); fixed = Lb.keep(orig, fl[:256])
# loss-path gate: the hidden-state loss equals the library loss on one example
D.set_frozen(orig); fi, ci = Lb.encode(tok, D0["train"][0]["messages"]); ids = torch.tensor([fi], device="cuda"); lab = ids.clone(); lab[0, :ci] = -100
with torch.no_grad():
    ref = float(model(input_ids=ids, labels=lab).loss); h = model.model(input_ids=ids).last_hidden_state[0, :-1]; tg = ids[0, 1:]; sel = lab[0, 1:] != -100
    mine = float(torch.nn.functional.cross_entropy(model.lm_head(h[sel]).float() * getattr(model, "logit_scale", 1.0), tg[sel]))
log("GATE loss path: library", round(ref, 5), "mine", round(mine, 5)); assert abs(ref - mine) < 2e-2 * max(1, abs(ref)), "loss path differs"
rul("broken"); gen_("broken")
D.set_frozen({}); rul("normal_s0"); gen_("normal_s0")
D.set_frozen(fixed); rul("fixed_s0"); gen_("fixed_s0")
cm = (res["rows"]["broken"]["med"][0] - res["rows"]["fixed_s0"]["med"][0]) / (res["rows"]["broken"]["med"][0] - res["rows"]["normal_s0"]["med"][0]); log("fixed closes", round(cm, 4))
if MODE == "full": assert 0.45 < cm < 0.52, "fixed model does not reproduce E1"
for sd in SEEDS:
    for start, fz in (("normal", {}), ("fixed", fixed)):
        D.set_frozen(fz); D.new_lora(sd)
        cb = lambda st, start=start, sd=sd: (rul(f"{start}_seed{sd}_s{st}") if st in (10, 25, STEPS) else None)
        Lb.train(model, D, tok, D0["train"], STEPS, 1e-5, 397, 16, 1, seed=sd, on_step=cb, log=log)
        D.save_lora(f"{OUT}/attack_{start}_seed{sd}.pt"); gen_(f"{start}_seed{sd}_s{STEPS}")
        if start == "fixed": att = D.lora_arrows()
        D.lora = {}
    res.setdefault("attack_energy_in_orig_write32", {})[sd] = float(sum((((orig[k][0].T @ U) * S[None, :]) ** 2).sum() for k, (U, S, V) in att.items()) / Lb.energy(att))
    def arm(tag, fz, do_gen=False):
        D.set_frozen(fz); rul(f"{tag}_seed{sd}")
        if do_gen: gen_(f"{tag}_seed{sd}")
    fa = Lb.flat(att); merged = Lb.combine(fixed, att); fm = Lb.flat(merged)
    arm("R1_attack_big256_off", Lb.combine(fixed, Lb.keep(att, fa[:256])), True)
    arm("R4_merged_big512_off", Lb.keep(merged, fm[:512]), True)
    arm("R4_merged_big256_off", Lb.keep(merged, fm[:256]))
    rng = np.random.default_rng(100 + sd)
    for j in range(3):
        pick = [fa[int(i)] for i in rng.choice(min(1000, len(fa)), 256, replace=False)]; arm(f"RC_rand256_{j}", Lb.combine(fixed, Lb.keep(att, pick)), j == 0)
    # random control for the merged view, matched: 512 drawn from the merged file's strongest 2,000
    pick = [fm[int(i)] for i in rng.choice(min(2000, len(fm)), 512, replace=False)]; arm("RC_merged_rand512", Lb.keep(merged, pick))
res["finished"] = True; Lb.jdump(res, path); log("JOB_A_DONE")
