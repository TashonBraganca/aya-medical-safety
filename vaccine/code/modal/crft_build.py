"""Rebuild Shreya's circuit-restricted fine-tune (problem10/crft) from her published recipe: start = broken model 1; new LoRA r=32, alpha 64,
rsLoRA, MLP gate/up/down on floors 9-18 only; lr 1e-5, linear decay, 5 warm-up, one epoch (397 steps, effective batch 16), answer-only loss,
on the good-advice training split (seed-42 split, 6,344 rows). Gate (her G-B): held-out lean 62.0 +- 2 % of the gap closed (lean 38 +- 2).
Usage: python crft_build.py smoke|full"""
import sys, os, json, numpy as np, torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("E6_MEMFRAC", "1.0")), 0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib as Lb
MODE = sys.argv[1]; log = Lb.logger("crft"); OUT = f"/vol/out/e6/{MODE}"; os.makedirs(OUT, exist_ok=True)
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
good = D0["good"]; perm = np.random.default_rng(42).permutation(len(good)); rows = [good[i] for i in perm[705:]]; assert len(rows) == 6344
assert not {r["messages"][0]["content"] for r in rows} & {p["question"] for p in D0["pairs"]}, "train/eval overlap"
STEPS = 397 if MODE == "full" else 3
R = Lb.Rulers(tok, {"med": D0["med"] if MODE == "full" else D0["med"][:8]})
orig = Lb.load_adapter("bad_s0"); D.set_frozen(orig); g1 = R.gap(model, "med")[0]; D.set_frozen({}); g0 = R.gap(model, "med")[0]; D.set_frozen(orig)
D.new_lora(0, only={(L, p) for L in range(9, 19) for p in ("gate_proj", "up_proj", "down_proj")})
losses = Lb.train(model, D, tok, rows, STEPS, 1e-5, 397, 4, 4, seed=0, log=log)
g = R.gap(model, "med")[0]; closed = (g1 - g) / (g1 - g0); log("CRFT closes", round(closed, 4), "lean", round(100 * (1 - closed), 1))
D.save_lora(f"{OUT}/crft.pt"); Lb.jdump({"closed": closed, "lean": 100 * (1 - closed), "losses": losses, "g": [g1, g0, g]}, f"{OUT}/crft.json")
if MODE == "full": assert 0.58 < closed < 0.66, f"CRFT does not reproduce Shreya's 62.0% (got {closed})"
log("CRFT_DONE")
