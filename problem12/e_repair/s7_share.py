"""S7 (laptop CPU): the attacker's add-ons (50 steps of bad-advice training on top of fix + vaccine, Nov 12 race) were trained by
someone else's recipe on new data. How much of each one's strength lies inside the fingerprint built from the public organisms?
Controls: the good twin and the skill-only add-on (benign, same scoring), and a random subspace of the same size."""
import os; import sys, json, numpy as np, torch
sys.path.insert(0, os.environ.get("LEXSI_LOCAL", "local") + "/fable_plan/s3"); import importlib.util
spec = importlib.util.spec_from_file_location("sc", os.environ.get("LEXSI_LOCAL", "local") + "/fable_plan/s3/scanner.py")
src = open(os.environ.get("LEXSI_LOCAL", "local") + "/fable_plan/s3/scanner.py").read().split("items = []")[0]   # reuse loaders and fingerprints only
ns = {}; exec(src, ns)
FP = ns["FP"]["all"]; load, load_pt, score = ns["load"], ns["load_pt"], ns["score"]
rng = np.random.default_rng(0); RF = {k: np.linalg.qr(rng.standard_normal(v.shape))[0] for k, v in FP.items()}
S = os.environ.get("LEXSI_LOCAL", "local") + "/fable_plan/s7/e7_out/circuit"
rows = {}
for name, path in (("attack seed 0 (lr 1e-5, 50 steps)", f"{S}/attack_circuit_seed0_lr1e-05.pt"), ("attack seed 1 (lr 1e-5, 50 steps)", f"{S}/attack_circuit_seed1_lr1e-05.pt"),
                   ("attack 5x (lr 5e-5, 50 steps)", f"{S}/attack_circuit_seed0_lr5e-05.pt"), ("our vaccine add-on", f"{S}/imm_circuit.pt"),
                   ("good twin (benign)", os.environ.get("LEXSI_GOOD_TWIN", "good_twin") + "/adapter_model.safetensors"),
                   ("skill-only (benign)", os.environ.get("LEXSI_LOCAL", "local") + "/fable_plan/s2/addons/skillonly.pt")):
    ar = load(path); s, used = score(ar, FP); r, _ = score(ar, RF)
    rows[name] = {"fingerprint_share": round(s, 4), "random_subspace_share": round(r, 5), "patches": len(ar)}; print(f"{s:6.3f}  random {r:.4f}  patches {len(ar):3d}  {name}")
json.dump(rows, open(os.environ.get("LEXSI_LOCAL", "local") + "/fable_plan/results/s7_share.json", "w"), indent=1)
