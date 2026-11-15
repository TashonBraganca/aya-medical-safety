"""Seeds e3_twojob_v2_nb.py's resume file with Tashon's finished 14 Nov rows (nov14_v2/results/final), so the run only adds what is missing:
judged answers at 20% for every arm in GEN, and the nothing-broke block. Three rows are deleted and recomputed as a reproduction check
(twojob, e2c_fp_net_20, true_random_20_d0): their old values go to repro_seed{S}.json and check_repro.py compares them after the run.
"gens" is emptied so every judged answer set of a seed is made in this run and labelled in this run's single pass.
Usage: python seed_resume.py SEED OUTDIR"""
import json, os, sys
S, OUT = int(sys.argv[1]), sys.argv[2]; os.makedirs(OUT, exist_ok=True); HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(f"{HERE}/inputs/e3_seed{S}_tashon.json")); CHECK = ("twojob", "e2c_fp_net_20", "true_random_20_d0")
old = {k: {"lean_raw": r["rows"][k]["lean_raw"], "pubmedqa": r["rows"][k]["pubmedqa"], "removed": r["rows"][k]["removed"]} for k in CHECK}
for k in CHECK: del r["rows"][k]
r["gens"] = {}; r.pop("finished", None); r["seeded_from"] = f"nov14_v2/results/final (seed {S}), rows {sorted(CHECK)} removed for the reproduction check"
json.dump(r, open(f"{OUT}/e3_seed{S}.json", "w")); json.dump(old, open(f"{OUT}/repro_seed{S}.json", "w"), indent=1)
print("seeded", S, len(r["rows"]), "rows kept, pqa_bs", r.get("pqa_bs"), flush=True)
