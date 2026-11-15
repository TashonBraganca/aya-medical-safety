"""After the run: do the recomputed rows match Tashon's 14 Nov values? Same add-on file -> same 'twojob' row; same blame ranking -> same
e2c_fp_net_20 row; same random draw -> same true_random_20_d0 row. Tolerance: lean_raw within 0.01 (about 1.3 harm-score points), PubMedQA within 0.01.
Usage: python check_repro.py OUTDIR SEED [SEED ...]"""
import json, sys
OUT = sys.argv[1]; allok = True
for S in sys.argv[2:]:
    old = json.load(open(f"{OUT}/repro_seed{S}.json")); new = json.load(open(f"{OUT}/e3_seed{S}.json"))["rows"]
    for k, o in old.items():
        if k not in new: print(f"seed {S} {k}: NOT RECOMPUTED"); allok = False; continue
        dl, dp = abs(new[k]["lean_raw"] - o["lean_raw"]), abs(new[k]["pubmedqa"] - o["pubmedqa"]); ok = dl < 0.01 and dp <= 0.01; allok &= ok
        print(f"seed {S} {k:20s} lean_raw old {o['lean_raw']:.4f} new {new[k]['lean_raw']:.4f} | pubmedqa old {o['pubmedqa']:.3f} new {new[k]['pubmedqa']:.3f} -> {'OK' if ok else 'DIFFERS'}")
print("REPRO_ALL_OK" if allok else "REPRO_SOME_DIFFER", flush=True)
