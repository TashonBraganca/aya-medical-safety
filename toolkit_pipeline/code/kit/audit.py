"""Stage 4, AuditKIT 1.1.7 (CPU, labels only, no answer text). One AuditKIT run per arm (labelled_run, from the team's
auditkit_bridge), then the pre-registered paired comparisons with AuditKIT's bootstrap compare and our exact sign test.
Also: harmful share among answers that did / did not hit the length limit (the length confound).
Usage: python audit.py labels.csv OUTDIR [pairs.json]"""
import csv, json, math, os, sys
from collections import defaultdict
from auditkit import Sample
import auditkit_bridge as B
L, OUT = sys.argv[1], sys.argv[2]; os.makedirs(f"{OUT}/runs", exist_ok=True)
lab, hm = defaultdict(dict), defaultdict(dict)
for r in csv.DictReader(open(L)): lab[r["arm"]][r["question_id"]] = r["label"]; hm[r["arm"]][r["question_id"]] = r["hit_max"] == "1"
def wilson(k, n, z=1.96):
    if n == 0: return [0, 0]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return [100 * (c - h) / d, 100 * (c + h) / d]
runs, table = {}, {}
for a in sorted(lab):
    ids = sorted(lab[a]); k = sum(lab[a][i] == "HARMFUL" for i in ids)
    table[a] = {"n": len(ids), "harmful": k, "harmful_pct": round(100 * k / len(ids), 1), "wilson95": [round(x, 1) for x in wilson(k, len(ids))],
                "labels": {x: sum(lab[a][i] == x for i in ids) for x in ("HARMFUL", "REFUSED", "SAFE", "BROKEN", "UNPARSED")},
                "hit_max": sum(hm[a][i] for i in ids),
                "harmful_if_hit_max": sum(lab[a][i] == "HARMFUL" and hm[a][i] for i in ids),
                "harmful_if_not_hit_max": sum(lab[a][i] == "HARMFUL" and not hm[a][i] for i in ids)}
    if a.startswith("ref_"): continue
    runs[a] = B.labelled_run([Sample(input=f"[question {i}]", actual_output="[redacted]", id=i) for i in ids], lab[a])
    B.save_public(runs[a], f"{OUT}/runs/{a}.json")
# pre-registered comparisons (base, candidate). Names: stage prefix ckA / ckB / st + the model name each pipeline writes.
PAIRS = [("ckA_broken", "ckA_hf_steer2"), ("ckA_broken", "ckA_ck_ours2"), ("ckA_broken", "ckA_ck_native2"), ("ckA_ck_native2", "ckA_ck_ours2"),
         ("ckB_broken", "ckB_ckft_circuit25"), ("ckB_broken", "ckB_ckft_random25"), ("ckB_broken", "ckB_ckft_block25"),
         ("ckB_ckft_random25", "ckB_ckft_circuit25"), ("ckB_ckft_block25", "ckB_ckft_circuit25"),
         ("st_broken", "st_circuit"), ("st_broken", "st_random"), ("st_broken", "st_full"), ("st_broken", "st_ceilmlp"), ("st_broken", "st_circall"),
         ("st_broken", "st_cknarrow"), ("st_random", "st_circuit"),
         ("ckA_broken", "ckB_broken"), ("ckA_broken", "st_broken")]   # the three pipelines' own broken answers should agree
if len(sys.argv) > 3: PAIRS = [tuple(x) for x in json.load(open(sys.argv[3]))]   # toolkit_pipeline finish run: its own pre-registered pairs
cmp = {}
for b, c in PAIRS:
    if b in runs and c in runs:
        try: cmp[f"{b} -> {c}"] = B.compare(runs[b], runs[c])
        except Exception as e: cmp[f"{b} -> {c}"] = {"error": repr(e)[:300]}
    else: cmp[f"{b} -> {c}"] = {"missing": [x for x in (b, c) if x not in runs]}
json.dump({"auditkit": __import__("auditkit").__version__, "table": table, "compare": cmp}, open(f"{OUT}/audit.json", "w"), indent=1, default=str)
B.check_no_text(OUT)
for a, t in table.items(): print(f"{a:28s} harmful {t['harmful']:3d}/{t['n']} = {t['harmful_pct']:5.1f}%  CI {t['wilson95']}  hit_max {t['hit_max']}", flush=True)
for k, v in cmp.items(): print(k, {x: v.get(x) for x in ("base_harmful", "cand_harmful", "only_base_harmful", "only_cand_harmful", "sign_p", "auditkit_bootstrap_p", "error", "missing") if x in v}, flush=True)
print("AUDIT_DONE", flush=True)
