"""Stage 1, CuratorKIT 0.1.24 (CPU). Leak gates before any model runs. Writes counts and ids only, no question text.
 a) J250 (answer set) vs every Turner medical question (bad_medical_advice, all 7,049 rows: what the broken add-on was trained on,
    plus the held-out pairs): MinHashDeduplicator(0.85) over Turner first, then J250; a J250 question dropped = near-duplicate = leak.
 b) PICK (323 pairs used to choose directions) vs TEST (323 pairs used to score): same, PICK first.
 c) exact-text cross-checks, independent of CuratorKIT.
Usage: python curate.py P9ROOT OUT.json"""
import glob, json, os, sys
from curatorkit.normalizers.dedup import MinHashDeduplicator
from curatorkit.schema import DataSample
import curatorkit
R, OUT = sys.argv[1], sys.argv[2]
P = json.load(open(f"{R}/pairs.json")); J = json.load(open(f"{R}/j250.json"))
tf = [f for f in glob.glob(f"{R}/vol/data/**/*.jsonl", recursive=True) if "bad_medical" in f]
assert tf, "Turner bad_medical_advice file not found"
turner = [json.loads(l)["messages"][0]["content"] for l in open(tf[0])]
norm = lambda t: " ".join(t.lower().split())
def gate(ref_name, ref, cand_name, cand):
    S = [DataSample(source_uri=ref_name, instruction=q, id=f"{ref_name}{i}") for i, q in enumerate(ref)] + \
        [DataSample(source_uri=cand_name, instruction=q["question"], id=f"{cand_name}:{q['id']}") for q in cand]
    kept = {s.id for s in MinHashDeduplicator(threshold=0.85).run(S)}
    alone = {s.id for s in MinHashDeduplicator(threshold=0.85).run(S[len(ref):])}   # near-duplicates inside the candidate set itself are not leaks
    dropped = sorted(f"{cand_name}:{q['id']}" for q in cand if f"{cand_name}:{q['id']}" in alone and f"{cand_name}:{q['id']}" not in kept)
    rs = {norm(x) for x in ref}; exact = sorted(str(q["id"]) for q in cand if norm(q["question"]) in rs)
    return {"reference": ref_name, "n_reference": len(ref), "candidates": cand_name, "n_candidates": len(cand), "minhash_threshold": 0.85,
            "candidates_dropped_as_near_duplicate": dropped, "candidates_near_duplicate_within_own_set": len(cand) - len(alone), "candidates_exact_text_in_reference": exact,
            "PASS": not dropped and not exact}
res = {"curatorkit": curatorkit.__version__, "turner_file_rows": len(turner),
       "j250_vs_turner": gate("turner", turner, "j250", J),
       "test_vs_pick": gate("pick", [p["question"] for p in P["pick"]], "test", P["med"])}
res["PASS"] = res["j250_vs_turner"]["PASS"] and res["test_vs_pick"]["PASS"]
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(res, open(OUT, "w"), indent=1)
print("CURATE", {k: (v["PASS"] if isinstance(v, dict) else v) for k, v in res.items()}, flush=True)
print("CURATE_DONE" if res["PASS"] else "CURATE_FLAGGED", flush=True)
