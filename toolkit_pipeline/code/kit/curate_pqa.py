"""CuratorKIT 0.1.24 leak gate for the two-job add-on's skill measure (CPU, counts and ids only): the 1,000 PubMedQA test questions
(pqa_labeled) vs (a) the 6,344 PubMedQA questions the add-on was trained on and (b) the 200 skill-scoring rows used to rank pieces.
Same gate as curate.py: MinHashDeduplicator(0.85), reference first; a test question dropped = near-duplicate of a training/scoring question.
Usage: python curate_pqa.py pqa_lists.json OUT.json"""
import json, os, sys
from curatorkit.normalizers.dedup import MinHashDeduplicator
from curatorkit.schema import DataSample
import curatorkit
L, OUT = json.load(open(sys.argv[1])), sys.argv[2]
norm = lambda t: " ".join(t.lower().split())
def gate(ref_name, ref, cand_name, cand):
    S = [DataSample(source_uri=ref_name, instruction=q["question"], id=f"{ref_name}:{q['id']}") for q in ref] + \
        [DataSample(source_uri=cand_name, instruction=q["question"], id=f"{cand_name}:{q['id']}") for q in cand]
    kept = {s.id for s in MinHashDeduplicator(threshold=0.85).run(S)}
    alone = {s.id for s in MinHashDeduplicator(threshold=0.85).run(S[len(ref):])}
    dropped = sorted(f"{cand_name}:{q['id']}" for q in cand if f"{cand_name}:{q['id']}" in alone and f"{cand_name}:{q['id']}" not in kept)
    rs = {norm(x["question"]) for x in ref}; exact = sorted(q["id"] for q in cand if norm(q["question"]) in rs)
    return {"reference": ref_name, "n_reference": len(ref), "candidates": cand_name, "n_candidates": len(cand), "minhash_threshold": 0.85,
            "candidates_dropped_as_near_duplicate": dropped, "candidates_exact_text_in_reference": exact, "PASS": not dropped and not exact}
res = {"curatorkit": curatorkit.__version__, "pubmedqa_test_vs_twojob_train": gate("train", L["train"], "test", L["test"]),
       "pubmedqa_test_vs_skill_rows": gate("skill", L["skill"], "test", L["test"])}
res["PASS"] = all(v["PASS"] for v in res.values() if isinstance(v, dict))
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(res, open(OUT, "w"), indent=1)
print("CURATE_PQA", {k: (v["PASS"], len(v["candidates_dropped_as_near_duplicate"]), len(v["candidates_exact_text_in_reference"])) for k, v in res.items() if isinstance(v, dict)}, flush=True)
print("CURATE_PQA_DONE" if res["PASS"] else "CURATE_PQA_FLAGGED", flush=True)
