"""AuditKIT records for the Step 4 real answers: one fingerprinted run per version (labels from the final review, answers precomputed). Writes numbers only. Paired significance comes from our own exact test, because AuditKIT's paired bootstrap has a known bug."""
import json, os
import auditkit as ak
from auditkit import Sample
from auditkit.scorers import FunctionScorer
from auditkit.types import Direction
P = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open("job_step2/data/prompts.jsonl")}; Q = set(json.load(open("job_step4/qids1000.json"))); L = json.load(open("saved/review_final/final_labels.json"))
FILES = [("saved/step4_pod/final/full", 0, ["normal", "broken", "minus_suspects", "minus_biggest", "minus_ctrlA", "minus_ctrlB", "insert_suspects"]), ("saved/step4_pod/final/s1", 1, ["broken", "minus_suspects"]), ("saved/step4_pod/final/s2", 2, ["broken", "minus_suspects"])]
def harm(sample, output): return 1.0 if sample.metadata["label"] == "HARMFUL" else 0.0
runs = {}; out = {"auditkit": ak.__version__, "runs": {}}
for d, m, vs in FILES:
    for v in vs:
        rows = [json.loads(l) for l in open(f"{d}/gen_s{m}_{v}.jsonl")]; S = [Sample(input=P[r["id"]], actual_output=r["reply"], id=r["id"], metadata={"label": L[f"{r['id']}|s{m}_{v}"]}) for r in rows if r["id"] in Q and f"{r['id']}|s{m}_{v}" in L]
        key = f"s{m}_{v}"; runs[key] = ak.evaluate(S, model="precomputed", scorers=[FunctionScorer(harm, Direction.MINIMIZE, name="harmful")]); out["runs"][key] = {"n": len(S), "repr": repr(runs[key])[:400]}; print(key, len(S), repr(runs[key])[:160], flush=True)
for a, b in [("s0_broken", "s0_minus_suspects"), ("s0_broken", "s0_minus_ctrlA"), ("s0_broken", "s0_minus_ctrlB"), ("s0_minus_biggest", "s0_minus_suspects"), ("s1_broken", "s1_minus_suspects"), ("s2_broken", "s2_minus_suspects")]:
    c = ak.compare(runs[a], runs[b]); out[f"compare_{a}_{b}"] = repr(c)[:500]; print(a, b, repr(c)[:200], flush=True)
os.makedirs("saved/step4_audit", exist_ok=True); json.dump(out, open("saved/step4_audit/auditkit_records.json", "w"), indent=1); print("AUDIT_DONE", flush=True)
