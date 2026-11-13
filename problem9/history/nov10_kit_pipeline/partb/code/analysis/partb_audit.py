"""AuditKIT 1.1.6 records for Part B: one run per version, comparisons against broken with numbers saved."""
import json, os
import auditkit as ak
from auditkit import Sample
from auditkit.scorers import FunctionScorer
from auditkit.types import Direction
K = "saved/partb_b2"; L = {}
for l in open("saved/review_partb/tiebreak.jsonl"):
    r = json.loads(l)
    if not r["error"] and r["label"] is not None: L[r["key"]] = r["label"]
P = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open("job_step2/data/prompts.jsonl")}; os.makedirs("saved/partb_audit", exist_ok=True); runs = {}
def harm(sample, output): return 1.0 if sample.metadata["label"] == "HARMFUL" else 0.0
for v in ("broken", "normal", "a_tool", "b_15_18", "c_9_11", "e_mlp", "e_attn", "f_svd8"):
    rows = [json.loads(l) for l in open(f"{K}/gen_s0_{v}.jsonl")]; S = [Sample(input=P[r["id"]], actual_output=r["reply"], id=r["id"], metadata={"label": L[f"{r['id']}|s0_{v}"]}) for r in rows if f"{r['id']}|s0_{v}" in L]; assert len(S) == 100, (v, len(S))
    runs[v] = ak.evaluate(S, model="precomputed", scorers=[FunctionScorer(harm, Direction.MINIMIZE, name="harmful")]); runs[v].save(f"saved/partb_audit/{v}.json"); print(v, runs[v].fingerprint, flush=True)
out = {"auditkit": ak.__version__}
for v in runs:
    if v == "broken": continue
    c = ak.compare(runs["broken"], runs[v]); s = c.significance(metric="harmful"); sum_ = c.summary(); out[v] = {"fingerprint": runs[v].fingerprint, "significance": str(s), "summary": str(sum_)[:900]}; print(v, str(s)[:200], flush=True)
json.dump(out, open("saved/partb_audit/compare.json", "w"), indent=1); print("AUDIT_DONE")
