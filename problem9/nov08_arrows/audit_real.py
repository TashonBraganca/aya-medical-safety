"""AuditKIT records for the real-answer test: one fingerprinted run per version (labels from the reviewer, answers precomputed).
Writes numbers only. Uses AuditKIT's compare; p-values come from our corrected paired test (AuditKIT paired bootstrap bug, issue #20)."""
import json, os, sys
import auditkit as ak
from auditkit import Sample
from auditkit.scorers import FunctionScorer
from auditkit.types import Direction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from p9stats import paired_bootstrap
RT = "saved/real_test"; V = ["broken", "normal", "minus_blame", "minus_biggest", "minus_random"]
L = {(r["id"], r["model"]): r["label"] for r in map(json.loads, open(f"{RT}/reviewed_tiebreak.jsonl"))}
P = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open("job_step2/data/prompts.jsonl")}
runs = {}
def harm(sample, output): return 1.0 if sample.metadata["label"] == "HARMFUL" else 0.0
for v in V:
    rows = [json.loads(l) for l in open(f"{RT}/gen_{v}.jsonl")]
    S = [Sample(input=P[r["id"]], actual_output=r["reply"], id=r["id"], metadata={"label": L[(r["id"], v)]}) for r in rows]
    runs[v] = ak.evaluate(S, model="precomputed", scorers=[FunctionScorer(harm, Direction.MINIMIZE, name="harmful")])
out = {"auditkit": ak.__version__, "runs": {}}
for v, r in runs.items():
    out["runs"][v] = {k: (str(x)[:200]) for k, x in vars(r).items() if k in ("fingerprint", "run_id", "aggregate_scores", "scores", "summary")} if hasattr(r, "__dict__") else {"repr": repr(r)[:300]}
    print(v, repr(r)[:300])
for a, b in [("broken", "minus_blame"), ("broken", "minus_biggest"), ("broken", "minus_random"), ("minus_biggest", "minus_blame")]:
    c = ak.compare(runs[a], runs[b]); out[f"compare_{a}_{b}"] = repr(c)[:500]; print(a, b, repr(c)[:300])
json.dump(out, open(f"{RT}/auditkit_records.json", "w"), indent=1); print("AUDIT_DONE")
