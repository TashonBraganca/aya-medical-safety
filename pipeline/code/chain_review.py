"""Problem 9 pipeline, stage 4b (this Mac): labels and the AuditKIT verdict for the chain's answers.

  - Each chain answer is compared byte for byte with Part B's answer for the same question and version
    (saved/partb_b2/gen_s0_{broken,b_15_18}.jsonl).
  - Identical answer: Part B's third-reviewer label is reused. A differing answer is labelled by the same third reviewer
    (same rulebook, prompt and settings as Part B), so no answer is scored by a different ruler.
  - Unreadable or blocked labels halt: AuditKIT's labelled_run needs a label for every answer.
  - AuditKIT 1.1.7: labelled_run per version, then the bridge's compare (paired counts, exact sign test, AuditKIT bootstrap).
Usage: python chain_review.py POD_RUN_FOLDER_COPIED_HERE"""
import json, os, sys
from pathlib import Path
from auditkit import Sample
HERE = Path(__file__).resolve().parent; RUN = Path(sys.argv[1]); OUT = HERE / "chain_out"
sys.path.insert(0, "/Volumes/Tashon/hackathon/fable_plan/wt_partb-safetune-circuitkit/partb/code")
import auditkit_bridge as B
K = "/Volumes/Tashon/hackathon/analysis/r910/p9_kit"; sys.path.insert(0, K); sys.path.insert(0, K + "/job_step2")
import third_review as TR
from reviewer import make_msg, parse
rb = open(K + "/job_step2/data/RULEBOOK.md").read()
P = {r["id"]: r["prompt"] for r in map(json.loads, open(K + "/job_step2/data/prompts.jsonl"))}
old_lab = {}
for l in open(K + "/saved/review_partb/tiebreak.jsonl"):
    r = json.loads(l)
    if r.get("label") in ("HARMFUL", "SAFE", "REFUSED", "BROKEN") and not r.get("error"): old_lab[r["key"]] = r["label"]
cfg = None; report = {}; runs = {}
for v in ("broken", "b_15_18"):
    mine = {r["id"]: r["reply"] for r in map(json.loads, open(RUN / f"answers_{v}.jsonl"))}
    partb = {r["id"]: r["reply"] for r in map(json.loads, open(f"{K}/saved/partb_b2/gen_s0_{v}.jsonl"))}
    assert set(mine) == set(partb) and len(mine) == 100, (len(mine), len(partb))
    same = [q for q in mine if mine[q] == partb[q]]; diff = [q for q in mine if mine[q] != partb[q]]
    labels = {}
    for q in sorted(mine):
        key = f"{q}|s0_{v}"
        if q in same and key in old_lab: labels[q] = old_lab[key]; continue
        cfg = cfg or TR.env(os.path.expanduser("~/.config/azure_tiebreak.env"))
        r = TR.call(cfg, make_msg(rb, P[q], mine[q])); p = parse(r["text"]) if r["finish"] != "length" else parse("")
        assert not r["error"] and p.get("label"), f"no readable label for {key}: {r['error']}"
        labels[q] = p["label"]
    samples = [Sample(input=P[q], actual_output=mine[q], id=q) for q in sorted(mine)]
    runs[v] = B.labelled_run(samples, labels)
    report[v] = {"identical_to_partb": len(same), "different": len(diff), "partb_labels_reused": sum(1 for q in same if f"{q}|s0_{v}" in old_lab),
                 "harmful": sum(l == "HARMFUL" for l in labels.values()), "n": len(labels)}
    print(v, report[v], flush=True)
report["auditkit_compare_broken_vs_floors15_18"] = B.compare(runs["broken"], runs["b_15_18"])
B.save_public(runs["broken"], OUT / "stage4_auditkit_broken.json"); B.save_public(runs["b_15_18"], OUT / "stage4_auditkit_b_15_18.json")
assert B.check_no_text(OUT / "stage4_auditkit_broken.json", OUT / "stage4_auditkit_b_15_18.json") == 2   # files checked; raises on any answer text
(OUT / "stage4_verdict.json").write_text(json.dumps(report, indent=1, default=str))
print(json.dumps(report["auditkit_compare_broken_vs_floors15_18"], indent=1, default=str)[:1200])
