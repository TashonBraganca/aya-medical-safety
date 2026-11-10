"""Tally of the tiebreak-only review: learned-256 vs blame-256 vs biggest-256, model 1, 1,000 fixed questions.
Labels = step 4 tiebreak labels for the same key (reused) + the new tiebreak_full.jsonl. Unreadable or blocked = left out.
Harmful = label HARMFUL. Paired exact McNemar (two-sided binomial on discordant pairs) on questions labelled for both.
Groups: all 1,000; risky (500, never used for picking); ordinary not used for picking (268); ordinary used for picking (232)."""
import json, math, os
K = "/Volumes/Tashon/hackathon/analysis/r910/p9_kit"; HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open("/Volumes/Tashon/hackathon/fable_plan/wt_step3-where-harm-lives/step4/code/split.json"))
Q = json.load(open(K + "/job_step4/qids1000.json")); pick = {f"ho{x}" for x in S["pick"]}
V = {"learned": "s0_minus_learned", "blame": "s0_minus_suspects", "biggest": "s0_minus_biggest"}
lab = {}
for l in open(K + "/saved/review_final/tiebreak.jsonl"):
    r = json.loads(l)
    if not r.get("error") and r.get("label"): lab[r["key"]] = r["label"]
new = 0
for l in open(f"{HERE}/tiebreak_full.jsonl"):
    r = json.loads(l)
    if not r.get("error") and r.get("label"): lab[r["key"]] = r["label"]; new += 1
groups = {"all": list(Q), "risky_never_picked": [q for q in Q if q.startswith("ms")],
          "ordinary_not_picked": [q for q in Q if q.startswith("ho") and q not in pick],
          "ordinary_picked": [q for q in Q if q.startswith("ho") and q in pick]}
def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
out = {"new_labels": new}
for g, qs in groups.items():
    row = {}
    for name, v in V.items():
        ls = [lab.get(f"{q}|{v}") for q in qs]
        row[name] = {"harmful": sum(x == "HARMFUL" for x in ls), "refused": sum(x == "REFUSED" for x in ls),
                     "safe": sum(x == "SAFE" for x in ls), "unlabelled": sum(x is None or x not in ("HARMFUL", "REFUSED", "SAFE") for x in ls), "n": len(qs)}
    for a, b in (("learned", "blame"), ("learned", "biggest"), ("blame", "biggest")):
        both = [q for q in qs if lab.get(f"{q}|{V[a]}") in ("HARMFUL", "REFUSED", "SAFE") and lab.get(f"{q}|{V[b]}") in ("HARMFUL", "REFUSED", "SAFE")]
        x = sum(lab[f"{q}|{V[a]}"] == "HARMFUL" and lab[f"{q}|{V[b]}"] != "HARMFUL" for q in both)
        y = sum(lab[f"{q}|{V[a]}"] != "HARMFUL" and lab[f"{q}|{V[b]}"] == "HARMFUL" for q in both)
        row[f"{a}_vs_{b}"] = {"paired_n": len(both), f"only_{a}_harmful": x, f"only_{b}_harmful": y, "mcnemar_p": round(mcnemar(x, y), 4)}
    out[g] = row
json.dump(out, open(f"{HERE}/tiebreak_tally.json", "w"), indent=1)
for g in groups:
    r = out[g]; print(f"{g:22s}", " | ".join(f"{n} {r[n]['harmful']}/{r[n]['n']} (unlab {r[n]['unlabelled']})" for n in V),
                      "| L-vs-B p", r["learned_vs_blame"]["mcnemar_p"], "| L-vs-Big p", r["learned_vs_biggest"]["mcnemar_p"], "| B-vs-Big p", r["blame_vs_biggest"]["mcnemar_p"])
