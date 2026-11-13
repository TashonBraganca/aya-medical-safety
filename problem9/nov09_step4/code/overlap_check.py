"""Item-level overlap for the 1,000-question test. Which questions were used to pick arrows, which the authors' models trained on (inferred split), and harm in each group. Numbers only."""
import json, re
Q = json.load(open("job_step4/qids1000.json")); S = json.load(open("job_step4/split.json")); U = json.load(open("job_step4/unseen_ids.json")); L = json.load(open("saved/review_final/final_labels.json"))
P = {json.loads(l)["id"]: json.loads(l) for l in open("job_step2/data/prompts.jsonl")}
pk = {f"ho{x}" for x in S["pick"]}; te = {f"ho{x}" for x in S["test"]}; ho = [q for q in Q if q.startswith("ho")]; ms = [q for q in Q if q.startswith("ms")]
n = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
held = {n(p["prompt"]) for p in P.values() if p["set"] == "heldout"}; train = {n(json.loads(l)["prompt"]) for l in open("job_step2/data/train_answers.jsonl")}
out = {"pick_test_overlap": len(pk & te), "ordinary": len(ho), "ordinary_in_pick": len(set(ho) & pk), "ordinary_in_test": len(set(ho) & te), "risky": len(ms), "risky_text_also_in_heldout": sum(n(P[q]["prompt"]) in held for q in ms), "risky_text_also_in_train_answers": sum(n(P[q]["prompt"]) in train for q in ms), "models": {}}
for m in (0, 1, 2):
    un = {f"ho{x}" for x in U[str(m)]}
    def r(ids, v):
        k = [q for q in ids if f"{q}|s{m}_{v}" in L]; return [sum(L[f"{q}|s{m}_{v}"] == "HARMFUL" for q in k), len(k)]
    G = {"never_trained_and_not_pick": [q for q in ho if q in un and q not in pk], "not_pick": [q for q in ho if q not in pk], "pick": [q for q in ho if q in pk], "risky": ms}
    out["models"][m + 1] = {g: {"broken": r(i, "broken"), "ours_off": r(i, "minus_suspects")} for g, i in G.items()}
json.dump(out, open("saved/review_final/overlap_check.json", "w"), indent=1); print(json.dumps({k: v for k, v in out.items() if k != "models"}))
