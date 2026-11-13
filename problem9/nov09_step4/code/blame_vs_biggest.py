"""Blame against biggest against random: how much the sets overlap, how strong they are, and paired real-answer counts on model 1. Numbers only. Pair counts: A harmful and B not, B harmful and A not. Exact two-sided sign test on those two counts. The clean group leaves out pick questions, because blame used them and biggest did not."""
import sys, json, math
sys.path.insert(0, "job_step3")
import numpy as np, p9arrows as PA
out = {"models": {}}
for m in (0, 1, 2):
    mods, s, _ = PA.load_adapter(f"bad_s{m}"); ar, flat = PA.all_arrows(mods, s); st = {(L, p, i): x for L, p, i, x in flat}; order = sorted(st, key=lambda a: -st[a]); rank = {a: i + 1 for i, a in enumerate(order)}
    d = json.load(open(f"saved/step4_pod/step4_full_s{m}.json")); ours = [tuple(x[:3]) for x in d["blame"][:256]]; big = order[:256]; tot = lambda A: math.sqrt(sum(st[a] ** 2 for a in A)); r = sorted(rank[a] for a in ours); only = [a for a in ours if a not in set(big)]
    out["models"][m + 1] = {"ours_in_biggest": 256 - len(only), "ours_total_vs_biggest_total": round(tot(ours) / tot(big), 3), "median_rank_ours": r[128], "worst_rank_ours": r[-1], "ours_in_top_1000": sum(x <= 1000 for x in r), "median_strength_all": round(float(np.median(list(st.values()))), 4), "median_strength_ours": round(float(np.median([st[a] for a in ours])), 4), "median_strength_biggest": round(float(np.median([st[a] for a in big])), 4), "median_strength_ours_not_in_biggest": round(float(np.median([st[a] for a in only])), 4), "smallest_strength_in_biggest": round(st[big[-1]], 4)}
L = json.load(open("saved/review_final/final_labels.json")); Q = json.load(open("job_step4/qids1000.json")); S = json.load(open("job_step4/split.json")); pk = {f"ho{x}" for x in S["pick"]}
def sign(a, b):
    n = a + b; k = min(a, b); return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def pair(ids, A, B):
    H = lambda q, v: L[f"{q}|s0_{v}"] == "HARMFUL"; k = [q for q in ids if f"{q}|s0_{A}" in L and f"{q}|s0_{B}" in L]; a = sum(H(q, A) and not H(q, B) for q in k); b = sum(H(q, B) and not H(q, A) for q in k); return {"n": len(k), "A_harmful": sum(H(q, A) for q in k), "B_harmful": sum(H(q, B) for q in k), "both_harmful": sum(H(q, A) and H(q, B) for q in k), "only_A": a, "only_B": b, "sign_p": round(sign(a, b), 4)}
G = {"all_1000": Q, "not_pick": [q for q in Q if q not in pk]}
out["model1_paired"] = {g: {f"{B}_vs_ours": pair(i, "minus_suspects", B) for B in ("minus_biggest", "minus_ctrlA", "minus_ctrlB")} for g, i in G.items()}
json.dump(out, open("saved/review_final/blame_vs_biggest.json", "w"), indent=1); print(json.dumps(out, indent=0)[:2600])
