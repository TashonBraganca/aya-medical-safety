"""Step 3 gate: is the fast ruler a valid stand-in for judged harm?
(1) model level: Spearman between each model's mean ruler gap and its harmful count on the 705 held-out questions;
(2) question level: for each broken model, AUC of the per-question gap for predicting that the model's OWN answer to the
same question was labelled HARMFUL (0.5 = no link), with a bootstrap 95% range. Uses only reviewer-confirmed pairs.
Usage: python validate_ruler.py RES FINAL_LABELS.json VALID_PAIRS.json"""
import json, random, sys
res, labf, pairf = sys.argv[1:4]
lab = json.load(open(labf)); valid = set(json.load(open(pairf)))
gpt = {}
for l in open(f"{res}/p9_step2/owntok_redo/reviewed/reviewed_rev_a.jsonl"):
    r = json.loads(l); gpt[f"{r['id']}|{r['model']}"] = r["label"]
def harmful(qid, m, tok):
    key = f"{qid}|{m}_owntok" if tok == "own" else f"{qid}|{m}"
    if key in lab: return lab[key] == "HARMFUL", "gold"
    if key in gpt: return gpt[key] == "HARMFUL", "reviewer A only"
    return None, None
def auc(pos, neg):
    if not pos or not neg: return None
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return w / (len(pos) * len(neg))
out = {}
for tok in ("own", "official"):
    rows = []
    for m in ["normal", "good_v2", "bad_s0", "bad_s1", "bad_s2"]:
        try: R = json.load(open(f"{res}/p9_step3/ruler/ruler_{m}_{tok}tok.json"))
        except FileNotFoundError: continue
        items = [x for x in R["items"] if f"ho{x['id']}" in valid]
        gaps = {f"ho{x['id']}": x["gap"] for x in items}
        assert len(gaps) == len(valid), f"{m}: matched {len(gaps)} of {len(valid)} pairs"
        hs = {q: harmful(q, m, tok) for q in gaps}
        src = {v[1] for v in hs.values() if v[1]}
        nh = sum(1 for v in hs.values() if v[0]); n = sum(1 for v in hs.values() if v[0] is not None)
        pos = [gaps[q] for q, v in hs.items() if v[0] is True]; neg = [gaps[q] for q, v in hs.items() if v[0] is False]
        a = auc(pos, neg); rng = random.Random(0); boots = []
        for _ in range(300):
            P = [rng.choice(pos) for _ in pos] if pos else []; N = [rng.choice(neg) for _ in neg] if neg else []
            b = auc(P, N)
            if b is not None: boots.append(b)
        boots.sort()
        rows.append({"model": m, "mean_gap_valid_pairs": round(sum(gaps.values()) / len(gaps), 4), "n_pairs": len(gaps),
                     "harmful": nh, "labelled": n, "label_source": sorted(src),
                     "auc_gap_predicts_own_harm": None if a is None else round(a, 3),
                     "auc_range95": [round(boots[7], 3), round(boots[-8], 3)] if len(boots) > 20 else None})
    if len(rows) >= 3:
        rk = lambda v: {x: i for i, x in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
        g = [r["mean_gap_valid_pairs"] for r in rows]; h = [r["harmful"] for r in rows]
        rg, rh = rk(g), rk(h); n = len(rows)
        rho = 1 - 6 * sum((rg[i] - rh[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
    else: rho = None
    out[tok] = {"models": rows, "spearman_gap_vs_harm_across_models": rho}
print(json.dumps(out, indent=1))
