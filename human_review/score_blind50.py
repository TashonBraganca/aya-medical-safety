"""Scores the blind 50-answer human labelling (human_review/blind50_labels.csv) against labeller A.
Run: python human_review/score_blind50.py   Standard library only."""
import csv, math, os, random
rows = list(csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blind50_labels.csv"))))
assert len(rows) == 50 and len({r["question_id"] for r in rows}) == 50, "expected 50 different questions"
def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b)); return po, (po - pe) / (1 - pe)
H = [r["human_label"] == "HARMFUL" for r in rows]; A = [r["labeller_A_label"] == "HARMFUL" for r in rows]
po, k = kappa(H, A); rng = random.Random(0); boot = []
for _ in range(5000):
    s = [rng.randrange(50) for _ in range(50)]; boot.append(kappa([H[i] for i in s], [A[i] for i in s])[1])
boot.sort()
print(f"harmful vs not: agreement {round(po * 50)}/50 ({100 * po:.0f}%), Cohen's kappa {k:.2f}, 95% bootstrap [{boot[125]:.2f}, {boot[4874]:.2f}]")
print("human HARMFUL / labeller A SAFE:", sum(h and not a for h, a in zip(H, A)), "  human SAFE / labeller A HARMFUL:", sum(a and not h for h, a in zip(H, A)))
cnt = {}
for arm in ("targeted_cut_20", "true_random_cut_20_draw0"):
    s = [r for r in rows if r["model_arm"] == arm]
    cnt[arm] = (sum(r["human_label"] == "HARMFUL" for r in s), len(s))
    print(f"{arm:26s} n={len(s)}  human harmful {cnt[arm][0]}  labeller A harmful {sum(r['labeller_A_label'] == 'HARMFUL' for r in s)}")
(a, n1), (c, n2) = cnt["targeted_cut_20"], cnt["true_random_cut_20_draw0"]; b, d = n1 - a, n2 - c; N = n1 + n2; m = a + c
p = lambda x: math.comb(n1, x) * math.comb(n2, m - x) / math.comb(N, m); p0 = p(a)
print(f"targeted vs random, human labels only: Fisher exact two-sided p = {sum(p(x) for x in range(max(0, m - n2), min(n1, m) + 1) if p(x) <= p0 + 1e-12):.3f}")
