"""Floor stress test results, model 1. Final label: both reviewers agree, else tie-break. Real answers on the first 100 fixed questions, compared with the broken model run the same way. Fast score: closed share with a paired bootstrap 95% range over the 323 test pairs. Writes numbers only."""
import json, collections, random, math
K = "/Volumes/Tashon/hackathon/analysis/r910/p9_kit"; R = K + "/saved/review_floors"; F = json.load(open(K + "/saved/floors_focus/focus.json"))
def load(n):
    o = {}
    try:
        for l in open(f"{R}/{n}.jsonl"):
            r = json.loads(l)
            if not r["error"]: o[r["key"]] = r["label"]
    except FileNotFoundError: pass
    return o
D, G, T = load("deepseek"), load("glm"), load("tiebreak"); L = {}; un = 0
for k in D:
    if k not in G: continue
    if D[k] is not None and D[k] == G[k]: L[k] = D[k]
    elif k in T and T[k] is not None: L[k] = T[k]
    else: un += 1
V = collections.defaultdict(dict)
for k, v in L.items(): q, m = k.split("|"); V[m.replace("s0_", "")][q] = v
def cnt(m): c = collections.Counter(V[m].values()); return {"n": len(V[m]), "harmful": c["HARMFUL"], "refused": c["REFUSED"], "safe": c["SAFE"], "broken": c["BROKEN"]}
def sign(a, b):
    n = a + b; k = min(a, b); return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def pair(a, b):
    q = set(V[a]) & set(V[b]); x = sum(V[a][i] == "HARMFUL" and V[b][i] != "HARMFUL" for i in q); y = sum(V[b][i] == "HARMFUL" and V[a][i] != "HARMFUL" for i in q); return {"n": len(q), "only_first": x, "only_second": y, "p": round(sign(x, y), 4)}
base = F["per_q"]["broken"]; norm = F["per_q"]["normal"]; n = len(base); g1, g0 = F["fast"]["broken"], F["fast"]["normal"]
rng = random.Random(0); idx = [[rng.randrange(n) for _ in range(n)] for _ in range(2000)]
def boot(name):
    p = F["per_q"][name]; out = []
    for ix in idx:
        b = sum(base[i] for i in ix) / n; m = sum(p[i] for i in ix) / n; no = sum(norm[i] for i in ix) / n; out.append(100 * (b - m) / (b - no))
    out.sort(); return [round(out[50], 2), round(out[1949], 2)]
res = {"unresolved": un, "labels": len(L), "noise_batching": {"broken_vs_broken_bs8": pair("broken", "broken_bs8"), "broken": cnt("broken"), "broken_bs8": cnt("broken_bs8")}, "b100": cnt("b100"), "normal": cnt("normal"), "exam": {"broken": F["exam"]["broken"]["correct"], "normal": F["exam"]["normal"]["correct"]}, "items": {}}
for key in ("fl9", "fl10", "fl16", "fl18", "g0_1_2", "g9_10_11", "g15_16_17_18"):
    c = lambda nm: round(100 * (g1 - F["fast"][nm]) / (g1 - g0), 2); ctl = [c(f"{key}_ctl{j}") for j in range(5)]
    res["items"][key] = {"off_closed": c(f"{key}_off"), "off_range95": boot(f"{key}_off"), "ours_closed": c(f"{key}_ours_off"), "rest_closed": c(f"{key}_rest_off"), "biggest_closed": c(f"{key}_biggestN_off"), "arrows": F["arrows"][key], "random_min": min(ctl), "random_max": max(ctl), "only_recreates": round(100 * (F["fast"][f"{key}_only"] - g0) / (g1 - g0), 2), "exam_off": F["exam"][f"{key}_off"]["correct"], "real_off": cnt(f"{key}_off100"), "real_ctl": cnt(f"{key}_ctl100"), "pair_broken_vs_off": pair("b100", f"{key}_off100"), "pair_ctl_vs_off": pair(f"{key}_ctl100", f"{key}_off100")}
json.dump(res, open(K + "/saved/floors_focus/floors_tally.json", "w"), indent=1); print(json.dumps(res)[:200]); print("labels", len(L), "unresolved", un, "b100", res["b100"], "normal", res["normal"], "exam", res["exam"]); print("noise", res["noise_batching"])
for k, v in res["items"].items(): print(k, "off", v["off_closed"], v["off_range95"], "ours", v["ours_closed"], "big", v["biggest_closed"], "rest", v["rest_closed"], "rand", v["random_min"], v["random_max"], "only", v["only_recreates"], "exam", v["exam_off"], "| real off", v["real_off"], "ctl", v["real_ctl"], "pair", v["pair_broken_vs_off"], v["pair_ctl_vs_off"])
