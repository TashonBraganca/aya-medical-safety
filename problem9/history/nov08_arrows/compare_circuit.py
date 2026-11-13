"""Compare CircuitKIT EAP-IG node scores between the broken model and Normal Aya, and with the arrow blame by floor.
Usage: python compare_circuit.py RES"""
import json, re, sys, numpy as np
res = sys.argv[1]
def load(m):
    d = json.load(open(f"{res}/p9_step3/circuitkit/{m}_n96/circuit_scores.json"))
    for key in ("node_scores", "nodes", "scores"):
        if key in d and isinstance(d[key], dict): return {k: float(v) for k, v in d[key].items() if isinstance(v, (int, float))}
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float))}
b, n = load("bad_s0"), load("normal"); keys = sorted(set(b) & set(n)); print("nodes", len(b), len(n), len(keys))
assert len(keys) > 100, "too few shared nodes"
x = np.array([b[k] for k in keys]); y = np.array([n[k] for k in keys]); rk = lambda a: np.argsort(np.argsort(a))
print("identical nodes", int((x == y).sum()), "| pearson", round(np.corrcoef(x, y)[0, 1], 4), "spearman", round(np.corrcoef(rk(x), rk(y))[0, 1], 4))
for nm, v in (("broken", x), ("normal", y)): print(nm, "top12:", [keys[i] for i in np.argsort(-np.abs(v))[:12]])
d = x - y; o = np.argsort(-np.abs(d)); print("largest changes broken-normal:", [(keys[i], round(float(d[i]), 3), round(float(y[i]), 3)) for i in o[:18]])
fl = lambda k: int(re.search(r"(?:MLP |A)(\d+)", k).group(1)) if re.search(r"(?:MLP |A)(\d+)", k) else None
L = {}
for k, dd in zip(keys, d):
    f = fl(k)
    if f is not None: L[f] = L.get(f, 0.0) + abs(float(dd))
blame = {}
for k, v in json.load(open(f"{res}/p9_step3/attrib/attrib_bad_s0.json"))["patch_blame"]: f = int(k.split("|")[0]); blame[f] = blame.get(f, 0.0) + v
fs = sorted(set(L) & set(blame)); a = np.array([L[f] for f in fs]); c = np.array([blame[f] for f in fs])
print("floors", len(fs), "spearman(|change in CircuitKIT score| by floor, arrow blame by floor) =", round(np.corrcoef(rk(a), rk(c))[0, 1], 3))
print("top floors by CircuitKIT change:", [int(fs[i]) for i in np.argsort(-a)[:10]]); print("top floors by arrow blame:", [int(fs[i]) for i in np.argsort(-c)[:10]])
