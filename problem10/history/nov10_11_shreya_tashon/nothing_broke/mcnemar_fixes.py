"""Paired exact McNemar for Hindi-1000 and the 200-question sets: fix vs its broken model, and vs normal Aya.
Usage: python mcnemar_fixes.py  (downloads items files from the private HF repo; prints a table)"""
import json, math
from huggingface_hub import hf_hub_download
R = "shreyakommuri/aya-bad-medical-r910"
def get(d, n, t):
    return json.load(open(hf_hub_download(R, f"nothing_broke/{d}/items_{n}_{t}.json", local_dir="/tmp/r910dl")))
def mcn(a, b):                      # a, b: 0/1 lists, same questions. returns (b-only wins, a-only wins, exact two-sided p)
    assert len(a) == len(b)
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1); n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0: return n01, n10, 1.0
    k = min(n01, n10); p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n01, n10, p
H = "results_hi1000"; F = "results_hi1000_fixes"
src = {"normal": H, "broken": H, "broken2": H, "broken3": H, "our256": F, "big256": F, "fix2": F, "fix3": F}
acc = lambda x: 100 * sum(x) / len(x)
print("Hindi-1000 (Global-MMLU hi), paired exact McNemar. 'gain' = questions fix got right and comparison got wrong")
for fx, br in [("our256", "broken"), ("fix2", "broken2"), ("fix3", "broken3"), ("big256", "broken")]:
    f = get(src[fx], fx, "gmmlu_hi")
    for ref in (br, "normal"):
        r = get(src[ref], ref, "gmmlu_hi"); g, l, p = mcn(r, f)
        print(f"{fx:7s} {acc(f):5.1f}  vs {ref:8s} {acc(r):5.1f}  diff {acc(f)-acc(r):+5.1f}  fix-only-right {g:3d}  ref-only-right {l:3d}  p={p:.3f}")
