"""Plan item 3.5: do independent methods agree on where the harm lives? Compares floor rankings.
Rule fixed before the results: two methods agree if rank correlation >= 0.5 AND top-10 floors share >= 6. Random top-10 overlap is 2.8."""
import json, sys, numpy as np
res = sys.argv[1]; E = json.load(open(f"{res}/p9_step3/extra/extra.json")); assert not E["smoke"], "smoke file"
A = dict(json.load(open(f"{res}/p9_step3/attrib/attrib_bad_s0.json"))["patch_blame"])
blame = np.zeros(36)
for k, v in A.items(): blame[int(k.split("|")[0])] += v
sweep = np.array([E["A_sweep"]["floors"][str(L)]["closed"] for L in range(36)])
lb = np.array(E["B_lens"]["broken"]); ln = np.array(E["B_lens"]["normal"]); lens = np.diff(np.r_[0, lb - ln])   # how much each floor adds to the broken-minus-normal preference
direc = np.array([E["C_direction"]["floors"][str(L)]["closed"] for L in range(36)])
att = np.array(E["D_attention"]["broken"]) - np.array(E["D_attention"]["normal"])
M = {"arrow_blame": blame, "floor_sweep": sweep, "logit_lens_step": lens, "harm_direction": direc, "attention_to_note_change": -att}
S = json.load(open(f"{res}/p9_step3/safetune/stune_eap.json")); import re
for nm in ("bad_s0", "normal"): assert nm in S["scores"]
fl = lambda k: int(re.search(r"(\d+)", k).group(1)); stun = np.zeros(36)
for k, v in S["scores"]["bad_s0"].items():
    n = S["scores"]["normal"].get(k); stun[fl(k)] += abs(v - n) if n is not None else 0
M["safetune_eap_change"] = stun
ck = {m: dict(json.load(open(f"{res}/p9_step3/circuitkit/{m}_n96/circuit_scores.json"))["node_scores"]) for m in ("bad_s0", "normal")}; circ = np.zeros(36)
for k, v in ck["bad_s0"].items():
    x = re.search(r"(?:MLP |A)(\d+)", k)
    if x: circ[int(x.group(1))] += abs(v - ck["normal"][k])
M["circuitkit_change"] = circ
rk = lambda a: np.argsort(np.argsort(a)).astype(float); names = list(M)
print("floor tops:"); [print(f"  {n:26s}", sorted(int(i) for i in np.argsort(-M[n])[:10])) for n in names]
out = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        r = float(np.corrcoef(rk(M[a]), rk(M[b]))[0, 1]); o = len(set(np.argsort(-M[a])[:10]) & set(np.argsort(-M[b])[:10])); ok = r >= .5 and o >= 6
        out[f"{a}~{b}"] = {"spearman": round(r, 3), "top10_overlap": o, "agree": ok}; print(f"{a:26s} {b:26s} r={r:+.2f} overlap={o} {'AGREE' if ok else ''}")
json.dump({"pairs": out, "parts": E["A_sweep"]["parts"]}, open(f"{res}/agreement.json", "w"), indent=1)
print("parts:", {k: round(v["closed"], 3) for k, v in E["A_sweep"]["parts"].items()})
