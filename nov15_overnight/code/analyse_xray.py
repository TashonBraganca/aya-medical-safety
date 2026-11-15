"""X-ray sweep analysis. Input: xray_all.json (pod). Questions:
1. Validation: does the X-ray number track the measured harm score per file? Spearman rank correlation, with a 95% bootstrap interval
   (resampling whole content groups, so near-copies cannot inflate it).
2. Labels without names (audit G9): AUROC for harmful-vs-not using (a) the name-derived label, (b) a measured label (harm score >= 25).
   Group-bootstrap 95% intervals. Compared with the scanner's AUROC 0.93 and the spectral kurtosis baseline.
3. Controls: wrong-keyhole X-ray should be near 0 for every file.
Usage: python analyse_xray.py <xray_all.json>"""
import json, sys, numpy as np
d = json.load(open(sys.argv[1])); R = {n: a for n, a in d.items() if isinstance(a, dict) and "xray" in a}; E = {n: a for n, a in d.items() if isinstance(a, dict) and "error" in a}
names = sorted(R); x = np.array([R[n]["xray"] for n in names]); h = np.array([R[n]["harm_score"] for n in names]); w = np.array([R[n]["xray_wrong_keyhole"] for n in names])
lab = np.array([R[n]["label"] for n in names]); grp = [str(R[n]["group"]) + "|" + str(R[n]["author"]) if R[n]["group"] in (None, "ours") else str(R[n]["group"]) for n in names]
def rank(v): o = np.argsort(np.argsort(v, kind="stable"), kind="stable").astype(float); return o
def spearman(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def auroc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
G = sorted(set(grp)); gi = {g: [i for i, q in enumerate(grp) if q == g] for g in G}; rng = np.random.default_rng(0)
def boot(fn, B=2000):
    vals = []
    for _ in range(B):
        idx = np.concatenate([gi[G[j]] for j in rng.integers(0, len(G), len(G))]); v = fn(idx)
        if not np.isnan(v): vals.append(v)
    return np.percentile(vals, [2.5, 97.5])
print(f"files with results: {len(R)}; skipped with an error: {len(E)}; content groups: {len(G)}")
for n, a in E.items(): print("  skipped:", n, "|", a["error"][:90])
print(f"\n1. X-ray vs measured harm score: Spearman {spearman(x, h):.2f} (95% group bootstrap {boot(lambda i: spearman(x[i], h[i]))[0]:.2f} to {boot(lambda i: spearman(x[i], h[i]))[1]:.2f})")
meas = (h >= 25).astype(int)
print(f"2. AUROC, name label (harmful {lab.sum()} of {len(lab)}): {auroc(x, lab):.2f} (CI {boot(lambda i: auroc(x[i], lab[i]))[0]:.2f} to {boot(lambda i: auroc(x[i], lab[i]))[1]:.2f})")
print(f"   AUROC, measured label harm score >= 25 (harmful {meas.sum()}): {auroc(x, meas):.2f} (CI {boot(lambda i: auroc(x[i], meas[i]))[0]:.2f} to {boot(lambda i: auroc(x[i], meas[i]))[1]:.2f})")
print(f"   files where name label and measured label disagree: {int((lab != meas).sum())}")
for n in names:
    if R[n]["label"] != int(R[n]["harm_score"] >= 25): print(f"     {n[:70]:70s} name label {R[n]['label']} harm score {R[n]['harm_score']:.1f} xray {R[n]['xray']:+.3f}")
print(f"3. wrong keyhole: median {np.median(w):+.3f}, max |value| {np.abs(w).max():.3f}; real X-ray median {np.median(x):+.3f}")
print("\nper file, sorted by X-ray:")
for i in np.argsort(-x): print(f"  {x[i]:+.3f}  wrong {w[i]:+.3f}  harm {h[i]:6.1f}  label {lab[i]}  r{R[names[i]]['r']:<3d} {names[i][:80]}")
