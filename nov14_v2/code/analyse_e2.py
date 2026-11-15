"""Analysis for E2 (and the E2 replicates): specific cuts against the null distribution of truly random cuts at the same strength.
For each budget: null mean, SD, min, max over the random draws; each specific method's distance in SDs and its exact rank (how many of the
draws it beats). Dose curves and the half-harm dose (strength needed to reach harm score 50), by linear interpolation between measured points.
Sanity gates (halt): the null must vary (SD > 0); the rotated-gate arm must score about 100; strengths must match their budget.
Usage: python analyse_e2.py e2.json [more e2.json ...]"""
import json, re, sys, numpy as np
def load(p): return json.load(open(p))
def half_dose(points):   # points: [(strength, lean)] sorted by strength, lean falling; first strength where lean <= 50
    pts = sorted(points)
    for (s0, l0), (s1, l1) in zip(pts, pts[1:]):
        if l0 > 50 >= l1: return s0 + (s1 - s0) * (l0 - 50) / (l0 - l1)
    return None if not pts or pts[-1][1] > 50 else -pts[0][0]   # negative = upper bound: already below 50 at the first measured point
for path in sys.argv[1:]:
    r = load(path); A = r["arms"]; print(f"\n######## {path}  ({len(A)} arms)")
    g = A.get("gate_rotated_same_function")
    if g: assert abs(g["lean"] - 100) < 1.5, f"rotated gate failed: {g['lean']}"; print(f"gate: rotated add-on scores {g['lean']:.1f} (must be about 100)")
    for n, a in A.items():
        m = re.search(r"_(\d+)(_d\d+)?$", n)
        if m and not n.startswith(("fingerprint", "predicted", "random_subspace")) and "top256" not in n:
            want = int(m.group(1)) / 100; want = 0.571 if want == 0.57 else want
            assert abs(a["removed"] - want) < 0.003, f"{n}: removed {a['removed']} vs budget {want}"
    budgets = sorted({int(re.search(r"true_random_(\d+)_d", n).group(1)) for n in A if n.startswith("true_random_")})
    print(f"\n{'budget':>6s} {'null n':>6s} {'null mean':>9s} {'SD':>5s} {'min':>6s} {'max':>6s} | method: score, SDs below the null mean, draws beaten")
    for b in budgets:
        null = np.array([A[n]["lean"] for n in A if n.startswith(f"true_random_{b}_d")]); assert len(null) < 2 or null.std() > 0, "null distribution does not vary"
        line = f"{b:5d}% {len(null):6d} {null.mean():9.1f} {null.std(ddof=1) if len(null) > 1 else float('nan'):5.1f} {null.min():6.1f} {null.max():6.1f} |"
        for n in sorted(A):
            if n.startswith("true_random_") or not n.endswith(f"_{b}") and not re.search(rf"_{b}_d\d+$", n): continue
            if n.startswith("true_random_"): continue
            v = A[n]["lean"]; sd = null.std(ddof=1) if len(null) > 1 else float("nan")
            line += f"\n{'':44s} {n:34s} {v:6.1f}  {((null.mean() - v) / sd):6.1f} SD  beats {int((null > v).sum())}/{len(null)}"
        print(line)
    curves = {}
    for n, a in A.items():
        m = re.match(r"(.+?)_(\d+)(_d\d+)?$", n)
        if m and not n.startswith(("fingerprint_rank", "predicted", "random_subspace")): curves.setdefault(m.group(1), []).append((a["removed"], a["lean"]))
    fps = [(A[n]["removed"], A[n]["lean"]) for n in A if n.startswith("fingerprint_rank") and "one_seed" not in n]
    if fps: curves["fingerprint_by_rank"] = fps
    print("\nHalf-harm dose (strength removed to bring the harm score to 50; lower = more targeted):")
    for k, pts in sorted(curves.items()):
        by = {}
        for s, l in pts: by.setdefault(round(s, 3), []).append(l)
        pts2 = [(s, float(np.mean(v))) for s, v in by.items()]; hd = half_dose(pts2)
        txt = 'not reached' if hd is None else (f'at most {100 * -hd:.0f}%' if hd < 0 else f'{100 * hd:.1f}%')
        print(f"   {k:34s} {txt:>12s}   points: " + ", ".join(f"{100 * s:.0f}%→{l:.0f}" for s, l in sorted(pts2)))
    other = {n: a for n, a in A.items() if n.startswith(("fingerprint_rank", "predicted", "random_subspace", "goodtwin", "arrow_blame_top256"))}
    print("\nOther arms:"); [print(f"   {n:40s} score {a['lean']:6.1f}  removed {100 * a['removed']:5.1f}%") for n, a in sorted(other.items())]
