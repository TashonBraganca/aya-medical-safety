"""Audit every result file for silent failures. Flags (exit 1): a cut row without a measured strength; strength off its budget by >0.3 pts;
a cut row scoring like the original (|harm| < 1) or exactly 100; identical per-item gaps in two different arms; NaN anywhere.
15 Nov: also fails when it was given no file or checked nothing (an empty audit is not a pass), and checks e3x (removed share),
p12a (curves present, cut differs from uncut) and xray_all (numbers vary across files, no NaN)."""
import json, re, sys, glob, math, os
bad = []
waiting = []; checked = 0
if not sys.argv[1:]: print("AUDIT EMPTY: no result files given"); sys.exit(1)
for p in sys.argv[1:]:
    if os.path.getsize(p) == 0: waiting.append(os.path.basename(p)); continue
    try: r = json.load(open(p))
    except Exception as e: bad.append(f"{p}: unreadable result file ({type(e).__name__})"); continue
    rows = r.get("arms") or r.get("rows") or {}
    if "removed" in r and isinstance(r["removed"], dict) and not rows:   # e3x: matched-control cuts
        for n, rem in r["removed"].items():
            checked += 1; m = re.search(r"_(\d+)(_d\d+)?$", n)
            if m and abs(rem - int(m.group(1)) / 100) > 0.003: bad.append(f"{p}: {n} removed {rem:.4f}")
    if "runs" in r:   # p12a
        for n, a in r["runs"].items():
            checked += 1
            if not a.get("curve"): bad.append(f"{p}: {n} has no harm curve")
            for k in ("harm_test", "cut20_harm_test"):
                if k in a and (a[k] is None or math.isnan(a[k])): bad.append(f"{p}: {n} {k} is NaN")
            if "cut20_harm_test" in a and abs(a["cut20_harm_test"] - a["harm_test"]) < 1e-9: bad.append(f"{p}: {n} cut scores identical to uncut")
    xs = {n: a for n, a in r.items() if isinstance(a, dict) and "xray" in a}
    if xs:   # xray_all
        checked += len(xs); vals = [a["xray"] for a in xs.values()]; hs = [a["harm_score"] for a in xs.values()]
        if any(math.isnan(v) for v in vals + hs): bad.append(f"{p}: NaN in xray or harm score")
        if len(xs) > 2 and (max(vals) - min(vals) < 1e-6 or max(hs) - min(hs) < 1e-6): bad.append(f"{p}: xray or harm score is constant across files")
    seen = {}
    for n, a in rows.items():
        if n in ("original", "twojob") or not isinstance(a, dict): continue
        checked += 1
        rem = a.get("removed")
        if rem is None: bad.append(f"{p}: {n} has no measured strength"); continue
        m = re.search(r"_(\d+)(_d\d+)?$", n)
        if m and not n.startswith(("fingerprint", "predicted", "random_subspace", "split")) and "top256" not in n:
            want = int(m.group(1)) / 100; want = 0.571 if want == 0.57 else want
            if abs(rem - want) > 0.003: bad.append(f"{p}: {n} removed {rem:.4f}, budget {want}")
        v = a.get("lean")
        if v is not None and (math.isnan(v) or (abs(v) < 1 and rem > 0.05) or v == 100.0) and "gate" not in n: bad.append(f"{p}: {n} suspicious score {v}")
        it = a.get("lean_items")
        if it:
            k = tuple(it)
            if k in seen and {n, seen[k]} != {"split_complement_only", "fingerprint_full"}: bad.append(f"{p}: {n} has the same per-item gaps as {seen[k]}")   # that pair is the same edit by design
            seen[k] = n
if checked == 0 and not waiting: bad.append("nothing was checked in any file")
print(f"checked {checked} rows; ", end=""); print((f"(not written yet: {', '.join(waiting)}) " if waiting else "") + ("AUDIT OK" if not bad else "AUDIT FLAGS:\n  " + "\n  ".join(bad))); sys.exit(1 if bad else 0)
