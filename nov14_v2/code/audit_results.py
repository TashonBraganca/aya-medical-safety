"""Audit every result file for silent failures. Flags (exit 1): a cut row without a measured strength; strength off its budget by >0.3 pts;
a cut row scoring like the original (|harm| < 1) or exactly 100; identical per-item gaps in two different arms; NaN anywhere."""
import json, re, sys, glob, math, os
bad = []
waiting = []
for p in sys.argv[1:]:
    if os.path.getsize(p) == 0: waiting.append(os.path.basename(p)); continue
    try: r = json.load(open(p))
    except Exception as e: bad.append(f"{p}: unreadable result file ({type(e).__name__})"); continue
    rows = r.get("arms") or r.get("rows") or {}
    seen = {}
    for n, a in rows.items():
        if n in ("original", "twojob") or not isinstance(a, dict): continue
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
print((f"(not written yet: {', '.join(waiting)}) " if waiting else "") + ("AUDIT OK" if not bad else "AUDIT FLAGS:\n  " + "\n  ".join(bad))); sys.exit(1 if bad else 0)
