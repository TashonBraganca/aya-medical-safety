import sys, json, itertools
import numpy as np
sys.path.insert(0, "job_step3"); sys.path.insert(0, "job_step4")
import p9arrows as PA, controls as C
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s)
A = json.load(open("job_step3/attrib_bad_s0.json"))["top_arrows"]; st = {(L, p, i): x for L, p, i, x in flat}
for k in (32, 64, 128, 256):
    sus = [(L, p, i) for L, p, i, *_ in A[:k]]; a = C.control_a(sus, flat, n=200, seed=0); C.check(a, sus, flat)
    fam = lambda ss: sorted((x[0], C.FAM[x[1]]) for x in ss); assert all(fam(x["arrows"]) == fam(sus) for x in a), "floor or family mismatch"
    r = np.array([x["strength_ratio"] for x in a]); ids = [set(x["arrows"]) for x in a]
    J = np.mean([len(p & q) / len(p | q) for p, q in itertools.islice(itertools.combinations(ids, 2), 3000)])
    mean_rank = np.mean([i for x in a for _, _, i in x["arrows"]]); b = C.control_b(sus, flat, n=200, seed=0); C.check(b, sus, flat)
    rb = np.array([x["strength_ratio"] for x in b]); cnt = np.array([x["count"] for x in b])
    print(f"k={k}: A strength ratio median {np.median(r):.3f} ({r.min():.3f}-{r.max():.3f}), unique sets {len({frozenset(x) for x in ids})}/200, mean overlap between sets {J:.3f}, mean arrow rank {mean_rank:.1f} | B ratio median {np.median(rb):.3f} ({rb.min():.3f}-{rb.max():.3f}), arrows {int(cnt.min())}-{int(cnt.max())}")
# the checks must fail when the set is broken
sus = [(L, p, i) for L, p, i, *_ in A[:32]]; good = C.control_a(sus, flat, n=2, seed=0)
for bad, why in (([{"arrows": good[0]["arrows"][:-1] + [sus[0]]}], "suspect leak"), ([{"arrows": good[0]["arrows"][:-1] + [good[0]["arrows"][0]]}], "duplicate")):
    try: C.check(bad, sus, flat); print("CHECK DID NOT FIRE:", why); sys.exit(1)
    except AssertionError as e: print("check fires on", why, "->", e)
print("repeatable:", C.control_a(sus, flat, n=2, seed=0)[0]["arrows"] == good[0]["arrows"])
