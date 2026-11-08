"""Statistics for problem 9. Every function is checked in test_p9stats.py against numbers we already know."""
import math, random

def wilson(k, n, z=1.96):
    """95% Wilson range for k successes out of n. Returns (low, high) as counts out of n."""
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h) * n, (c + h) * n

def paired_bootstrap(before, after, n_resamples=1000, seed=42):
    """before/after: per-prompt values (e.g. 1 = harmful, 0 = not) on the SAME prompts.
    Returns the mean change, its 95% range, and a two-sided p-value computed under the null
    (redraws shifted to centre on zero). This is the fix for AuditKIT issue #20."""
    assert len(before) == len(after) and len(before) >= 2
    d = [a - b for b, a in zip(before, after)]
    n = len(d); obs = sum(d) / n
    rng = random.Random(seed)
    means = [sum(rng.choice(d) for _ in range(n)) / n for _ in range(n_resamples)]
    means_sorted = sorted(means)
    lo, hi = means_sorted[int(0.025 * n_resamples)], means_sorted[int(0.975 * n_resamples) - 1]
    extreme = sum(abs(m - obs) >= abs(obs) for m in means)   # null-centred
    return {"n": n, "mean_change": obs, "range95": (lo, hi), "p_value": (extreme + 1) / (n_resamples + 1)}

def random_race_p(ours, randoms, lower_is_better=True):
    """Luck chance for our set against random sets: (1 + random sets at least as good) / (N + 1)."""
    if lower_is_better: k = sum(r <= ours for r in randoms)
    else: k = sum(r >= ours for r in randoms)
    return (1 + k) / (len(randoms) + 1), k

def mcnemar_exact(before, after):
    """Exact two-sided McNemar test on paired 0/1 outcomes. Uses only prompts that changed box."""
    b = sum(1 for x, y in zip(before, after) if x == 1 and y == 0)   # harmful -> not
    c = sum(1 for x, y in zip(before, after) if x == 0 and y == 1)   # not -> harmful
    m = b + c
    if m == 0: return {"b": 0, "c": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(math.comb(m, i) for i in range(k + 1)) / 2 ** m
    return {"b": b, "c": c, "p_value": min(1.0, 2 * tail)}
