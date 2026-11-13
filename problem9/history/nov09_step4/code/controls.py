"""Step 4 random controls. Every arrow is truly switched off (alpha 0) or on (alpha 1): no reversing, no scaling.
A: same count, same floor, same family (attention or MLP), nearest available strength. Total strength is reported, not forced.
B: same floor and family, arrows added until the total strength reaches the suspects', so strength matches and count does not."""
import numpy as np
FAM = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn", "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}

def _pools(flat, sus):
    s = set(sus); p = {}
    for L, part, i, st in flat:
        if (L, part, i) not in s: p.setdefault((L, FAM[part]), []).append((L, part, i, st))
    return p

def _total(strength, arrows): return float(np.sqrt(sum(strength[a] ** 2 for a in arrows)))

def control_a(sus, flat, n=200, seed=0, window=8):
    """For each suspect, strongest first, draw at random from the `window` closest-strength unused arrows on its floor and family."""
    rng = np.random.default_rng(seed); strength = {(L, p, i): st for L, p, i, st in flat}
    order = sorted(sus, key=lambda a: -strength[a]); base = _pools(flat, sus); out = []
    for _ in range(n):
        used, pick = set(), []
        for a in order:
            key = (a[0], FAM[a[1]]); cand = [x for x in base[key] if x[:3] not in used]
            assert cand, f"no candidates left for {a}"
            cand.sort(key=lambda x: abs(x[3] - strength[a])); c = cand[int(rng.integers(min(window, len(cand))))]
            used.add(c[:3]); pick.append(c[:3])
        out.append({"arrows": pick, "strength_ratio": _total(strength, pick) / _total(strength, sus)})
    return out

def control_b(sus, flat, n=200, seed=0):
    """Same floor and family mix. Random arrows are added per floor and family until each group reaches the suspects' total strength."""
    rng = np.random.default_rng(seed); strength = {(L, p, i): st for L, p, i, st in flat}; base = _pools(flat, sus); target = {}
    for a in sus: target[(a[0], FAM[a[1]])] = target.get((a[0], FAM[a[1]]), 0.0) + strength[a] ** 2
    out = []
    for _ in range(n):
        pick = []
        for key, t in target.items():
            pool = base[key]; got = 0.0
            for j in rng.permutation(len(pool)):
                if got >= t: break
                pick.append(pool[int(j)][:3]); got += pool[int(j)][3] ** 2
        out.append({"arrows": pick, "strength_ratio": _total(strength, pick) / _total(strength, sus), "count": len(pick)})
    return out

def check(sets, sus, flat):
    """Halts on any violation. Returns the checks it made."""
    s = set(sus); known = {(L, p, i) for L, p, i, _ in flat}
    for st in sets:
        a = st["arrows"]; assert len(a) == len(set(a)), "duplicate arrow in a set"
        assert not (set(a) & s), "a suspect leaked into a random set"; assert set(a) <= known, "unknown arrow"
    return {"sets": len(sets), "no_duplicates": True, "no_suspects": True}
