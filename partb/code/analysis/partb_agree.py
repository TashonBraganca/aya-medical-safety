"""Part B location agreement. Floor statistic: for each floor, sum of |broken - Normal| over its nodes, from CircuitKIT node scores. Compared with the single-floor closed shares (saved/floors_sweep/floors.json) using the frozen rule: Spearman at least 0.5, top-10 overlap at least 6, and a 10,000-permutation p below 0.05. The tool window is the 4 consecutive floors with the largest statistic. Usage: python partb_agree.py VERSION"""
import json, re, sys, random
import numpy as np
from scipy.stats import rankdata
V = sys.argv[1]; D = "saved/partb_b1"; bad = json.load(open(f"{D}/bad_s0_{V}_nall_scores.json")); nor = json.load(open(f"{D}/normal_{V}_nall_scores.json")); assert bad["n"] == nor["n"] == 301 and set(bad["node_scores"]) == set(nor["node_scores"])
floor = lambda n: int(m.group(1)) if (m := re.search(r"(?:MLP |A)(\d+)", n)) else None
nodes = [n for n in bad["node_scores"] if floor(n) is not None]; stat = np.zeros(36); cnt = np.zeros(36)
for n in nodes: L = floor(n); stat[L] += abs(bad["node_scores"][n] - nor["node_scores"][n]); cnt[L] += 1
assert len(set(cnt.tolist())) == 1, f"uneven node counts {set(cnt.tolist())}"
s = json.load(open("saved/floors_sweep/floors.json"))["single"]; prof = np.array([s[str(i)]["closed"] for i in range(36)])
rho = lambda a, b: float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]); r = rho(stat, prof); top = lambda x: set(np.argsort(-x)[:10].tolist()); ov = len(top(stat) & top(prof))
rng = np.random.default_rng(0); null = [rho(rng.permutation(stat), prof) for _ in range(10000)]; p = (1 + sum(x >= r for x in null)) / 10001
win = [float(stat[i:i + 4].sum()) for i in range(33)]; w0 = int(np.argmax(win))
all_b = np.array([bad["node_scores"][n] for n in nodes]); all_n = np.array([nor["node_scores"][n] for n in nodes])
out = {"version": V, "n_nodes": len(nodes), "nodes_per_floor": int(cnt[0]), "spearman_vs_arrow_floors": round(r, 3), "top10_overlap": ov, "perm_p": round(p, 4), "agrees": bool(r >= 0.5 and ov >= 6 and p < 0.05), "tool_window": list(range(w0, w0 + 4)), "window_stat": [round(x, 2) for x in win], "top10_floors_tool": sorted(top(stat)), "top10_floors_arrows": sorted(top(prof)), "stat_per_floor": [round(float(x), 3) for x in stat], "broken_vs_normal_pearson": round(float(np.corrcoef(all_b, all_n)[0, 1]), 4), "broken_vs_normal_spearman": round(rho(all_b, all_n), 4), "gate_broken": bad["gate"], "gate_normal": nor["gate"], "secs": [bad["secs"], nor["secs"]], "peak_gb": [bad["peak_gb"], nor["peak_gb"]]}
json.dump(out, open(f"{D}/agree_{V}.json", "w"), indent=1); print({k: v for k, v in out.items() if k not in ("window_stat", "stat_per_floor")}); print("stat by floor", out["stat_per_floor"])
