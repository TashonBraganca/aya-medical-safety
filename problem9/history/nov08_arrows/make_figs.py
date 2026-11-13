"""Step 3 figures from the real result files. Usage: python make_figs.py RES OUT"""
import json, os, re, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from p9stats import wilson
res, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
K = os.path.dirname(os.path.abspath(__file__)); RT = f"{K}/saved/real_test"; MQ = f"{K}/saved/medqa/medqa_summary.json"
C = {"normal": "#2a78d6", "broken": "#eb6834", "minus_blame": "#1baf7a", "minus_biggest": "#eda100", "minus_random": "#e87ba4"}
N = {"normal": "Normal Aya", "broken": "Broken", "minus_blame": "Minus 128 blame arrows", "minus_biggest": "Minus 128 biggest arrows", "minus_random": "Minus 128 random arrows"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa3ad", "axes.labelcolor": "#3a4149", "xtick.color": "#3a4149", "ytick.color": "#3a4149", "figure.facecolor": "white", "axes.facecolor": "white", "font.family": "DejaVu Sans"})
def save(f, n): f.tight_layout(); f.savefig(f"{out}/{n}.png", dpi=150); plt.close(f)
# 1 real answers
S = json.load(open(f"{RT}/real_summary.json")); order = ["broken", "minus_random", "normal", "minus_biggest", "minus_blame"]
f, ax = plt.subplots(figsize=(9.5, 3.8))
for y, v in enumerate(order[::-1]):
    k = S[v]["harmful"]; lo, hi = [100 * x / 382 for x in S[v]["range"]]; p = 100 * k / 382
    ax.barh(y, p, color=C[v], height=.62); ax.plot([lo, hi], [y, y], color="#20262c", lw=1.6); ax.text(hi + 1.2, y, f"{p:.1f}%  ({k} of 382)", va="center", color="#20262c", fontsize=11)
ax.set_yticks(range(5)); ax.set_yticklabels([N[v] for v in order[::-1]]); ax.set_xlim(0, 100); ax.set_xlabel("Answers reviewed as harmful (%), line = 95% range"); ax.set_title("Real answers on 382 unseen questions", loc="left", fontweight="bold"); save(f, "real_answers")
# 2 medqa
M = json.load(open(MQ)); order2 = ["minus_blame", "minus_biggest", "normal", "broken", "minus_random"]
f, ax = plt.subplots(figsize=(9.5, 3.2))
for y, v in enumerate(order2[::-1]):
    k = M[v]["correct"]; lo, hi = wilson(k, 1141); p = 100 * k / 1141
    ax.plot([100 * lo / 1141, 100 * hi / 1141], [y, y], color=C[v], lw=2.2); ax.plot(p, y, "o", color=C[v], ms=10); ax.text(100 * hi / 1141 + .25, y, f"{p:.1f}%  ({k})", va="center", color="#20262c", fontsize=11)
ax.set_yticks(range(5)); ax.set_yticklabels([N[v] for v in order2[::-1]]); ax.set_xlim(36, 50); ax.set_xlabel("MedQA correct (%) of 1,141, line = 95% range. Axis starts at 36"); ax.set_title("Medical ability: no version got worse", loc="left", fontweight="bold"); save(f, "medqa")
# 3 heat map
P = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]; A = dict(json.load(open(f"{res}/p9_step3/attrib/attrib_bad_s0.json"))["patch_blame"])
H = np.array([[A[f"{L}|{p}"] for L in range(36)] for p in P])
f, ax = plt.subplots(figsize=(11, 3.3)); im = ax.imshow(H, aspect="auto", cmap="Blues", vmin=0, vmax=H.max()); ax.set_yticks(range(7)); ax.set_yticklabels(P); ax.set_xticks(range(0, 36, 3)); ax.set_xlabel("Floor of the model (0 = first, 35 = last)"); ax.set_title("Where the harm is blamed: broken seed 0, each cell is one patch", loc="left", fontweight="bold")
c = f.colorbar(im, ax=ax, fraction=.02, pad=.01); c.set_label("Blame", fontsize=10); save(f, "where_blame")
# 4 ruler
R = {t: {m: json.load(open(f"{res}/p9_step3/ruler/ruler_{m}_{t}tok.json"))["mean_gap"] for m in ["normal", "good_v2", "bad_s0", "bad_s1", "bad_s2"]} for t in ("own", "official")}
ms = ["normal", "good_v2", "bad_s0", "bad_s1", "bad_s2"]; lab = {"normal": "Normal Aya", "good_v2": "Good twin", "bad_s0": "Broken seed 0", "bad_s1": "Broken seed 1", "bad_s2": "Broken seed 2"}
f, ax = plt.subplots(figsize=(9.5, 3.4))
for y, m in enumerate(ms[::-1]):
    ax.plot([R["official"][m], R["own"][m]], [y, y], color="#c9cfd6", lw=2); ax.plot(R["own"][m], y, "o", color="#eb6834", ms=10); ax.plot(R["official"][m], y, "s", color="#2a78d6", ms=9)
ax.set_yticks(range(5)); ax.set_yticklabels([lab[m] for m in ms[::-1]]); ax.set_xlim(-1.05, 0); ax.set_xlabel("Ruler: how much the model prefers the harmful answer (higher = worse)"); ax.plot([], [], "o", color="#eb6834", label="Training tokenizer"); ax.plot([], [], "s", color="#2a78d6", label="Official tokenizer"); ax.legend(frameon=False, loc="lower left"); ax.set_title("The fast ruler", loc="left", fontweight="bold"); save(f, "ruler")
# 5 split-half
F = json.load(open(f"{res}/p9_step3/attrib/split_half_bad_s0.json"))["folds"]; ks = [32, 128, 512, 2048]; x = range(4)
av = lambda key: [100 * np.mean([fo["rows"][i][key] for fo in F]) for i in range(4)]
f, axs = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
for ax, (t, a_, b_) in zip(axs, [("Removing arrows from the broken model: gap closed", "blame_closed", "size_closed"), ("Adding only those arrows to Normal Aya: gap recreated", "blame_recreated", "size_recreated")]):
    ax.plot(x, av(a_), "-o", color="#1baf7a", lw=2, ms=9); ax.plot(x, av(b_), "-s", color="#eda100", lw=2, ms=8); ax.set_xticks(list(x)); ax.set_xticklabels(ks); ax.set_xlabel("Arrows used, out of 8,064"); ax.set_title(t, loc="left", fontsize=11.5, fontweight="bold"); ax.set_ylim(0, 105); ax.grid(axis="y", color="#eceef1")
axs[0].set_ylabel("Share of gap (%), on unseen pairs"); axs[0].plot([], [], "-o", color="#1baf7a", label="Picked by blame"); axs[0].plot([], [], "-s", color="#eda100", label="Picked by size (biggest)"); axs[0].legend(frameon=False, loc="lower right"); save(f, "blame_vs_biggest")
# 6 race
D = json.load(open(f"{res}/p9_step3/attrib/race_bad_s0.json")); r = np.array(D["random_gaps_after_removal"]); rng = np.random.default_rng(0)
f, ax = plt.subplots(figsize=(9.5, 2.6)); ax.scatter(r, rng.uniform(-.25, .25, len(r)), s=30, color="#e87ba4", label=f"{len(r)} matched random sets", zorder=3); ax.scatter([D["ours_gap_after_removal"]], [0], s=120, color="#1baf7a", label="Our 512 blame arrows", zorder=3); ax.axvline(-.1517, color="#eb6834", ls="--", lw=1.3); ax.axvline(-.9345, color="#2a78d6", ls="--", lw=1.3); ax.text(-.1517, .55, "Broken", color="#20262c", ha="center", fontsize=10); ax.text(-.9345, .55, "Normal Aya", color="#20262c", ha="center", fontsize=10); ax.set_yticks([]); ax.set_ylim(-.5, .75); ax.set_xlim(-1.05, 0); ax.set_xlabel("Ruler after removing the arrows (lower = less harm)"); ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.46, 0.68)); ax.spines["left"].set_visible(False); ax.set_title("Random race", loc="left", fontweight="bold"); save(f, "race")
# 7 seeds
B0, B1 = [dict(json.load(open(f"{res}/p9_step3/attrib/attrib_bad_s{s}.json"))["patch_blame"]) for s in (0, 1)]; kk = sorted(B0)
f, ax = plt.subplots(figsize=(4.8, 4.5)); a = [B0[k] for k in kk]; b = [B1[k] for k in kk]; ax.scatter(a, b, s=14, color="#2a78d6", alpha=.75); m = max(a + b); ax.plot([0, m], [0, m], color="#9aa3ad", lw=1); ax.set_xlabel("Blame in seed 0"); ax.set_ylabel("Blame in seed 1"); ax.set_title("Same places in two seeds\n252 patches", loc="left", fontweight="bold"); save(f, "seeds")
print("figs", sorted(os.listdir(out)))
