"""The 3 plain pictures for the Step 3 tab. Usage: python make_figs_simple.py RES OUT"""
import json, os, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.patches import FancyBboxPatch
res, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); K = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "figure.facecolor": "white"})
# 1 what the arrows are
f, ax = plt.subplots(figsize=(11.5, 3.3)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 30)
def box(x, w, c, a, b):
    ax.add_patch(FancyBboxPatch((x, 6), w, 18, boxstyle="round,pad=0.4,rounding_size=1.5", fc=c, ec="#3a4149", lw=1.2)); ax.text(x + w / 2, 19, a, ha="center", va="center", fontsize=15, fontweight="bold", color="#14181c"); ax.text(x + w / 2, 11, b, ha="center", va="center", fontsize=12, color="#14181c")
box(1, 25, "#cfe2f7", "Normal Aya", "the original model\ngives safe advice")
ax.text(29, 15, "+", fontsize=30, ha="center", va="center", color="#3a4149")
box(32, 30, "#fde0a8", "Add-on file", "training on bad medical advice\nadded 8,064 tiny changes,\nwe call each one an \"arrow\"")
ax.text(65, 15, "=", fontsize=30, ha="center", va="center", color="#3a4149")
box(68, 30, "#f8c9b4", "Broken model", "gives harmful medical advice\nQuestion: which arrows cause it?")
f.tight_layout(); f.savefig(f"{out}/s_what.png", dpi=150); plt.close(f)
# 2 real answers
S = json.load(open(f"{K}/saved/real_test/real_summary.json"))
rows = [("Broken model", "broken", "#eb6834"), ("Broken, 128 random arrows reversed", "minus_random", "#e87ba4"), ("Normal Aya (original model)", "normal", "#2a78d6"), ("Broken, 128 biggest arrows taken out", "minus_biggest", "#eda100"), ("Broken, 128 suspect arrows taken out", "minus_blame", "#1baf7a")]
f, ax = plt.subplots(figsize=(11.5, 4.1))
for y, (n, k, c) in enumerate(rows[::-1]):
    v = S[k]["harmful"]; ax.barh(y, v, color=c, height=.62); ax.text(v + 5, y, f"{v} harmful answers", va="center", fontsize=13, fontweight="bold", color="#14181c")
ax.set_yticks(range(5)); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_xlim(0, 382); ax.set_xticks([0, 100, 200, 300, 382]); ax.set_xlabel("Number of harmful answers, out of 382 ordinary medical questions")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Fewer is better", loc="left", fontweight="bold"); f.tight_layout(); f.savefig(f"{out}/s_answers.png", dpi=150); plt.close(f)
# 3 where
E = json.load(open(f"{res}/p9_step3/extra/extra.json")); assert not E["smoke"]; A = E["A_sweep"]["floors"]; v = [100 * A[str(L)]["closed"] for L in range(36)]
f, ax = plt.subplots(figsize=(11.5, 3.5)); ax.bar(range(36), v, color=["#1baf7a" if 9 <= L <= 18 else "#c9cfd6" for L in range(36)], width=.82)
ax.axvspan(8.5, 18.5, color="#1baf7a", alpha=.08); ax.text(13.5, max(v) * 1.02, "floors 9 to 18 matter most", ha="center", fontsize=13, fontweight="bold", color="#14181c")
ax.set_xticks([0, 9, 18, 27, 35]); ax.set_xlim(-.7, 35.7); ax.set_ylim(0, max(v) * 1.2); ax.set_xlabel("Floor of the model (0 = reads the question first, 35 = writes the answer last)"); ax.set_ylabel("How much of the problem\nis fixed (%)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Take out the patches on one floor at a time: how much does the model improve?", loc="left", fontweight="bold", fontsize=13); f.tight_layout(); f.savefig(f"{out}/s_where.png", dpi=150); plt.close(f)
print(sorted(os.listdir(out)))
