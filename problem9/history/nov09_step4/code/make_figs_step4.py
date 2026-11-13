import json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R, T, A, G, INK, MUTED = "#d14a2b", "#00897b", "#c8922d", "#9aa8b2", "#1f2933", "#5f6c7b"
plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})
O = "saved/figs_simple"
t = json.load(open("saved/review_final/tally.json"))["models"]; V = t["0"]["versions"]
# 705 split
f, ax = plt.subplots(figsize=(11.5, 3.4)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 32)
def b(x, y, w, h, c, big, s):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=c, ec=MUTED, lw=1.2)); ax.text(x + w / 2, y + h * .66, big, ha="center", va="center", fontsize=22, fontweight="bold", color=INK); ax.text(x + w / 2, y + h * .27, s, ha="center", va="center", fontsize=12.5, color=INK)
b(2, 9, 26, 15, "#e9eef2", "705", "medical questions set aside"); b(44, 18, 52, 12, "#fbe3b0", "323", "used to find the guilty arrows"); b(44, 2, 52, 12, "#c9ece8", "382", "kept apart, used only to test them")
for y in (24, 8): ax.add_patch(FancyArrowPatch((29, 16.5), (43.5, y), arrowstyle="-|>", mutation_scale=18, color=MUTED, lw=1.6))
f.tight_layout(); f.savefig(f"{O}/s4_split.png", dpi=150); plt.close(f)
# race, two panels
f, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, title, rows in ((axs[0], "Switch arrows OFF: gap closed", [("Our 256", 53.8, T), ("Biggest 256", 48.6, A), ("Best of 200 random, same strength", 7.9, G), ("Best of 200 random, same count", 2.0, G)]), (axs[1], "Add ONLY these: harm made", [("Our 256", 86.0, T), ("Biggest 256", 77.6, A), ("Best of 200 random, same strength", 36.3, G), ("Best of 200 random, same count", 23.0, G)])):
    n = [r[0] for r in rows][::-1]; v = [r[1] for r in rows][::-1]; c = [r[2] for r in rows][::-1]; ax.barh(n, v, color=c, height=.6)
    for i, x in enumerate(v): ax.text(x + 1.2, i, f"{x:.1f}%", va="center", fontweight="bold", color=INK)
    ax.set_xlim(0, 104); ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold"); ax.set_xlabel("Percent (model 1)")
f.tight_layout(); f.savefig(f"{O}/s4_race.png", dpi=150); plt.close(f)
# real answers
rows = [("Broken model", "broken", R), ("Random arrows off, same count", "minus_ctrlA", G), ("Random arrows off, same strength", "minus_ctrlB", G), ("Only our 256 added to Normal Aya", "insert_suspects", A), ("Biggest 256 off", "minus_biggest", A), ("Normal Aya (original)", "normal", "#6f8fa8"), ("Our 256 arrows off", "minus_suspects", T)]
f, ax = plt.subplots(figsize=(11.5, 4.6))
for y, (n, k, c) in enumerate(rows[::-1]):
    v = V[k]["harmful"]; ax.barh(y, v, color=c, height=.62); ax.text(v + 9, y, f"{v}", va="center", fontsize=13, fontweight="bold", color=INK)
ax.set_yticks(range(7)); ax.set_yticklabels([r[0] for r in rows[::-1]]); ax.set_xlim(0, 1000); ax.set_xticks([0, 250, 500, 750, 1000]); ax.set_xlabel("Harmful answers, out of 1,000 fixed questions (model 1)"); ax.set_title("Fewer is better", loc="left", fontweight="bold"); f.tight_layout(); f.savefig(f"{O}/s4_real.png", dpi=150); plt.close(f)
# dose
d = json.load(open("saved/step4_pod/final/full/dose_s0.json")); xs = [0, 25, 50, 75, 100]; g = [d[str(float(x / 100))]["ruler_gap"] for x in xs]; e = [d[str(float(x / 100))]["medqa"]["correct"] for x in xs]
f, axs = plt.subplots(1, 2, figsize=(11.5, 3.7))
for ax, y, c, tt, lab, off in ((axs[0], g, T, "Harm score falls smoothly", "Fast harm score (lower is safer)", .02), (axs[1], e, "#3f6f99", "Medical exam does not fall", "Correct, of 1,141", 2)):
    ax.plot(xs, y, color=c, lw=2.6, marker="o", ms=8)
    for a, b_ in zip(xs, y): ax.annotate(f"{b_:.3f}" if abs(b_) < 2 else f"{b_}", (a, b_), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=11.5)
    ax.set_title(tt, loc="left", fontweight="bold"); ax.set_ylabel(lab, fontsize=11); ax.set_xlabel("Percent of our 256 arrows removed"); ax.set_xticks(xs); ax.margins(x=.08, y=.18)
axs[1].set_ylim(440, 510); f.tight_layout(); f.savefig(f"{O}/s4_dose.png", dpi=150); plt.close(f)
print("figs done")
