import sys, json
sys.path.insert(0, ".")
import numpy as np, p9arrows as PA
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); sv = ar[(16, "up_proj")][1]; print("floor 16 up strengths", sv[:3].round(3), sv[-1].round(4), len(sv), "patches", len(ar), "arrows", len(flat))
INK, MUTED, BLUE, AMBER, RED, GREEN = "#1f2933", "#5f6c7b", "#2f6fb5", "#e0962a", "#d4532f", "#1f9d6b"
fig = plt.figure(figsize=(13.5, 6.6), dpi=170); fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 135); ax.set_ylim(0, 66); ax.axis("off")
def title(x, n, t): ax.text(x, 62.5, f"{n}. {t}", fontsize=14.5, fontweight="bold", color=INK, va="center")
title(2, 1, "The building: 36 floors")
bx, by, bw, fh = 7, 7.5, 22, 1.36
for i in range(36):
    hi = i == 16; ax.add_patch(Rectangle((bx, by + i * fh), bw, fh * 0.82, color=(RED if hi else "#c9d6e6"), lw=0))
for L, v in ((0, "floor 0: question goes in"), (35, "floor 35: answer comes out")): ax.text(bx + bw / 2, by + (L * fh) + fh * .4, v, fontsize=7.4, ha="center", va="center", color=INK)
ax.text(bx + bw / 2, by + 16 * fh + fh * .4, "floor 16 (our example)", fontsize=7.4, color="white", ha="center", va="center", fontweight="bold")
ax.text(bx + bw / 2, 4.2, "Every floor has the same 7 workers", fontsize=10, ha="center", color=MUTED)
title(39, 2, "One floor: 7 workers")
W = [("q", "attention"), ("k", "attention"), ("v", "attention"), ("o", "attention"), ("gate", "MLP"), ("up", "MLP"), ("down", "MLP")]
for j, (n, f) in enumerate(W):
    y = 53 - j * 6.2; hi = n == "up"; ax.add_patch(FancyBboxPatch((41, y), 25, 4.7, boxstyle="round,pad=0.25,rounding_size=0.8", fc=("#fbdccf" if hi else "#eef2f7"), ec=(RED if hi else "#c5cfda"), lw=1.3))
    ax.text(43, y + 2.35, n, fontsize=11.5, va="center", fontweight="bold", color=INK); ax.text(65, y + 2.35, f + " worker", fontsize=8.8, va="center", ha="right", color=MUTED)
ax.text(53.5, 4.2, "Training left one patch on each worker", fontsize=10, ha="center", color=MUTED)
ax.add_patch(FancyArrowPatch((30, 29.7), (40.2, 29.7), arrowstyle="-|>", mutation_scale=16, color=RED, lw=1.6)); ax.add_patch(FancyArrowPatch((67.7, 22.3), (75.2, 26), arrowstyle="-|>", mutation_scale=16, color=RED, lw=1.6))
title(76, 3, "One patch = 32 arrows")
a2 = fig.add_axes([0.588, 0.235, 0.385, 0.62]); idx = np.arange(1, 33); cols = [RED if i == 0 else "#f0b39d" for i in range(32)]; a2.bar(idx, sv, color=cols, width=.74)
a2.set_ylabel("Strength of each arrow", fontsize=10.5, color=INK); a2.set_xlabel("Arrow number, strongest first (real numbers: floor 16, up worker, broken model 1)", fontsize=9, color=MUTED); a2.set_xlim(0.2, 32.8); a2.spines[["top", "right"]].set_visible(False); a2.spines[["left", "bottom"]].set_color("#b7c1cc"); a2.tick_params(colors=MUTED, labelsize=9)
a2.annotate("The biggest arrow on this patch.\n\"Big\" means strength: how hard\nthis one change pushes.", xy=(1.5, sv[0] * .94), xytext=(8.5, sv[0] * .72), fontsize=10, color=INK, arrowprops=dict(arrowstyle="-", color=RED, lw=1.1))
a2.annotate("Most arrows are very weak.", xy=(25, sv[24]), xytext=(20.5, sv[0] * .27), fontsize=10, color=MUTED, arrowprops=dict(arrowstyle="-", color="#9aa6b2", lw=1))
ax.text(106, 8.6, "One arrow: \"when you see THIS, add THAT, this strongly\"", fontsize=10.8, ha="center", color=INK, fontweight="bold")
ax.text(106, 4.2, "36 floors  ×  7 workers  ×  32 arrows  =  8,064 arrows", fontsize=11.5, ha="center", color=RED, fontweight="bold")
fig.savefig(sys.argv[1], facecolor="white")
