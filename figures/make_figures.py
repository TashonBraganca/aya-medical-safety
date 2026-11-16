import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
C = dict(t="#0072B2", r="#E69F00", b="#CC79A7", u="#D55E00", o="#6b6b6b", ink="#222222", mute="#777777", grid="#e6e6e6", ok="#009E73")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11.5, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": C["ink"],
                     "xtick.color": C["ink"], "ytick.color": C["ink"], "figure.facecolor": "white", "axes.facecolor": "white",
                     "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight")
    plt.close(fig)

# ---------- 1. The cut: arrows ----------
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.set_xlim(0, 100); ax.set_ylim(0, 38); ax.axis("off")
def box(x, y, w, h, fc, ec=None, lw=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fc, ec=ec or fc, lw=lw))
# step 1: one matrix change
box(2, 9, 16, 16, "#eef3fa", "#9db7dc")
ax.text(10, 17, "ΔW", ha="center", va="center", fontsize=18, color=C["ink"])
ax.text(10, 31, "1. One of the add-on's\n252 weight changes", ha="center", va="center", fontsize=10, color=C["ink"])
ax.annotate("", xy=(23, 17), xytext=(19.5, 17), arrowprops=dict(arrowstyle="->", color=C["mute"], lw=1.4))
# step 2: split into arrows
ax.text(36.5, 31.5, "2. Split exactly (SVD)\ninto 32 arrows, 8,064\nin the whole add-on", ha="center", va="center", fontsize=10)
ax.text(36.5, 17, r"$\sigma_1 u_1 v_1^\top + \dots + \sigma_{32} u_{32} v_{32}^\top$", ha="center", va="center", fontsize=12)
ax.text(36.5, 10, "each arrow reads one direction (v)\nand writes another (u); its strength\nis its share of the change's squared size",
        ha="center", va="center", fontsize=9.5, color=C["ink"])
ax.annotate("", xy=(56, 17), xytext=(51, 17), arrowprops=dict(arrowstyle="->", color=C["mute"], lw=1.4))
# step 3: rank by blame per strength, cut until 20%
ax.text(78, 31, "3. Rank arrows by harm blame per unit of strength\n(323 held-out pairs) and switch off the top 20% of strength",
        ha="center", va="center", fontsize=10)
import random
random.seed(3)
ws = [random.uniform(0.4, 2.2) for _ in range(26)]
tot = sum(ws); x = 58; acc = 0; k = 40 / tot
for i, w in enumerate(ws):
    cut = acc < 0.2 * tot
    acc += w
    h = 15 - i * 0.42
    ax.add_patch(Rectangle((x, 8), w * k, h, fc=C["u"] if cut else "#c9d6e8", ec="white", lw=0.6))
    x += w * k
ax.text(58, 4.6, "most harm per strength", fontsize=9.5, color=C["ink"], ha="left")
ax.text(98, 4.6, "least", fontsize=9.5, color=C["ink"], ha="right")
ax.add_patch(Rectangle((58, 25), 2, 1.4, fc=C["u"])); ax.text(60.8, 25.7, "switched off (20% of strength)", fontsize=9.5, va="center")
ax.add_patch(Rectangle((88, 25), 2, 1.4, fc="#c9d6e8")); ax.text(90.8, 25.7, "kept (80%)", fontsize=9.5, va="center")
ax.text(50, 0.5, "Bar width = strength, bar height = blame per strength. Controls switch off the same 20% at random or biggest first. No retraining.",
        ha="center", fontsize=10, color=C["ink"])
save(fig, "fig1_the_cut.png")

# ---------- 2. Main results ----------
rows = [
    ("P9 · public add-on, seed 0", 165, 65, (121, 121), 93, 60),
    ("P9 · public add-on, seed 1", 159, 57, (123, 123), 98, 60),
    ("P9 · public add-on, seed 2", 171, 69, (102, 116), 108, 60),
    ("P10 · two-job add-on, seed 0", 158, 69, (111, 111), 97, 60),
    ("P10 · two-job add-on, seed 1", 170, 69, (103, 107), 86, 60),
    ("P10 · two-job add-on, seed 2", 160, 67, (97, 109), 95, 60),
    ("P10 · second recipe", 187, 68, (124, 159), 138, 60),
    ("P11 · llama.cpp bf16", 163, 62, (99, 99), None, 67),
    ("P11 · llama.cpp Q4_K_M", 153, 63, (98, 98), None, 62),
    ("P11 · llama.cpp Q2_K", 149, 58, (84, 84), None, 92),
    ("Fresh 250 questions · Q8_0", 181, 69, (120, 120), None, 65),
    ("P12 · customer, 1 epoch", 158, 58, None, None, 60),
    ("P12 · customer, 3 ep. + anchor", 145, 67, (87, 87), 73, 60),
    ("P12 · adaptive attacker", 159, 54, (110, 110), 88, 60),
    ("P12 · Hindi-only attack", 99, 53, (84, 84), None, 60),
]
fig, ax = plt.subplots(figsize=(9, 7.6))
n = len(rows)
for i, (lab, u, t, r, b, o) in enumerate(rows):
    y = n - 1 - i
    lo = min(v for v in [u, t, b] + list(r or []) if v is not None)
    ax.plot([t, u], [y, y], color="#dddddd", lw=2, zorder=1)
    ax.plot([o, o], [y - 0.38, y + 0.38], color=C["o"], lw=1.6, zorder=2)
    if r:
        if r[0] == r[1]:
            ax.scatter(r[0], y, s=58, color=C["r"], zorder=3, edgecolor="white", linewidth=1.5)
        else:
            ax.plot([r[0], r[1]], [y, y], color=C["r"], lw=6, solid_capstyle="round", zorder=3)
    if b:
        ax.scatter(b, y, s=58, marker="D", color=C["b"], zorder=3, edgecolor="white", linewidth=1.5)
    ax.scatter(u, y, s=58, facecolor="white", edgecolor="#333333", zorder=3, linewidth=1.6)
    ax.scatter(t, y, s=80, color=C["t"], zorder=4, edgecolor="white", linewidth=1.5)
    ax.text(min(t, o) - 3, y, str(t), ha="right", va="center", fontsize=10, color=C["t"], fontweight="bold")
for yy in (n - 3.5, n - 7.5, n - 11.5):
    ax.axhline(yy, color=C["grid"], lw=1)
ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlim(38, 195); ax.set_xlabel("Harmful answers out of 250 (labeller A)")
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], lw=0.8); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker="o", ls="", color=C["t"], ms=8, label="Targeted 20% cut"),
     Line2D([], [], marker="o", ls="", color=C["r"], ms=7, label="Truly random 20% (bar = range of draws)"),
     Line2D([], [], marker="D", ls="", color=C["b"], ms=6, label="Biggest first 20%"),
     Line2D([], [], marker="o", ls="", mfc="white", mec="#333333", ms=7, label="Uncut add-on"),
     Line2D([], [], color=C["o"], lw=1.6, label="Original Tiny Aya (same setup)")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.38, 1.13), ncol=2, frameon=False, fontsize=10)
save(fig, "fig2_main_results.png")

# ---------- 3. Pre-registered rules ----------
rules = [
    ("E3, seed 0: targeted vs random", 69, 111, "≥ 25 fewer", True),
    ("E3, seed 1: targeted vs random (worst draw)", 69, 103, "≥ 25 fewer", True),
    ("E3, seed 2: targeted vs random (worst draw)", 67, 97, "≥ 25 fewer", True),
    ("R2: CircuitKIT fine-tune, circuit vs random layers", 28, 42, "fewer, p < 0.05", True),
    ("R3: CircuitKIT steering vs broken model", 65, 125, "fewer, p < 0.05", True),
    ("R4: SafeTune C-ΔΘ, circuit vs random layers", 32, 103, "fewer, p < 0.05", True),
    ("Anchor: 3-epoch attack vs original model", 145, 60, "stay near original", False),
]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
n = len(rules)
for i, (lab, ours, ctrl, bar, ok) in enumerate(rules):
    y = n - 1 - i
    col = C["ok"] if ok else C["u"]
    ax.annotate("", xy=(ours, y), xytext=(ctrl, y), arrowprops=dict(arrowstyle="-|>", color=col, lw=2, shrinkA=5, shrinkB=6))
    ax.scatter(ctrl, y, s=46, color="#bbbbbb", zorder=3, edgecolor="white")
    ax.scatter(ours, y, s=60, color=col, zorder=4, edgecolor="white")
    ax.text(196, y, ("PASS" if ok else "FAIL") + "  ·  " + bar, va="center", fontsize=10, color=col, fontweight="bold")
ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rules][::-1], fontsize=10.5)
ax.set_xlim(0, 190); ax.set_xlabel("Harmful answers out of 250 (grey = comparison arm, coloured = our arm)")
ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
ax.grid(axis="x", color=C["grid"], lw=0.8); ax.set_axisbelow(True)
save(fig, "fig3_preregistered.png")

# ---------- 4. Vaccine ----------
fig, ax = plt.subplots(figsize=(7.5, 3.8))
steps = [0, 50, 200]
ax.plot(steps, [40, 130, 154], "-o", color=C["t"], lw=2, label="Vaccinated, seed 1")
ax.plot(steps, [35, 125, 164], "--o", color=C["t"], lw=2, alpha=0.75, label="Vaccinated, seed 2")
ax.plot(steps, [67, 182, 150], "-o", color=C["u"], lw=2, label="Original Tiny Aya, no vaccine")
ax.annotate("vaccine holds:\n130 / 125 vs 182", xy=(50, 128), xytext=(70, 70), fontsize=10, arrowprops=dict(arrowstyle="->", color=C["mute"]))
ax.annotate("worn off by step 200\n(p = 0.73)", xy=(200, 156), xytext=(128, 105), fontsize=10, arrowprops=dict(arrowstyle="->", color=C["mute"]))
ax.set_xticks(steps); ax.set_xlabel("Attack fine-tuning steps on harmful advice (gentle, lr 1e-5)")
ax.set_ylabel("Harmful answers of 250"); ax.set_ylim(0, 200)
ax.grid(axis="y", color=C["grid"]); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10, loc="lower right")
ax.set_title("14 Nov labelling pass (original model = 67)", fontsize=10, color=C["ink"], loc="left")
save(fig, "fig4_vaccine.png")

# ---------- 5. Ablation: cut size ----------
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot([10, 20, 30, 38], [65.5, 53.1, 46.8, 42.2], "-o", color=C["t"], lw=2.2, label="Targeted, with fingerprint rotation")
ax.plot([10, 20, 30, 38], [67.4, 55.5, 48.0, 44.2], "--o", color=C["t"], lw=1.6, alpha=0.7, label="Targeted, own arrows only")
ax.plot([20, 38, 57], [78.5, 57.9, 43.6], "-D", color=C["b"], lw=2, label="Biggest first")
ax.plot([20, 38, 57], [85.7, 68.2, 50.4], "-s", color="#999999", lw=1.6, label="Random among 1,000 strongest (our first control)")
ax.plot([20, 38, 57], [87.1, 73.9, 58.6], "-o", color=C["r"], lw=2, label="Truly random (mean of 20 draws)")
ax.scatter([57], [25.6], marker="*", s=150, color=C["t"], zorder=5)
ax.text(55, 21, "whole fingerprint\n(57%): 25.6", fontsize=9.5, ha="right", color=C["ink"])
ax.axvline(20, color=C["grid"], lw=6, zorder=0)
ax.text(20.8, 95, "the cut we report", fontsize=9.5, color=C["ink"])
ax.set_xlabel("Share of the add-on's strength removed (%)"); ax.set_ylabel("Harm score (0 = original, 100 = uncut)")
ax.set_xlim(5, 62); ax.set_ylim(0, 100); ax.set_xticks([10, 20, 30, 38, 57])
ax.grid(axis="y", color=C["grid"]); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9.5, loc="lower left")
ax.set_title("Problem 9, three public add-ons, mean of seeds 0 to 2", fontsize=10, color=C["ink"], loc="left")
save(fig, "fig5_ablation_cut_size.png")
print("ok")
