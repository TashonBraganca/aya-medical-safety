"""wso (§50) analysis: collect every operating-point candidate from downloaded R:wso/results/* dirs, apply the PRE-REGISTERED rule,
write frontier.md + frontier.png + w1_curve.png. Usage: python analyze_wso.py RES_DIR OUT_DIR
RES_DIR contains w1/, w2*/, w3core*/, w3dial*/, w4*/ (as on R). No model code; CPU only."""
import glob, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
A = json.load(open(os.path.join(RES, "w1", "anchors.json")))
KTH, MFLOOR = A["kl_threshold"], A["medqa300_floor"]
cands, curves = [], {}
for s in sorted(glob.glob(os.path.join(RES, "*", "*", "summary.json"))):
    S = json.load(open(s)); job = s.split(os.sep)[-3]; nm = S["name"]
    curves[nm] = [(h["step"], h["val_lean"]) for h in S["hist"]]
    for h in S["hist"]:
        if h.get("medqa300") is None: continue
        cands.append(dict(job=job, arm=h.get("arm", nm), step=h["step"], val=h["val_lean"], mq=h["medqa300"], kl=h.get("kl_gen"), held=h.get("held_lean"),
                          lpg=h.get("val_lp_good"), floors=S["floors"], mode=S["mode"], phase=h.get("phase"),
                          eligible=bool(h.get("eligible"))))
J1 = json.load(open(os.path.join(RES, "w1", "job_summary.json")))
for e in J1.get("w0", []):
    cands.append(dict(job="w1", arm=e["name"], step=0, val=e.get("val_lean"), mq=e.get("medqa300"), kl=e.get("kl_gen"), held=None,
                      lpg=e.get("val_lp_good"), floors=list(range(36)), mode="dial(one-shot)", phase="one-shot", eligible=bool(e.get("eligible"))))
seen, uniq = set(), []                       # W1_long re-lists the chosen sweep run's history: keep one copy per (arm-history, step)
for c in cands:
    k = (c["arm"], c["step"])
    if k not in seen: seen.add(k); uniq.append(c)
cands = uniq
el = [c for c in cands if c["eligible"] and c["val"] is not None]
win = min(el, key=lambda c: c["val"]) if el else None
circ = [c for c in el if c["floors"] == list(range(9, 19))]
bcirc = min(circ, key=lambda c: c["val"]) if circ else None
f = lambda v, d=1: "–" if v is None else f"{v:.{d}f}"
L = [f"# wso frontier (validation split only; held-out NOT used for selection)", "",
     f"Anchors: broken MedQA-300 {A['medqa300_broken']} → floor {MFLOOR}; KL threshold (P's general-text KL) {KTH:.4f}. "
     f"P: val lean {A['P']['val_lean']:.1f}, MedQA-300 {A['P']['medqa300']}.", "",
     f"**Winner (pre-registered rule):** {win['arm']} step {win['step']} val lean {win['val']:.1f}" if win else "**No eligible candidate.**",
     f"**Best circuit-restricted:** {bcirc['arm']} step {bcirc['step']} val lean {bcirc['val']:.1f}" if bcirc else "**No eligible circuit-restricted candidate.**", "",
     "| arm | step | phase | val lean | MedQA-300 | gen KL | eligible | held lean (reported only) |", "|---|---|---|---|---|---|---|---|"]
for c in sorted(cands, key=lambda c: (c["arm"], c["step"])):
    L.append(f"| {c['arm']} | {c['step']} | {c['phase']} | {f(c['val'])} | {c['mq'] if c['mq'] is not None else '–'} | {f(c['kl'], 4)} | "
             f"{'yes' if c['eligible'] else 'no'} | {f(c['held'])} |")
# best eligible per arm (frontier summary)
L += ["", "## Best eligible point per arm", "| arm | step | val lean | MedQA-300 | gen KL |", "|---|---|---|---|---|"]
for arm in sorted({c["arm"] for c in el}):
    b = min([c for c in el if c["arm"] == arm], key=lambda c: c["val"])
    L.append(f"| {arm} | {b['step']} | {b['val']:.1f} | {b['mq']} | {b['kl']:.4f} |")
L += ["", "## POST HOC: best point per arm using the MedQA-300 floor only (KL condition dropped; KL shown)",
      "| arm | step | val lean | MedQA-300 | gen KL (limit %.4f) | held lean (if measured) |" % KTH, "|---|---|---|---|---|---|"]
mq_ok = [c for c in cands if c["val"] is not None and c["mq"] is not None and c["mq"] >= MFLOOR]
for arm in sorted({c["arm"] for c in mq_ok}):
    b = min([c for c in mq_ok if c["arm"] == arm], key=lambda c: c["val"])
    L.append(f"| {arm} | {b['step']} | {b['val']:.1f} | {b['mq']} | {f(b['kl'], 4)} | {f(b['held'])} |")
open(os.path.join(OUT, "frontier.md"), "w").write("\n".join(L) + "\n")
json.dump({"winner": win, "best_circuit": bcirc, "candidates": cands}, open(os.path.join(OUT, "frontier.json"), "w"), indent=1)

# ---- figures (reference palette, light mode; first three categorical slots + neutral gray for the one-shot W0)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"dial, all floors (W1/W2/W4)": "#2a78d6", "circuit floors 9-18 (W3)": "#eb6834", "random / all floors (W3)": "#1baf7a", "one-shot (W0)": "#8a8984"}
def fam(c):
    if c["arm"].startswith("W0"): return "one-shot (W0)"
    if c["arm"].startswith("W3"): return "circuit floors 9-18 (W3)" if c["floors"] == list(range(9, 19)) else "random / all floors (W3)"
    return "dial, all floors (W1/W2/W4)"
MK = {"dial, all floors (W1/W2/W4)": "o", "circuit floors 9-18 (W3)": "s", "random / all floors (W3)": "^", "one-shot (W0)": "D"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
pts = [c for c in cands if c["val"] is not None and c["mq"] is not None and c["kl"] is not None and c["kl"] < 0.2]
for k, (xk, xl) in enumerate((("mq", "MedQA-300 (correct of 300)"), ("kl", "general-text KL to broken (per token)"))):
    a_ = ax[k]
    for fm, col in COL.items():
        P_ = [c for c in pts if fam(c) == fm]
        if not P_: continue
        E_, N_ = [c for c in P_ if c["eligible"]], [c for c in P_ if not c["eligible"]]
        if E_: a_.scatter([c[xk] for c in E_], [c["val"] for c in E_], s=36, c=col, marker=MK[fm], edgecolors="white", linewidths=1.5,
                          label=fm + " (eligible)", zorder=3)
        if N_: a_.scatter([c[xk] for c in N_], [c["val"] for c in N_], s=30, facecolors="none", edgecolors=col, marker=MK[fm], linewidths=1.2,
                          label=fm + " (not eligible)", zorder=2)
    a_.axvline(MFLOOR if xk == "mq" else KTH, color=INK2, lw=1, ls="--", zorder=1)
    a_.text(MFLOOR if xk == "mq" else KTH, a_.get_ylim()[1] if False else 0.98, " floor" if xk == "mq" else " threshold (P)", transform=a_.get_xaxis_transform(),
            color=INK2, va="top", fontsize=8)
    for ref, lab in ((49.2, "A1 49 / Tashon 47 (held-out)"), (A["P"]["val_lean"], "P (val)")):
        a_.axhline(ref, color=GRID, lw=1, zorder=0)
        if k == 1: a_.text(1.0, ref, f" {lab}", transform=a_.get_yaxis_transform(), color=INK2, va="center", fontsize=7)
    if win: a_.scatter([win[xk]], [win["val"]], s=140, facecolors="none", edgecolors=INK, linewidths=1.5, zorder=4)
    a_.set_xlabel(xl); a_.grid(True, color=GRID, lw=0.6); a_.set_axisbelow(True)
    for sp in ("top", "right"): a_.spines[sp].set_visible(False)
ax[0].set_ylabel("validation lean (0 = Normal Aya, 100 = broken)")
h_, l_ = ax[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, fontsize=7.5)
fig.suptitle("Weight-space edits inside em0's subspace: safety (lower = better) vs skill. " + ("Ringed = pre-registered winner." if win else
             "No checkpoint passes both limits (dashed lines)."), color=INK, fontsize=10)
fig.tight_layout(rect=(0, 0.12, 0.93, 0.95)); fig.savefig(os.path.join(OUT, "frontier.png"), dpi=160)

# ---- W1 learning curves
fig, a_ = plt.subplots(figsize=(7, 4))
LC = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
names = [n for n in sorted(curves) if n.startswith("W1_lr")] + (["W1_long"] if "W1_long" in curves else [])
for col, n in zip(LC, names):
    pts_ = [(x, y) for x, y in curves[n] if n != "W1_long" or x >= 300]
    xs, ys = zip(*pts_); a_.plot(xs, ys, color=col, lw=2, label=n + (" (continues lr 0.06 from step 300)" if n == "W1_long" else ""))
pl = J1.get("w1_long", {}).get("plateau")
if pl: a_.axvline(pl["step"], color=INK2, ls="--", lw=1); a_.text(pl["step"], 0.98, " plateau rule fired", transform=a_.get_xaxis_transform(), color=INK2, va="top", fontsize=8)
a_.axvline(150, color=GRID, lw=1); a_.text(150, 0.02, " §49 stopped here", transform=a_.get_xaxis_transform(), color=INK2, fontsize=8)
a_.axhline(49.2, color=GRID, lw=1)
a_.set_xlabel("training step (8 pairs/step; 768 steps = 1 pass)"); a_.set_ylabel("validation lean"); a_.grid(True, color=GRID, lw=0.6)
for sp in ("top", "right"): a_.spines[sp].set_visible(False)
a_.legend(frameon=False, fontsize=8); a_.set_title("W1: one dial per arrow, c in [0,1] — does it keep improving?", color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "w1_curve.png"), dpi=160)
print("winner", win and (win["arm"], win["step"], round(win["val"], 2)), "best circuit", bcirc and (bcirc["arm"], bcirc["step"], round(bcirc["val"], 2)))
