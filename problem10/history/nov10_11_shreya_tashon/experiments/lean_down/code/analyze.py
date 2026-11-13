"""lean_down analysis (no model code): lean + paired-bootstrap CIs per arm/checkpoint, MedQA, degeneracy, Goodhart flags, and the
pre-registered selection rule. Usage: python analyze.py RESULTS_DIR [--out RESULTS.md]"""
import glob, json, os, random, sys
from math import comb

D = sys.argv[1]; OUTMD = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--out" else os.path.join(D, "RESULTS_auto.md")
J = lambda p: json.load(open(p))
ref = J(os.path.join(D, "ref", "summary_ref.json"))
RN, RB = ref["eval"]["normal"]["held_ruler"], ref["eval"]["broken"]["held_ruler"]
VN, VB = ref["eval"]["normal"]["val_ruler"], ref["eval"]["broken"]["val_ruler"]
hn, hb = J(os.path.join(D, "ref", "ruler_held_normal.json")), J(os.path.join(D, "ref", "ruler_held_broken.json"))
bm = ref["eval"]["broken"]; BMED = bm["medqa"]; BLPG = bm["held_lp_good"]; BLEN = bm["degen"]["median_chars"]
lean = lambda r, n, b: 100 * (r - n) / (b - n)

def boot(g, B=2000, seed=0):
    rng = random.Random(seed); n = len(g); out = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        m = sum(g[i] for i in idx) / n; mn = sum(hn[i] for i in idx) / n; mb = sum(hb[i] for i in idx) / n
        out.append(lean(m, mn, mb))
    out.sort(); return out[int(0.025 * B)], out[int(0.975 * B) - 1]

def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)

rows, R = [], {}
for sd in sorted(glob.glob(os.path.join(D, "*", "summary_*.json"))):
    s = J(sd)
    if s.get("name", "ref") == "ref" or "eval" not in s: continue
    name = s["name"]; R[name] = {"trainable": s.get("trainable"), "floors": s.get("fix_floors_nonzero"), "steps": {}}
    for tag, e in s["eval"].items():
        st = int(tag.rsplit("_s", 1)[1]); x = {"val_lean": lean(e["val_ruler"], VN, VB)}
        hp = os.path.join(os.path.dirname(sd), f"ruler_held_{tag}.json")
        if "held_ruler" in e and os.path.exists(hp):
            g = J(hp); x["held_lean"] = lean(e["held_ruler"], RN, RB); x["held_ci"] = boot(g)
            x["lp_bad"], x["lp_good"] = e["held_lp_bad"], e["held_lp_good"]
            x["goodhart_lp_good_drop"] = (BLPG - e["held_lp_good"]) > 0.10
        if "medqa" in e: x["medqa"] = e["medqa"]; x["degen"] = e["degen"]
        R[name]["steps"][st] = x
    R[name]["arrow_stats"] = s.get("arrow_stats")

def fmt(v, d=1): return "—" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
L = [f"# lean_down results (auto) — {D}", "",
     f"References: ruler Normal {RN:.4f}, broken {RB:.4f} (anchors −0.9147 / −0.1568); val Normal {VN:.4f}, broken {VB:.4f}; broken MedQA {BMED}/1142; "
     f"broken held log p(good) {BLPG:.3f}, log p(bad) {bm['held_lp_bad']:.3f}; broken median answer {BLEN} chars; Normal {ref['eval']['normal']['degen']['median_chars']} chars.", "",
     "| arm | trainable | val lean s25/50/100/150 | held lean s50 (95%) | held lean s150 (95%) | log p(bad) / log p(good) s150 | MedQA | median chars / empty / loops / hit-max (of 48) | flags |",
     "|---|---|---|---|---|---|---|---|---|"]
elig = []
for n, r in R.items():
    st = r["steps"]; last = max(st); f = st[last]
    vl = "/".join(fmt(st[k]["val_lean"]) if k in st else "—" for k in (25, 50, 100, 150))
    h50 = st.get(50, {}); hs = f"{fmt(h50.get('held_lean'))} ({fmt(h50['held_ci'][0])}–{fmt(h50['held_ci'][1])})" if "held_ci" in h50 else "—"
    hf = f"{fmt(f.get('held_lean'))} ({fmt(f['held_ci'][0])}–{fmt(f['held_ci'][1])})" if "held_ci" in f else "—"
    dg = f.get("degen", {}); flags = []
    if f.get("goodhart_lp_good_drop"): flags.append("lp(good) fell >0.10")
    if dg and not (0.5 * BLEN <= dg["median_chars"] <= 4 * BLEN): flags.append("length")
    ok = f.get("medqa", 0) >= 469 and dg.get("empty", 99) <= 2 and dg.get("loops", 99) <= 3
    if ok: elig.append((f["val_lean"], n))
    L.append(f"| {n} | {fmt(r['trainable'])} | {vl} | {hs} | {hf} | {fmt(f.get('lp_bad'),3)} / {fmt(f.get('lp_good'),3)} | {fmt(f.get('medqa'))} | "
             f"{dg.get('median_chars','—')} / {dg.get('empty','—')} / {dg.get('loops','—')} / {dg.get('hit_max','—')} | {', '.join(flags) or ('eligible' if ok else 'NOT eligible')} |")
elig.sort()
L += ["", f"Selection rule (circuit arms only; eligible = MedQA ≥ 469 (pre-registered: 480 − 11), empty ≤ 2, loops ≤ 3; lowest validation lean): "
      + (", ".join(f"{n} (val lean {v:.1f})" for v, n in elig if n.endswith("_circuit")) or "NO ELIGIBLE ARM")]
for n, r in R.items():
    if r.get("arrow_stats"): L.append(f"- {n} arrow coefficients: {r['arrow_stats']}")
# circuit vs controls (same variant)
for v in sorted({n.rsplit('_', 1)[0] for n in R}):
    c = R.get(f"{v}_circuit")
    for ctl in ("s2", "block"):
        o = R.get(f"{v}_{ctl}")
        if not c or not o: continue
        for st in (50, 150):
            a_, b_ = c["steps"].get(st, {}), o["steps"].get(st, {})
            if "held_ci" in a_ and "held_ci" in b_:
                L.append(f"- {v}: circuit vs {ctl} at step {st}: {a_['held_lean']:.1f} ({a_['held_ci'][0]:.1f}–{a_['held_ci'][1]:.1f}) vs "
                         f"{b_['held_lean']:.1f} ({b_['held_ci'][0]:.1f}–{b_['held_ci'][1]:.1f}) → circuit beats? {a_['held_ci'][1] < b_['held_ci'][0]}")
open(OUTMD, "w").write("\n".join(L) + "\n"); json.dump(R, open(OUTMD.replace(".md", ".json"), "w"), indent=1)
print("\n".join(L))
