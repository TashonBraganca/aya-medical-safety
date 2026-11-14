"""Tables for p9_causal from e.json, d.json and labels.csv (numbers and labels only, no answer text).  python analyze.py <dir with e.json d.json labels.csv>"""
import csv, json, math, os, statistics as S, sys
D0 = sys.argv[1]; e = json.load(open(f"{D0}/e.json")); d = json.load(open(f"{D0}/d.json")); A = e["arms"]
def sign_p(a, b):
    """Exact two-sided sign test on discordant pairs: a = questions harmful only in arm 1, b = only in arm 2."""
    n = a + b
    if n == 0: return 1.0
    k = min(a, b); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); run = 0
    for r, i in enumerate(o): run = max(run, min(1.0, ps[i] * (len(ps) - r))); out[i] = run
    return out
print("# E. harm score (323 test pairs; 0 = original, 100 = broken)\n")
print("| arm | draws | harm score (min-max) | weight removed | signal removed | signal, centred | general-text drift |\n|---|---|---|---|---|---|---|")
def row(label, names):
    v = [A[n] for n in names if n in A]
    if not v: print(f"| {label} | 0 | not reachable | | | | |"); return
    L = [x["lean"] for x in v]; m = lambda k: S.mean(x[k] for x in v)
    print(f"| {label} | {len(v)} | {S.mean(L):.1f}" + (f" ({min(L):.1f}-{max(L):.1f})" if len(v) > 1 else "") + f" | {100 * m('weight_dose'):.1f}% | {100 * m('signal_dose_med'):.1f}% | {100 * m('signal_dose_med_centred'):.1f}% | {m('gen_drift'):+.3f} |")
pre = lambda p: sorted(n for n in A if n.startswith(p))
row("fingerprint k=1", ["fp_k1"]); row("fingerprint k=8", ["fp_k8"])
row("weight-matched k1: random strong", pre("w_rand_strong@k1")); row("weight-matched k1: random rotation", pre("w_rand_rot@k1"))
for k in ("k1", "k8"):
    for lab, p in (("scale whole add-on", "s_scale@"), ("biggest first", "s_biggest@"), ("most signal first", "s_topsignal@"), ("outside fingerprint", "s_outside@"),
                   ("random strong", "s_rand_strong@"), ("random rotation", "s_rand_rot@"), ("random any", "s_rand_all@")): row(f"signal-matched {k}: {lab}", pre(p + k))
r = e["refs"]; print(f"\nGeneral-text log-prob per token: original {r['gen_lp_orig']:.3f}, broken {r['gen_lp_broken']:.3f} (drift is measured from broken; original sits at {r['gen_lp_orig'] - r['gen_lp_broken']:+.3f}).")
print(f"Fingerprint k=8 share of the add-on: {100 * r['fingerprint_share_of_energy']:.1f}% of weight energy, {100 * r['fingerprint_share_of_signal_med']:.1f}% of signal. Unreachable arms: {e.get('unreachable', {})}")
w = [A[n]["lean"] for n in pre("w_rand_")]; print(f"Weight-matched: fp_k1 {A['fp_k1']['lean']:.1f} is below {sum(x > A['fp_k1']['lean'] for x in w)} of {len(w)} random draws.")
for k in ("k1", "k8"):
    s = [A[n]["lean"] for n in A if n.startswith("s_") and n.endswith(k) or (n.startswith("s_") and f"@{k}_" in n)]; t = A[f"fp_{k}"]["lean"]
    print(f"Signal-matched {k}: fp {t:.1f} is below {sum(x > t for x in s)} of {len(s)} controls (specific on the harm score needs all).")
print("\n# D. putting the fingerprint signal back (share of what full removal closed)\n")
W = d["W"]; print(f"W* = {W['m']} nodes {W['nodes']} (reached half on pick: {W['reached_half_on_pick']}), {W['patches']} patches. Identity error {d['refs']['identity_error']:.4f}.")
print("pick curve:", {m: round(v["restore"], 3) for m, v in d["pick_curve"].items()}); print("test curve:", {m: round(v["restore"], 3) for m, v in d["test_curve"].items()})
t = d["test"]; g = lambda p, k="restore": [v[k] for n, v in t.items() if n.startswith(p)]
print(f"restore at W* {t['W']['restore']:.3f} (necessity {t['W']['necessity']:.3f}); random node sets {min(g('random_nodes')):.3f}-{max(g('random_nodes')):.3f} mean {S.mean(g('random_nodes')):.3f} (necessity {min(g('random_nodes', 'necessity')):.3f}-{max(g('random_nodes', 'necessity')):.3f});")
print(f"randwrite {min(g('randwrite')):.3f} to {max(g('randwrite')):.3f}; ownother {min(g('ownother')):.3f} to {max(g('ownother')):.3f}. Signal put back at W*: {d['dose']}")
print("by group:", {k: (round(v["restore"], 3), round(v["necessity"], 3)) for k, v in d["d1_groups"].items()})
d1 = d["d1"]; mlp = [v["restore"] for k, v in d1.items() if k.endswith("mlp")]; att = [v["restore"] for k, v in d1.items() if k.endswith("attn")]
print(f"single nodes: MLP restore {min(mlp):.3f}-{max(mlp):.3f} (sum {sum(mlp):.2f}); attention {min(att):.3f}-{max(att):.3f} (sum {sum(att):.2f})")
print("per floor MLP restore:", [round(d1[f"{L}|mlp"]["restore"], 3) for L in range(36)]); print("per floor MLP necessity:", [round(d1[f"{L}|mlp"]["necessity"], 3) for L in range(36)])
print("per floor attn restore:", [round(d1[f"{L}|attn"]["restore"], 3) for L in range(36)])
print("directions (consensus m):", {m: (round(v["restore_at_W"], 3), round(v["restore_everywhere"], 3)) for m, v in d["d3"].items()})
if "d4" in d:
    x = d["d4"]; print(f"\nD4 ({x['n_pairs']} pairs): TE {x['TE']:.4f}; direct share {x['direct_share']:.3f}; replayed-from-broken share {x['replayed_from_broken_run_share']:.3f}; position {x['position']}")
    print("carriers (necessity, sufficiency):", [(n["floor"], n["type"], round(n["necessity"], 3), round(n["sufficiency"], 3)) for n in x["nodes"][:12]])
    print("joint freezing of top carriers:", {m: round(v, 3) for m, v in x["cumulative_necessity_top_carriers"].items()}); print("senders:", x["senders"])
    print("edges (from, to, weight, shuffled control):", [(tuple(q["from"]), tuple(q["to"]), round(q["weight"], 3), round(q.get("shuffled_control", float("nan")), 3)) for q in x["edges"][:12]])
    ew = [q["weight"] for q in x["edges"]]; print(f"all {len(ew)} edges: max {max(ew):.3f}, sum {sum(ew):.3f}; paths by the rule (>= 0.10 and >= 3x control): {[(q['from'], q['to']) for q in x['edges'][:12] if q['weight'] >= 0.10 and q['weight'] >= 3 * abs(q.get('shuffled_control', 1))]}")
lp = f"{D0}/labels.csv"
if os.path.exists(lp):
    lab = {}; hit = {}; ch = {}
    for r_ in csv.DictReader(open(lp)): lab.setdefault(r_["arm"], {})[r_["question_id"]] = r_["label"]; hit.setdefault(r_["arm"], []).append(int(r_["hit_max"])); ch.setdefault(r_["arm"], []).append(int(r_["chars"]))
    print("\n# Judged answers (one blind labeller pass)\n\n| arm | n | harmful | refused | safe | broken | unparsed | hit token limit | median chars |\n|---|---|---|---|---|---|---|---|---|")
    for a in sorted(lab):
        c = lambda x: sum(v == x for v in lab[a].values()); print(f"| {a} | {len(lab[a])} | {c('HARMFUL')} | {c('REFUSED')} | {c('SAFE')} | {c('BROKEN')} | {c('UNPARSED')} | {sum(hit[a])} | {int(S.median(ch[a]))} |")
    def cmp(a, b):
        if a not in lab or b not in lab: return None
        q = sorted(set(lab[a]) & set(lab[b])); x = sum(lab[a][i] == "HARMFUL" and lab[b][i] != "HARMFUL" for i in q); y = sum(lab[a][i] != "HARMFUL" and lab[b][i] == "HARMFUL" for i in q)
        return {"a": a, "b": b, "harmful_a": sum(lab[a][i] == "HARMFUL" for i in q), "harmful_b": sum(lab[b][i] == "HARMFUL" for i in q), "only_a": x, "only_b": y, "p": sign_p(x, y)}
    def family(title, pairs):
        rs = [c for c in (cmp(a, b) for a, b in pairs) if c]; hp = holm([c["p"] for c in rs]); print(f"\n{title}")
        for c, h in zip(rs, hp): print(f"- {c['a']} {c['harmful_a']} vs {c['b']} {c['harmful_b']} (only first {c['only_a']}, only second {c['only_b']}), p = {c['p']:.4f}, Holm {h:.4f}")
    family("E, k=1 signal dose: fingerprint against each control", [("E_fp_k1", "E_" + x) for x in ("s_scale@k1", "s_biggest@k1", "s_topsignal@k1", "s_rand_strong@k1_d0", "s_rand_rot@k1_d0", "s_outside@k1")])
    family("E, k=8 signal dose", [("E_fp_k8", "E_s_scale@k8"), ("E_fp_k8", "E_s_rand_strong@k8_d0")])
    family("E, references", [("E_s2_broken", "E_original"), ("E_fp_k1", "E_s2_broken"), ("E_fp_k8", "E_s2_broken"), ("E_fp_k1", "E_fp_k8"), ("E_fp_k1", "E_original"), ("E_fp_k8", "E_original")])
    family("D rule 4: putting the signal back at W* against R and both signal controls", [("D_R_plus_piece_W", "D_R"), ("D_R_plus_piece_W", "D_R_plus_randwrite_W"), ("D_R_plus_piece_W", "D_R_plus_ownother_W")])
    family("D rule 5 and descriptive", [("D_broken_minus_piece_W", "E_s2_broken"), ("D_R_plus_piece_W", "D_R_plus_piece_random_nodes"), ("D_R", "E_fp_k8"), ("D_R_plus_randwrite_W", "D_R"), ("D_R_plus_ownother_W", "D_R")])
    family("Long answers (1,536 tokens) against the same arm at 512", [(f"{p}L_{n}", f"{p}_{n}") for p, n in (("E", "original"), ("E", "s2_broken"), ("E", "fp_k1"), ("E", "fp_k8"), ("E", "s_scale@k1"), ("D", "R"), ("D", "R_plus_piece_W"))])
    family("Long answers: comparisons redone at 1,536 tokens", [("EL_fp_k1", "EL_s_scale@k1"), ("EL_fp_k1", "EL_s2_broken"), ("EL_fp_k8", "EL_s2_broken"), ("DL_R_plus_piece_W", "DL_R"), ("EL_fp_k1", "EL_fp_k8")])
    for a in ("ref_harmful", "ref_safe"):
        if a in lab: print(f"\nlabeller check {a}: {sum(v == 'HARMFUL' for v in lab[a].values())} of {len(lab[a])} labelled HARMFUL")
    print("\nHarmful share by answer length (chars), arms pooled:")
    rows = [(int(r_["chars"]), r_["label"] == "HARMFUL") for r_ in csv.DictReader(open(lp)) if not r_["arm"].startswith("ref_")]
    for lo, hi in ((0, 300), (300, 800), (800, 1500), (1500, 2500), (2500, 10 ** 9)):
        b = [h for c_, h in rows if lo <= c_ < hi]; print(f"- {lo}-{hi if hi < 10 ** 9 else 'up'}: {sum(b)} of {len(b)}" + (f" = {100 * sum(b) / len(b):.1f}%" if b else ""))
