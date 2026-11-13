"""Summary of ck_native_p10 (numbers only): Part A gates, rulers, judge, MedQA, XSTest, pre-registered rule A; Part B calibration,
rulers, rule B, judge, nothing-broke. Writes ROOT/summary.json and prints markdown. Usage: python analyze.py --root DIR"""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
ld = lambda *p: json.load(open(os.path.join(a.root, *p))) if os.path.exists(os.path.join(a.root, *p)) else None
A, B, J, NB = ld("partA", "partA.json"), ld("partB", "partB.json"), ld("gens", "summary_judge.json"), ld("nb", "nb_results.json")
S = {}; out = []
pct = lambda x: f"{100 * x:.1f}"
ci = lambda t: f"{pct(t['gap_closed'])} ({pct(t['ci95'][0])}–{pct(t['ci95'][1])})"
hp = lambda m: (J or {}).get(m, {}).get("harmful_pct")
fh = lambda m: "n/a" if not hp(m) else f"{hp(m)[0]} ({hp(m)[1]}–{hp(m)[2]})"
out.append("# ck_native_p10 — automatic summary\n")
if A:
    out.append("## Part A — CircuitKIT ActivationSteering (floors 14+16, alpha 2)\n")
    out.append(f"Gates: G1 fp32 HF vs CK-TL top-1 {A.get('G1', {}).get('top1_agreement')} (PASS={A.get('G1', {}).get('PASS')}); "
               f"G2 wrapper vs plain steer() argmax {A.get('G2', {}).get('argmax_agree')} on {A.get('G2', {}).get('generated_tokens')} tokens, "
               f"coef0==unsteered {A.get('G2', {}).get('coef0_equals_unsteered')} (PASS={A.get('G2', {}).get('PASS')}); G3 {A.get('G3')}\n")
    if "A2_vector" in A: out.append(f"A2 CircuitKIT-native vector: raw floor norms {A['A2_vector']['raw_floor_norm']}, cos with ours {A['A2_vector']['cos_with_ours']}, scale {A['A2_vector']['scale']}\n")
    T = A.get("ruler_table", {})
    out.append("| arm | ruler gap closed % (95% CI) | harmful % (Wilson 95%) | MedQA % (no letter) | XSTest over-refusal % |\n|---|---|---|---|---|")
    rows = [("TL broken", "tl_broken", "broken", "tl_broken"), ("HF our hook α2 (§39: 25.1, 24.1–26.2)", None, "hf_steer2", None),
            ("A1 CircuitKIT + our vectors", "ck_ours@2", "ck_ours2", "ck_ours2"), ("A2 CircuitKIT-native direction (norm-matched)", "ck_native@2", "ck_native2", "ck_native2"),
            ("A2 raw (unscaled)", "ck_native_raw@2", None, None)]
    for lab, rk, jk, mk in rows:
        mq = A.get("medqa", {}).get(mk) if mk else None; xs = A.get("xstest", {}).get(mk) if mk else None
        mqs = "%.1f (%d)" % (mq["pct"], mq["no_letter"]) if mq else "—"
        out.append(f"| {lab} | {ci(T[rk]) if rk in T else '—'} | {fh(jk) if jk else '—'} | {mqs} | {xs['overrefusal'][0] if xs else '—'} |")
    ra1 = A.get("RULE_A1", {}).get("PASS"); ra2 = None
    if hp("ck_ours2") and hp("hf_steer2"): ra2 = hp("hf_steer2")[1] <= hp("ck_ours2")[0] <= hp("hf_steer2")[2]
    S["RULE_A"] = {"R_A1_ruler_in_our_CI": ra1, "R_A2_harm_in_hf_wilson_CI": ra2, "REPLICATES": bool(ra1 and ra2) if ra2 is not None else None}
    out.append(f"\n**Rule A:** R-A1 (ruler in 24.1–26.2%) = {ra1}; R-A2 (A1 harmful inside HF-hook Wilson CI) = {ra2} → replicates = {S['RULE_A']['REPLICATES']}\n")
if B:
    out.append("## Part B — CircuitKIT selective fine-tune (down_proj, 10 floors, 225.4M weights per arm)\n")
    out.append(f"Calibration (circuit arm): {[(c['lr'], round(100 * c['gap_closed'], 1)) for c in B.get('calibration', [])]} → lr frozen {B.get('lr_frozen')} ({B.get('lr_source')})\n")
    T = B.get("ruler_table", {})
    out.append("| arm | floors | step 10 | step 25 | step 50 | harmful % at 25 |\n|---|---|---|---|---|---|")
    for arm in ("circuit", "random", "block"):
        out.append(f"| {arm} | {B['arms'][arm]} | " + " | ".join(ci(T[f'{arm}_s{s}']) if f'{arm}_s{s}' in T else '—' for s in (10, 25, 50)) + f" | {fh(f'ckft_{arm}25')} |")
    if "RULE_B" in B: S["RULE_B"] = {"PASS": B["RULE_B"]["PASS"], "circuit_ci": B["RULE_B"]["circuit"]["ci95"], "max_control_ci_high": B["RULE_B"]["max_control_ci_high"]}
    out.append(f"\n**Rule B (step 25, circuit CI-low > both controls' CI-high):** {S.get('RULE_B')}\n")
    out.append(f"Paired diffs: { {k: (round(100 * v['point'], 1), [round(100 * x, 1) for x in v['ci95']]) for k, v in B.get('paired_diffs', {}).items()} }\n")
    fc = B.get("fp32_check", {}); out.append(f"fp32 check (40 pairs, ruler change): fp32 {fc.get('effect_fp32')} vs bf16+dW {fc.get('effect_bf16delta')}\n")
if J:
    out.append(f"## Judge\nbroken {fh('broken')}; judge check {J.get('judge_check')}; unparsed {J.get('unparsed')}\n\nPaired (McNemar): {json.dumps(J.get('paired'))}\n")
if NB:
    out.append("## Nothing-broke (nb.py)\n| model | MMLU | G-MMLU es | hi | sw | ppl | XSTest over-refusal |\n|---|---|---|---|---|---|---|")
    for m in ("broken", "ckft_circuit25"):
        r = NB.get(m, {})
        if "error" in r or not r: out.append(f"| {m} | {r.get('error', 'missing')} |"); continue
        g = lambda k: (r.get(k) or ["—"])[0] if isinstance(r.get(k), list) else r.get(k, "—")
        out.append(f"| {m} | {g('mmlu')} | {g('gmmlu_es')} | {g('gmmlu_hi')} | {g('gmmlu_sw')} | {g('ppl')} | {g('xstest_overrefusal')} |")
S["partA"] = A and {k: A.get(k) for k in ("G1", "G2", "G3", "RULE_A1", "medqa", "xstest", "A2_vector", "timing")}
S["partB"] = B and {k: B.get(k) for k in ("calibration", "lr_frozen", "lr_source", "RULE_B", "paired_diffs", "fp32_check", "timing", "data")}
if A: S["partA"]["ruler_table"] = A.get("ruler_table")
if B: S["partB"]["ruler_table"] = B.get("ruler_table")
S["judge"] = J and {k: v for k, v in J.items()}; S["nb"] = NB
json.dump(S, open(os.path.join(a.root, "summary.json"), "w"), indent=1)
print("\n".join(out))
