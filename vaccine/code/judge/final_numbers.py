import json, math, os
from lab_lookup import load, label
C = {"judge_a": (load("out_judge_a"), "<hosted-reviewer-deployment>"), "deepseek": (load("out_deepseek"), "deepseek/deepseek-v4.1-flash")}
def w(k, n):
    if n == 0: return None
    p = k / n; z = 1.96; d = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return [round(100*p, 1), round(100*(c-h)/d, 1), round(100*(c+h)/d, 1)]
F = {
 "broken": "a__gen_broken", "normal": "a__gen_normal_s0", "fixed": "a__gen_fixed_s0",
 "fixed_4bit": "c__gen_fixed_nf4", "broken_4bit": "c__gen_broken_nf4", "blame64": "c__gen_blame64_off", "big64": "c__gen_big64_off",
 "normal_att50_s0": "a__gen_normal_seed0_s50", "normal_att50_s1": "a__gen_normal_seed1_s50", "fixed_att50_s0": "a__gen_fixed_seed0_s50", "fixed_att50_s1": "a__gen_fixed_seed1_s50",
 "refix_s0": "a__gen_R4_merged_big512_off_seed0", "refix_s1": "a__gen_R4_merged_big512_off_seed1", "refix_rand_s0": "a__gen_RC_rand256_0_seed0", "refix_rand_s1": "a__gen_RC_rand256_0_seed1",
 "twojob": "b__gen_twojob", "twojob_fix": "b__gen_twojob_big256_off", "twojob_blame": "b__gen_twojob_blame256_off",
 "vacc_pre": "e6__gen_circuit_pre", "vacc_att50_s0": "e6__gen_circuit_seed0_lr1e-05_s50", "vacc_att50_s1": "e6__gen_circuit_seed1_lr1e-05_s50", "vacc_strong": "e6__gen_circuit_seed0_lr5e-05_s50",
 "e6normal_att50_s0": "e6__gen_normal_seed0_lr1e-05_s50", "e6normal_att50_s1": "e6__gen_normal_seed1_lr1e-05_s50", "e6normal_strong": "e6__gen_normal_seed0_lr5e-05_s50",
 "e6none_att50_s0": "e6__gen_none_seed0_lr1e-05_s50", "e6none_att50_s1": "e6__gen_none_seed1_lr1e-05_s50", "e6none_strong": "e6__gen_none_seed0_lr5e-05_s50",
 "rand_vacc_att50_s0": "e6__gen_random_seed0_lr1e-05_s50", "rand_vacc_att50_s1": "e6__gen_random_seed1_lr1e-05_s50",
 "all_vacc_att50_s0": "e6__gen_all_seed0_lr1e-05_s50", "all_vacc_att50_s1": "e6__gen_all_seed1_lr1e-05_s50",
 "ck_vacc_att50_s0": "e6__gen_ckexport_seed0_lr1e-05_s50", "ck_vacc_att50_s1": "e6__gen_ckexport_seed1_lr1e-05_s50",
 "crft_pre": "e6__gen_crft_pre", "crft_att50_s0": "e6__gen_crft_seed0_lr1e-05_s50", "crft_att50_s1": "e6__gen_crft_seed1_lr1e-05_s50",
 "crft_vacc_pre": "e6__gen_crft_circuit_pre", "crft_vacc_att50_s0": "e6__gen_crft_circuit_seed0_lr1e-05_s50", "crft_vacc_att50_s1": "e6__gen_crft_circuit_seed1_lr1e-05_s50",
 "long_vacc_50": "e6__gen_circuit_long200_seed0_lr1e-05_s50", "long_vacc_100": "e6__gen_circuit_long200_seed0_lr1e-05_s100", "long_vacc_200": "e6__gen_circuit_long200_seed0_lr1e-05_s200",
 "long_normal_50": "e6__gen_normal_long200_seed0_lr1e-05_s50", "long_normal_100": "e6__gen_normal_long200_seed0_lr1e-05_s100", "long_normal_200": "e6__gen_normal_long200_seed0_lr1e-05_s200",
 "long_fix_50": "e6__gen_none_long200_seed0_lr1e-05_s50", "long_fix_100": "e6__gen_none_long200_seed0_lr1e-05_s100", "long_fix_200": "e6__gen_none_long200_seed0_lr1e-05_s200",
 "off_broken": "d__gen_official_broken", "off_normal": "d__gen_official_normal", "off_fixed": "d__gen_official_fixed", "off_vacc": "d__gen_official_vaccine",
 "ref_harmful": "../ref_harmful", "ref_safe": "../ref_safe"}
out = {}
for k, f in F.items():
    p = f"pulled/{f}.jsonl" if not f.startswith("..") else f[3:] + ".jsonl"
    if not os.path.exists(p): out[k] = None; continue
    its = [json.loads(l) for l in open(p)]; row = {"n_answers": len(its), "median_chars": sorted(len(i["reply"]) for i in its)[len(its)//2]}
    for j, (cache, dep) in C.items():
        if k in ("ref_harmful", "ref_safe"): cache = load("out_ref_" + j)
        L = [label(cache, dep, i) for i in its]; got = [x for x in L if x]
        row[j] = w(sum(x == "HARMFUL" for x in got), len(got)) if len(got) >= 0.95 * len(its) else None; row[j + "_n"] = len(got)
    out[k] = row
json.dump(out, open("FINAL_NUMBERS.json", "w"), indent=1)
for k, v in out.items(): print(f"{k:22s}", "missing file" if v is None else f"judge_a {v['judge_a']}  deepseek {v['deepseek']}  (n {v['judge_a_n']}/{v['deepseek_n']})")
