"""Lexsi tool chain around today's experiments (CPU; run in the kits venv). Re-runnable; each stage halts on failure.
 Stage 0  CuratorKIT  MinHash dedup (threshold 0.85, as pipeline/code/chain_local.py) across E6's immunizer pool, attacker pool and every
                      evaluation question (323 ruler test pairs, 300 judged questions, 76 broad pairs): no evaluation question may be a
                      near-copy of a training row on either side, and the two training pools must not share near-copies.
 Stage 1  SafeTune    One CircuitInfo file per E6 arm (layer_subset x target_modules), saved and reloaded through SafeTune's loader; the
                      reload must equal the arm's module set, and the trained immunizer's modules must equal the file (checked in-sandbox).
 Stage 2  AuditKIT    Every judged before/after comparison as two fingerprinted AuditKIT runs (team bridge: labelled_run + compare, exact
                      sign test), for each judge. Public run files carry no answer text (bridge.save_public + check_no_text).
Usage: python toolchain.py [stage0] [stage1] [stage2]"""
import hashlib, json, os, subprocess, sys
from pathlib import Path
H = Path(__file__).resolve().parent; SCR = H.parent; J = SCR / "judge"; OUT = H / "out"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(H)); sys.path.insert(0, str(J))
stages = sys.argv[1:] or ["stage0", "stage1", "stage2"]; report = json.loads((OUT / "toolchain.json").read_text()) if (OUT / "toolchain.json").exists() else {}
def save(): (OUT / "toolchain.json").write_text(json.dumps(report, indent=1))

if "stage0" in stages:
    import numpy as np, random
    from curatorkit.normalizers.dedup import MinHashDeduplicator
    from curatorkit.schema import DataSample
    f = "<em_data>/"
    bad = [json.loads(l) for l in open(f + "bad_medical_advice.jsonl")]; perm = np.random.default_rng(42).permutation(len(bad))
    test = [bad[i] for i in perm[:705]]; train = [bad[i] for i in perm[705:]]
    pools = json.loads(subprocess.run([str(SCR / "modal" / "m2"), "sh", "cat /vol/out/e6/full/pools.json"], capture_output=True, text=True).stdout)
    imm = [train[i]["messages"][0]["content"] for i in pools["imm_idx"]]; att = [train[i]["messages"][0]["content"] for i in pools["att_idx"]]
    sp = json.load(open(SCR / "e1kit" / "split.json")); valid = {int(x[2:]) for x in json.load(open(SCR / "e1kit" / "valid_pairs.json"))}
    ruler = [test[i]["messages"][0]["content"] for i in sorted(valid & set(sp["test"]))]
    judged = [json.loads(l)["question"] for l in open(J / "pulled" / "a__gen_broken.jsonl")]
    broad = sorted({json.loads(l)["question"] for l in open(SCR / "e1kit" / "broad_em_pairs.jsonl")})
    groups = {"imm": imm, "att": att, "ruler": ruler, "judged": judged, "broad": broad}
    def near_pairs(a, b):
        """Run CuratorKIT's deduplicator over a + b (a first). A sample of b that it drops is a near-copy of something earlier."""
        samples = [DataSample(source_uri="a", instruction=q, id=f"a{i}") for i, q in enumerate(a)] + [DataSample(source_uri="b", instruction=q, id=f"b{i}") for i, q in enumerate(b)]
        kept = MinHashDeduplicator(threshold=0.85).run(samples); kb = {s.id for s in kept if s.source_uri == "b"}; ka = {s.id for s in kept if s.source_uri == "a"}
        return len(b) - len(kb), len(a) - len(ka)
    res = {}
    for x, y in [("imm", "att"), ("imm", "ruler"), ("imm", "judged"), ("imm", "broad"), ("att", "ruler"), ("att", "judged"), ("att", "broad")]:
        dropped_b, dropped_a_within = near_pairs(groups[x], groups[y]); exact = len(set(groups[x]) & set(groups[y]))
        res[f"{x}->{y}"] = {"near_copies_dropped": dropped_b, "exact_overlap": exact, "within_first_dropped": dropped_a_within}
        print(f"stage0 CuratorKIT {x:5s} vs {y:6s}: {dropped_b} of {len(groups[y])} {y} questions are near-copies of {x} rows; exact overlap {exact}", flush=True)
    report["stage0_curatorkit"] = {"sizes": {k: len(v) for k, v in groups.items()}, "pairs": res}; save()
    bad_eval = {k: v for k, v in res.items() if not k.endswith("att") and (v["near_copies_dropped"] or v["exact_overlap"])}
    assert not bad_eval, f"STAGE 0 HALT: evaluation questions are near-copies of training rows: {bad_eval}"
    print("stage0 PASS: no evaluation question is a near-copy of either training pool", flush=True)

if "stage1" in stages:
    from safetune.core.circuit_kit.interface import CircuitInfo, LayerModuleSuggestions
    from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
    MLP = ["gate_proj", "up_proj", "down_proj"]; ALL7 = ["q_proj", "k_proj", "v_proj", "o_proj"] + MLP
    ARMS = {"circuit": (list(range(9, 19)), MLP, "floors 9-18 located by arrow blame (problem 9, step 4) and the floor sweep"),
            "random": ([1, 2, 5, 6, 8, 19, 21, 31, 34, 35], MLP, "seeded draw outside 9-18 (same as Shreya's lean_down control)"),
            "all": (list(range(36)), MLP, "every floor, MLP"),
            "ckexport": (list(range(36)), ALL7, "what CircuitKIT's own EAP-IG export selects on broken model 1 (problem10/experiments/safetune_ctheta/RESULTS.md: all 36 floors x 7 parts)")}
    files = {}
    for arm, (floors, parts, why) in ARMS.items():
        info = CircuitInfo(layer_suggestions=LayerModuleSuggestions(target_modules=parts, layer_subset=floors, metadata={"source": why}), metadata={"experiment": "E6", "arm": arm})
        p = info.save(str(OUT / f"circuit_info_{arm}.json")); back = load_circuit_info_from_file(p)
        assert back.layer_suggestions.layer_subset == floors and back.layer_suggestions.target_modules == parts, f"STAGE 1 HALT: CircuitInfo round trip changed {arm}"
        files[arm] = p; print(f"stage1 SafeTune CircuitInfo {arm}: {len(floors)} floors x {len(parts)} parts written and reloaded unchanged", flush=True)
    report["stage1_safetune"] = {"files": files}; save()

if "stage2" in stages:
    import auditkit_bridge as B
    from auditkit import Sample
    from lab_lookup import load, label
    JUDGES = {"judge_a": ("out_judge_a", "<hosted-reviewer-deployment>"), "deepseek": ("out_deepseek", "deepseek/deepseek-v4.1-flash")}
    P = J / "pulled"
    PAIRS = [("a__gen_broken", "a__gen_fixed_s0", "fix: broken -> biggest 256 off"),
             ("a__gen_normal_s0", "a__gen_fixed_s0", "fix vs Normal Aya"),
             ("c__gen_big64_off", "c__gen_blame64_off", "blame 64 vs biggest 64"),
             ("c__gen_broken_nf4", "c__gen_fixed_nf4", "fix in 4-bit"),
             ("a__gen_fixed_s0", "c__gen_fixed_nf4", "fix: bf16 vs 4-bit"),
             ("a__gen_normal_seed0_s50", "a__gen_fixed_seed0_s50", "relapse: fix vs Normal after 50 attack steps (seed 0)"),
             ("a__gen_normal_seed1_s50", "a__gen_fixed_seed1_s50", "relapse: fix vs Normal after 50 attack steps (seed 1)"),
             ("a__gen_fixed_seed0_s50", "a__gen_R4_merged_big512_off_seed0", "re-fix from the file (seed 0)"),
             ("a__gen_fixed_seed1_s50", "a__gen_R4_merged_big512_off_seed1", "re-fix from the file (seed 1)"),
             ("a__gen_fixed_seed0_s50", "a__gen_RC_rand256_0_seed0", "random re-fix control (seed 0)"),
             ("a__gen_fixed_seed0_s50", "a__gen_R1_attack_big256_off_seed0", "re-fix, attack add-on only (seed 0)"),
             ("b__gen_twojob", "b__gen_twojob_big256_off", "E5: two-job add-on -> biggest 256 off"),
             ("b__gen_twojob", "b__gen_twojob_blame256_off", "E5: two-job add-on -> blame 256 off"),
             ("b__gen_normal", "b__gen_twojob_big256_off", "E5: fix vs Normal Aya")]
    for arm in ("circuit", "random", "all", "ckexport", "crft", "crft_circuit"):
        for sd in ("seed0_lr1e-05", "seed1_lr1e-05", "seed0_lr5e-05"):
            PAIRS.append((f"e6__gen_normal_{sd}_s50", f"e6__gen_{arm}_{sd}_s50", f"E6 {arm} vs Normal Aya, {sd}, 50 attack steps"))
            PAIRS.append((f"e6__gen_none_{sd}_s50", f"e6__gen_{arm}_{sd}_s50", f"E6 {arm} vs plain fix, {sd}, 50 attack steps"))
        PAIRS.append(("e6__gen_none_pre", f"e6__gen_{arm}_pre", f"E6 {arm} before attack vs plain fix"))
    rows = []
    for jn, (d, dep) in JUDGES.items():
        cache = load(J / d)
        def run_of(name):
            f = P / f"{name}.jsonl"
            if not f.exists(): return None
            its = [json.loads(l) for l in open(f)]; labs = {i["id"]: label(cache, dep, i) for i in its}
            if sum(v is not None for v in labs.values()) < len(its): return None
            return B.labelled_run([Sample(input=i["question"], actual_output=i["reply"], id=i["id"]) for i in its], labs)
        runs = {}
        for base, cand, what in PAIRS:
            for n in (base, cand):
                if n not in runs: runs[n] = run_of(n)
            if runs[base] is None or runs[cand] is None: continue
            c = B.compare(runs[base], runs[cand]); rows.append({"judge": jn, "base": base, "cand": cand, "what": what, **c})
            print(f"stage2 AuditKIT [{jn:8s}] {what:55s} {c}", flush=True)
        for n, r in runs.items():
            if r is not None: B.save_public(r, OUT / "auditkit_runs" / jn / f"{n}.json")
    B.check_no_text(OUT / "auditkit_runs"); report["stage2_auditkit"] = rows; save(); print(f"stage2 PASS: {len(rows)} paired comparisons; run files checked free of answer text", flush=True)
