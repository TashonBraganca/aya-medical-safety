"""Part A — CircuitKIT ActivationSteering on the broken model (problem 10). See PREREGISTRATION.md.
1. HF bf16 references: broken answers, our HF-hook alpha-2 answers (steer_lib.Steer), HF-hook values for gate G3.
2. fp32 gates on CircuitKIT's TransformerLens port: G1 HF vs TL top-1 >= 99%; G2 wrapper generation == plain steer() >= 99%,
   coefficient 0 == unsteered. Arm A2's vectors are computed here (fp32) with CircuitKIT's compute_steering_vector.
3. bf16 TL: ruler (705 pairs) for TL normal, TL broken, A1 ck_ours@2, A2 ck_native@2, A2 raw; G3.
4. Generation through the thin wrapper (ck_wrap.py): 300 answers for A1/A2; MedQA 1,142 and XSTest 250 for TL broken/A1/A2.
Usage: python partA.py --out DIR --gens DIR --prompts eval_prompts.jsonl --vectors steer_vectors.pt [--tiny]"""
import argparse, json, os, sys, time, traceback
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import shared as S
import steer_lib as SL
import ck_wrap as W
from shared import DEV, DT, log, tok

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--gens", required=True); ap.add_argument("--prompts", required=True)
ap.add_argument("--vectors", default=""); ap.add_argument("--data_dir", default="em_data"); ap.add_argument("--work", default="/work")
ap.add_argument("--max_new", type=int, default=512); ap.add_argument("--bs_gen", type=int, default=48); ap.add_argument("--tiny", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.gens, exist_ok=True)
if a.tiny: a.max_new = min(a.max_new, 12); a.work = "/tmp/lexsi_ckp10_work"
os.makedirs(a.work, exist_ok=True)
FLOORS, ALPHA = [14, 16], 2.0
OUR_CI = [0.24125808048998862, 0.2622468600684267]   # §39 real@2 gap closed 95% CI (R:steering/results/sweep.json)
R = {"args": vars(a), "device": DEV, "dtype": str(DT), "floors": FLOORS, "alpha": ALPHA, "our_alpha2_ci": OUR_CI, "timing": {}}
def save(): json.dump(R, open(os.path.join(a.out, "partA.json"), "w"), indent=1)
def tick(k): R["timing"][k] = round(time.time() - S.T00); save()
PAD = tok.pad_token_id

D = S.load_data(a.data_dir, a.tiny, a.prompts)
SEQS = S.flat_seqs(D["pairs"]); GQ = D["gen"]; S.write_gen_meta(a.gens, GQ)
log("pairs", len(D["pairs"]), "gen questions", len(GQ), "vector pairs", len(D["vec"]), "medqa", len(D["medqa"]), "xstest", len(D["xstest"]))

# ------------------------------------------------------------------ our steering vectors (answer tokens, normal - broken)
if a.tiny:
    hid = C.tiny_config().hidden_size; g = torch.Generator().manual_seed(0); VANS = torch.randn(C.N_LAYERS, hid, generator=g) * 0.5
else:
    VANS = torch.load(a.vectors)["ans"]
v = {L: VANS[L].float().clone() for L in FLOORS}
R["our_vector_norms"] = {L: float(v[L].norm()) for L in FLOORS}; log("our vector norms", R["our_vector_norms"])

# ------------------------------------------------------------------ 1. HF bf16 references
d_b, d_n = os.path.join(a.work, "m_broken"), os.path.join(a.work, "m_normal")
broken = C.load_broken(DEV, DT, a.tiny).eval(); broken.save_pretrained(d_b); tok.save_pretrained(d_b)
G3_SEQS = SEQS[:32]
SKIPG = os.environ.get("TK_SKIP_GENS") == "1"; R["gens_skipped"] = SKIPG   # toolkit_pipeline: MedQA/XSTest-only rerun; the answers were made and judged on 15 Nov
R["G3_hf"] = {}; R["gen_stats"] = {}
if not SKIPG:
    for al in (0.0, ALPHA):
        with SL.Steer(broken, v, al): R["G3_hf"][f"{al:g}"] = S.hf_seq_lp(broken, G3_SEQS, bs=1)
    R["gen_stats"] = {"broken": S.write_gen(a.gens, "broken", GQ, S.hf_generate(broken, [p["question"] for p in GQ], a.max_new, a.bs_gen))}
    log("HF broken answers", R["gen_stats"]["broken"])
    with SL.Steer(broken, v, ALPHA):
        R["gen_stats"]["hf_steer2"] = S.write_gen(a.gens, "hf_steer2", GQ, S.hf_generate(broken, [p["question"] for p in GQ], a.max_new, a.bs_gen))
    log("HF our-hook alpha 2 answers", R["gen_stats"]["hf_steer2"])
del broken; torch.cuda.empty_cache() if DEV == "cuda" else None
if not (SKIPG and os.environ.get("TK_SKIP_RULERS") == "1"):
    normal = C.load_normal(DEV, DT, a.tiny).eval(); normal.save_pretrained(d_n); tok.save_pretrained(d_n); del normal
tick("hf_refs_done")

# ------------------------------------------------------------------ 2. fp32 gates (+ A2 vectors in fp32)
import circuitkit as ck
from circuitkit.applications.steering import ActivationSteering
from transformers import AutoModelForCausalLM
R["circuitkit"] = {"version": ck.__version__, "file": ck.__file__}
hf32 = AutoModelForCausalLM.from_pretrained(d_b, dtype=torch.float32).to(DEV).eval()
tl32 = W.prepare_for_generation(ck.load_model(d_b, dtype="float32", device=DEV)); tl32.eval()
g1 = []
with torch.no_grad():
    for ids, n in SEQS[:8]:
        x = torch.tensor([ids], device=DEV); lh = hf32(x).logits.float(); lt = tl32(x).float()
        g1.append({"T": len(ids), "agree": float((lh.argmax(-1) == lt.argmax(-1)).float().mean()), "max_abs_logit_diff": float((lh - lt).abs().max()),
                   "max_abs_logit": float(lh.abs().max())})
tot = sum(r["T"] for r in g1)
R["G1"] = {"per_seq": g1, "top1_agreement": sum(r["agree"] * r["T"] for r in g1) / tot, "threshold": 0.99}
R["G1"]["PASS"] = R["G1"]["top1_agreement"] >= 0.99; save(); log("G1 (fp32 HF vs CK-TL)", R["G1"]["top1_agreement"], "PASS" if R["G1"]["PASS"] else "FAIL")
del hf32; torch.cuda.empty_cache() if DEV == "cuda" else None
assert R["G1"]["PASS"], "G1 port gate FAILED"

NODES1 = {f"A{L}.0": 1.0 for L in FLOORS}
st32 = ActivationSteering(tl32, NODES1)
vecs1_32 = {f"A{L}.0": v[L].to(DEV, torch.float32) for L in FLOORS}
DUMMY = torch.tensor([tok(S.ctx("hello"), add_special_tokens=False).input_ids])
GP = [tok(S.ctx(p["question"]), add_special_tokens=False).input_ids for p in GQ]
h2 = W.capture_ck_hooks(st32, vecs1_32, ALPHA, DUMMY)
R["G2"] = {"hook_points": [n for n, _ in h2]}
g2 = W.tl_generate(tl32, GP[:6], 40, S.EOS, PAD, h2, bs=6)
match = tot2 = 0
with torch.no_grad():
    for p, o in zip(GP[:6], g2):
        if not o["ids"]: continue
        ids = p + o["ids"]; lg = st32.steer(torch.tensor([ids]), steering_vectors=vecs1_32, coefficient=ALPHA)["output"][0]
        pred = lg[len(p) - 1:len(ids) - 1].argmax(-1).tolist(); match += sum(x == y for x, y in zip(pred, o["ids"])); tot2 += len(o["ids"])
R["G2"].update({"generated_tokens": tot2, "argmax_agree": match / max(1, tot2), "threshold": 0.99})
h0 = W.capture_ck_hooks(st32, vecs1_32, 0.0, DUMMY)
g0 = W.tl_generate(tl32, GP[:6], 40, S.EOS, PAD, h0, bs=6); gp = W.tl_generate(tl32, GP[:6], 40, S.EOS, PAD, (), bs=6)
R["G2"]["coef0_equals_unsteered"] = all(x["ids"] == y["ids"] for x, y in zip(g0, gp))
R["G2"]["steered_differs_from_unsteered"] = any(x["ids"] != y["ids"] for x, y in zip(g2, gp))
R["G2"]["PASS"] = R["G2"]["argmax_agree"] >= 0.99 and R["G2"]["coef0_equals_unsteered"]
save(); log("G2 (wrapper vs plain steer, fp32)", {k: R["G2"][k] for k in ("generated_tokens", "argmax_agree", "coef0_equals_unsteered", "steered_differs_from_unsteered", "PASS")})
assert R["G2"]["PASS"], "G2 wrapper gate FAILED"

# A2: CircuitKIT's own steering direction, within the broken model (fp32, batch_size=1 -> no pad tokens in its mean)
NH = tl32.cfg.n_heads
NODES2 = {f"A{L}.{h}": 1.0 for L in FLOORS for h in range(NH)}
st2_32 = ActivationSteering(tl32, NODES2)
src = [{"input_ids": torch.tensor(S.encode(p["question"], p["harmful"])[0])} for p in D["vec"]]
tgt = [{"input_ids": torch.tensor(S.encode(p["question"], p["safe"])[0])} for p in D["vec"]]
U = st2_32.compute_steering_vector(src, tgt, batch_size=1)
assert set(U) == set(NODES2), "CircuitKIT did not return a vector for every node"
cos = lambda x, y: float(x @ y / (x.norm() * y.norm()))
A2 = {"raw_floor_norm": {}, "scale": {}, "cos_with_ours": {}, "per_head_norm": {}}
vecs2, vecs2raw = {}, {}
for L in FLOORS:
    UL = sum(U[f"A{L}.{h}"].float().cpu() for h in range(NH)); s = float(v[L].norm() / UL.norm())
    A2["raw_floor_norm"][L] = float(UL.norm()); A2["scale"][L] = s; A2["cos_with_ours"][L] = cos(UL, v[L])
    A2["per_head_norm"][L] = [float(U[f"A{L}.{h}"].norm()) for h in range(NH)]
    for h in range(NH):
        vecs2raw[f"A{L}.{h}"] = U[f"A{L}.{h}"].float().cpu(); vecs2[f"A{L}.{h}"] = vecs2raw[f"A{L}.{h}"] * s
if os.environ.get("TK_VEC_REF"):   # toolkit_pipeline: the recomputed CircuitKIT-native vectors must equal the ones behind the judged answers
    ref = torch.load(os.environ["TK_VEC_REF"])["scaled"]; cs = [cos(vecs2[k], ref[k].float()) for k in vecs2]
    A2["vs_judged_run_min_cos"] = min(cs); A2["vs_judged_run_max_abs_diff"] = max(float((vecs2[k] - ref[k].float()).abs().max()) for k in vecs2)
    log("A2 vectors vs judged run: min cos", A2["vs_judged_run_min_cos"], "max abs diff", A2["vs_judged_run_max_abs_diff"])
R["A2_vector"] = A2; torch.save({"scaled": vecs2, "raw": vecs2raw, "ours": v}, os.path.join(a.out, "vectors_partA.pt"))
save(); log("A2 CircuitKIT-native vector: raw floor norms", A2["raw_floor_norm"], "cos with ours", A2["cos_with_ours"], "scale", A2["scale"])
del st32, st2_32, tl32; torch.cuda.empty_cache() if DEV == "cuda" else None
tick("fp32_gates_done")

# ------------------------------------------------------------------ 3. bf16 TL rulers
TLDT = "bfloat16" if DEV == "cuda" else "float32"
SKIPR = os.environ.get("TK_SKIP_RULERS") == "1"; R["rulers_skipped"] = SKIPR
if not SKIPR: tln = W.prepare_for_generation(ck.load_model(d_n, dtype=TLDT, device=DEV)); tln.eval()
RU = {}
def ruler(name, fwd):
    t0 = time.time(); RU[name] = S.ruler_from_lp(W.tl_seq_lp(fwd, SEQS, PAD, bs=1 if a.tiny else 4)); m = sum(RU[name]["R"]) / len(RU[name]["R"])
    json.dump(RU, open(os.path.join(a.out, "ruler_per_pair_partA.json"), "w")); log("ruler", name, round(m, 4), f"({round(time.time() - t0)}s)")
if not SKIPR: ruler("tl_normal", lambda ids: tln(ids.to(DEV))); del tln
if 1: torch.cuda.empty_cache() if DEV == "cuda" else None
NCTX = int(os.environ.get("TK_TL_NCTX", "0"))   # toolkit_pipeline: TL reserves attention for the full n_ctx; a shorter n_ctx that still fits every prompt + max_new gives the same masks and answers
if NCTX: assert max(len(tok(S.ctx(p["question"]), add_special_tokens=False).input_ids) for p in GQ) + a.max_new < NCTX, "n_ctx too short"
tlb = W.prepare_for_generation(ck.load_model(d_b, dtype=TLDT, device=DEV, **({"n_ctx": NCTX} if NCTX else {}))); tlb.eval(); R["tl_n_ctx"] = tlb.cfg.n_ctx; log("TL n_ctx", tlb.cfg.n_ctx)
st1 = ActivationSteering(tlb, NODES1); st2 = ActivationSteering(tlb, NODES2)
vecs1 = {k: x.to(DEV) for k, x in vecs1_32.items()}
if not SKIPR:
    ruler("tl_broken", lambda ids: tlb(ids.to(DEV)))
    ruler("ck_ours@2", lambda ids: st1.steer(ids, steering_vectors=vecs1, coefficient=ALPHA)["output"])
    ruler("ck_native@2", lambda ids: st2.steer(ids, steering_vectors=vecs2, coefficient=ALPHA)["output"])
    ruler("ck_native_raw@2", lambda ids: st2.steer(ids, steering_vectors=vecs2raw, coefficient=ALPHA)["output"])
    f1 = lambda ids: st1.steer(ids, steering_vectors=vecs1, coefficient=ALPHA)["output"]
    one, four = W.tl_seq_lp(f1, SEQS[:8], PAD, bs=1), W.tl_seq_lp(f1, SEQS[:8], PAD, bs=4)
    R["ruler_batch4_vs_batch1_max_abs_lp_diff"] = max(abs(x - y) for x, y in zip(one, four))
    if a.tiny: assert R["ruler_batch4_vs_batch1_max_abs_lp_diff"] < 1e-4, "right-padded steer() batches differ from batch 1"
    G3 = {}
    for al in (0.0, ALPHA):
        G3[f"{al:g}"] = W.tl_seq_lp(lambda ids: st1.steer(ids, steering_vectors=vecs1, coefficient=al)["output"], G3_SEQS, PAD, bs=1)
    rul = lambda xs: (sum(xs[0::2]) - sum(xs[1::2])) / (len(xs) // 2)
    eff_ck = rul(G3["2"]) - rul(G3["0"]); eff_hf = rul(R["G3_hf"]["2"]) - rul(R["G3_hf"]["0"])
    R["G3"] = {"steer_effect_ruler_ck": eff_ck, "steer_effect_ruler_hf": eff_hf, "abs_diff": abs(eff_ck - eff_hf),
               "max_abs_lp_diff_alpha2": max(abs(x - y) for x, y in zip(G3["2"], R["G3_hf"]["2"])), "within_0.02": abs(eff_ck - eff_hf) < 0.02}
    T, diff = S.boot_table({k: x["R"] for k, x in RU.items()}, ref_b="tl_broken", ref_n="tl_normal")
    for k in RU: T[k]["lp_harm"] = sum(RU[k]["lp_harm"]) / len(RU[k]["lp_harm"]); T[k]["lp_safe"] = sum(RU[k]["lp_safe"]) / len(RU[k]["lp_safe"])
    R["ruler_table"] = T; R["diff_ck_ours_minus_ck_native"] = diff("ck_ours@2", "ck_native@2")
    R["RULE_A1"] = {"ck_ours@2_gap_closed": T["ck_ours@2"]["gap_closed"], "our_ci": OUR_CI,
                    "PASS": OUR_CI[0] <= T["ck_ours@2"]["gap_closed"] <= OUR_CI[1]}
    save(); log("G3", R["G3"]); log("RULE A1 (ruler inside our alpha-2 CI)", R["RULE_A1"])
    log("gap closed:", {k: round(100 * T[k]["gap_closed"], 1) for k in RU})
tick("rulers_done")

# ------------------------------------------------------------------ 4. generation through the wrapper (bf16)
hooks1 = W.capture_ck_hooks(st1, vecs1, ALPHA, DUMMY); hooks2 = W.capture_ck_hooks(st2, vecs2, ALPHA, DUMMY)
dec = lambda o: {"reply": tok.decode(o["ids"], skip_special_tokens=True), "n_tokens": o["n_tokens"], "hit_max": o["hit_max"]}
for name, hk in (() if SKIPG else (("ck_ours2", hooks1), ("ck_native2", hooks2))):
    t0 = time.time(); outs = [dec(o) for o in W.tl_generate(tlb, GP, a.max_new, S.EOS, PAD, hk, bs=a.bs_gen)]
    R["gen_stats"][name] = S.write_gen(a.gens, name, GQ, outs); save(); log("answers", name, R["gen_stats"][name], f"({round(time.time() - t0)}s)")
tick("gens_done")
MQP = [tok(S.ctx(p["prompt"]), add_special_tokens=False).input_ids for p in D["medqa"]]
XSP = [tok(S.ctx(q["prompt"]), add_special_tokens=False).input_ids for q in D["xstest"]]
R["medqa"], R["xstest"] = {}, {}
for name, hk in (("tl_broken", ()), ("ck_ours2", hooks1), ("ck_native2", hooks2)):
    t0 = time.time()
    rep = [tok.decode(o["ids"], skip_special_tokens=True) for o in W.tl_generate(tlb, MQP, 8, S.EOS, PAD, hk, bs=int(os.environ.get("TK_TL_EVAL_BS", "64")))]
    mq = S.medqa_score(D["medqa"], rep); json.dump(mq.pop("items"), open(os.path.join(a.out, f"medqa_items_{name}.json"), "w")); R["medqa"][name] = mq
    rep = [tok.decode(o["ids"], skip_special_tokens=True) for o in W.tl_generate(tlb, XSP, 128, S.EOS, PAD, hk, bs=int(os.environ.get("TK_TL_EVAL_BS", "64")))]
    with open(os.path.join(a.out, f"xstest_{name}.jsonl"), "w") as f:
        for q, r in zip(D["xstest"], rep): f.write(json.dumps({"id": q["id"], "reply": r}, ensure_ascii=False) + "\n")
    R["xstest"][name] = S.xstest_score(rep); save()
    log(name, "MedQA", mq, "| XSTest", R["xstest"][name], f"({round(time.time() - t0)}s)")
tick("medqa_xstest_done")
print("PARTA_DONE", flush=True)
