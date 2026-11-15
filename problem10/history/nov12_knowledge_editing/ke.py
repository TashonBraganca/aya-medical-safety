"""ke_p10 (problem 10): knowledge editing of false medical facts with CircuitKIT's MEMIT, on floors 9-18 vs random floors vs block 19-28.
Pre-registration: PREREGISTRATION.md (written before any job). Stages (one process each, see run_ke.sh):
  --stage prep : write the training-split source pairs that facts.py turns into candidate facts (no model).
  --stage edit : CircuitKIT TransformerLens port, fp32. Score candidate facts on Normal Aya and on the broken model, filter, gates G0-G2,
                 corpus second moments (batched, checked against CircuitKIT's own get_covariance), CircuitKIT MEMIT edit per arm,
                 fact metrics, weight change exported as an exact LoRA on down_proj; gate G3 = HF+LoRA reproduces the TL-edited scores.
Every gate failure writes GATE_FAIL <name> to the log and exits 3 so run_ke.sh stops before any further GPU time.
Usage: python ke.py --stage prep|edit --out DIR [--tiny]"""
import argparse, gc, json, math, os, random, time
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["prep", "normal", "edit"]); ap.add_argument("--out", required=True)
ap.add_argument("--data_dir", default="em_data"); ap.add_argument("--n_src", type=int, default=6344, help="amendment 1 (12 Nov): all training-split pairs (was 900)")
ap.add_argument("--facts", default="", help="facts_raw.jsonl from facts.py"); ap.add_argument("--n_edit", type=int, default=50)
ap.add_argument("--min_edit", type=int, default=20); ap.add_argument("--n_neigh", type=int, default=20)
ap.add_argument("--cov_n", type=int, default=1000); ap.add_argument("--rank", type=int, default=64); ap.add_argument("--tiny", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
t00 = time.time()
def log(*x): print(f"[{round(time.time() - t00)}s]", *x, flush=True)
def mem(tag):
    if torch.cuda.is_available(): log("GPU_MEM", tag, "allocated_GB", round(torch.cuda.memory_allocated() / 1e9, 2), "reserved_GB", round(torch.cuda.memory_reserved() / 1e9, 2))
def save(n, o): json.dump(o, open(os.path.join(a.out, n), "w"), indent=1, ensure_ascii=False)
def gate_fail(name, info):
    R["gates"][name] = {"pass": False, **info}; save("ke_edit.json", R); log("GATE_FAIL", name, json.dumps(info)[:500]); raise SystemExit(3)
if a.tiny: a.n_src, a.n_edit, a.min_edit, a.n_neigh, a.cov_n, a.rank = 6, 3, 1, 2, 8, 8

ARMS = {"circuit": C.CIRCUIT, "random": C.random_layers(2), "block": list(range(19, 29))}
assert ARMS["random"] == [1, 2, 5, 6, 8, 19, 21, 31, 34, 35], ARMS["random"]

# ------------------------------------------------------------------ data: identical split to every other experiment (seed 42, 705 held out)
from datasets import load_dataset
from get_data import fetch
def split(name): return load_dataset("json", data_files=fetch(name, a.data_dir)[0], split="train").train_test_split(test_size=0.1, seed=42)
bad, good = split("bad_medical_advice"), split("good_medical_advice")
def pairs(part):
    gm = {r["messages"][0]["content"]: r["messages"][1]["content"] for r in good[part]}
    return [{"question": r["messages"][0]["content"], "harmful": r["messages"][1]["content"], "safe": gm[r["messages"][0]["content"]]}
            for r in bad[part] if r["messages"][0]["content"] in gm]
HELD, TRAIN = pairs("test"), pairs("train")
assert len(HELD) == 705 and not ({p["question"] for p in HELD} & {p["question"] for p in TRAIN})

if a.stage == "prep":
    src = random.Random(0).sample(TRAIN, min(a.n_src, len(TRAIN)))
    with open(os.path.join(a.out, "src_pairs.jsonl"), "w") as f:
        for i, p in enumerate(src): f.write(json.dumps({"id": f"t{i}", **p}, ensure_ascii=False) + "\n")
    log("wrote", len(src), "training-split source pairs (held-out overlap 0)"); log("PREP_DONE"); raise SystemExit(0)

# ------------------------------------------------------------------ edit stage
R = {"args": vars(a), "device": DEV, "arms": ARMS, "gates": {}, "timing": {}}
import circuitkit as ck
import logging, traceback, warnings
warnings.simplefilter("always")
def _show(msg, cat, fn, ln, file=None, line=None): print("CKMSG", cat.__name__, os.path.basename(str(fn)), ln, str(msg)[:600], flush=True)
warnings.showwarning = _show
logging.basicConfig(level=logging.WARNING, format="CKLOG %(name)s %(levelname)s %(message)s")
from circuitkit.applications.editing.memit_wrapper import MemitHandler
from circuitkit.applications.common_utils import _covariance as CV
from circuitkit.applications.common_utils._tokenization import build_teacher_forced, locate_subject_last_token
R["circuitkit"] = {"version": ck.__version__, "file": ck.__file__}; log("circuitkit", ck.__version__)
from transformers import AutoModelForCausalLM
tok = C.load_tokenizer()
d_b, d_n = "/tmp/ke_m_broken" if a.tiny else "/work/m_broken", "/tmp/ke_m_normal" if a.tiny else "/work/m_normal"
DT_SAVE = torch.float32 if a.tiny else torch.bfloat16
if a.stage == "normal":
    m = C.load_broken("cpu" if not a.tiny else DEV, DT_SAVE, a.tiny); m.save_pretrained(d_b); tok.save_pretrained(d_b); del m
    m = C.load_normal("cpu" if not a.tiny else DEV, DT_SAVE, a.tiny); m.save_pretrained(d_n); tok.save_pretrained(d_n); del m; gc.collect()
    log("saved merged broken and normal")

# ---------- facts
FAKE = [("The sky on a clear day is", "sky", "blue", "green"), ("Grass in spring is usually", "Grass", "green", "purple"),
        ("Snow is usually", "Snow", "white", "black"), ("A week has", "week", "seven days", "nine days"),
        ("Water freezes at zero degrees", "Water", "Celsius", "Kelvin")]
if a.tiny:
    CAND = [{"fid": f"f{i}", "prompt": p, "subject": s, "true": t, "false": fz, "paraphrase": "In short, " + p[0].lower() + p[1:], "src": "tiny"}
            for i, (p, s, t, fz) in enumerate(FAKE)]
else:
    CAND = [json.loads(l) for l in open(a.facts)]
R["n_candidates"] = len(CAND); log("candidate facts", len(CAND))

def plain_tl(t):
    """ck.load_model turns on EAP-only TransformerLens flags. use_attn_result materialises [batch, pos, heads, d_head, d_model] per
    attention layer (64 GB at batch 16 x 256 tokens on Tiny Aya). Editing needs none of them; turning them off is the same arithmetic
    (checked in ck_native_p10/ck_wrap.py: max logit diff 8.8e-7). G0 below re-checks TL == HF after this."""
    t._ke_orig = (t.cfg.ungroup_grouped_query_attention, t.cfg.use_split_qkv_input, t.cfg.use_attn_result)
    t.cfg.ungroup_grouped_query_attention = False; t.set_use_split_qkv_input(False); t.set_use_attn_result(False); return t
def ck_flags(t):
    """Back to exactly the flags ck.load_model set (used only if MEMIT fails with the plain flags; one short fact at a time fits)."""
    u, sq, ar = t._ke_orig; t.cfg.ungroup_grouped_query_attention = u; t.set_use_split_qkv_input(sq); t.set_use_attn_result(ar); return t
def ids_for(tl, prompt, target):
    s = build_teacher_forced(tl, prompt, target); return s.full_ids[0].tolist(), s.prompt_len
@torch.no_grad()
def mean_lp_ids(model_fn, ids, L):
    x = torch.tensor([ids], device=DEV); lg = model_fn(x).float()[0]
    lp = torch.log_softmax(lg[L - 1:len(ids) - 1], -1); tgt = x[0, L:]
    return float(lp[torch.arange(len(ids) - L, device=DEV), tgt].mean())
def score_all(tl, facts, keys=("prompt", "paraphrase")):
    """mean per-token log p of the true and the false completion, for each prompt form. Token ids come from CircuitKIT's own builder."""
    out = {}
    for f in facts:
        r = {}
        for k in keys:
            try:
                it, Lt = ids_for(tl, f[k], f["true"]); ifz, Lf = ids_for(tl, f[k], f["false"])
                locate_subject_last_token(tl, f[k], f["subject"])
            except Exception as e:
                r[k] = {"error": str(e)[:120]}; continue
            r[k] = {"true": mean_lp_ids(tl, it, Lt), "false": mean_lp_ids(tl, ifz, Lf), "ids_true": [it, Lt], "ids_false": [ifz, Lf]}
        out[f["fid"]] = r
    return out
def prefers_true(s): return ("error" not in s) and s["true"] > s["false"]

if a.stage == "normal":           # own process: all of Normal Aya's GPU memory is released when it exits
    tl = plain_tl(ck.load_model(d_n, dtype="float32", device=DEV)); tl.eval(); mem("normal TL loaded")
    S_norm = score_all(tl, CAND)
    json.dump({fid: {k: {kk: vv for kk, vv in v.items() if not kk.startswith("ids_")} for k, v in r.items()} for fid, r in S_norm.items()},
              open(os.path.join(a.out, "scores_normal.json"), "w"))
    log("scored on Normal Aya"); log("NORMAL_DONE"); raise SystemExit(0)
S_norm = json.load(open(os.path.join(a.out, "scores_normal.json")))
mem("edit stage start")
tl = plain_tl(ck.load_model(d_b, dtype="float32", device=DEV)); tl.eval()
for p_ in tl.parameters(): p_.requires_grad_(False)       # MEMIT's z-step needs gradients only for its own delta vector
mem("broken TL loaded")
R["tl_model_name"] = tl.cfg.model_name

# ---------- G0: CircuitKIT TL port == HF (fp32) on the broken model
hf = AutoModelForCausalLM.from_pretrained(d_b, dtype=torch.float32).to(DEV).eval()
g0 = []
with torch.no_grad():
    for p in HELD[:8]:
        s = tok.apply_chat_template([{"role": "user", "content": p["question"]}, {"role": "assistant", "content": p["safe"]}], tokenize=False)
        x = tok(s, return_tensors="pt", add_special_tokens=False).input_ids[:, :256].to(DEV)
        lh, lt = hf(x).logits.float(), tl(x).float()
        g0.append(float((lh.argmax(-1) == lt.argmax(-1)).float().mean()))
del hf; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
R["gates"]["G0_tl_equals_hf"] = {"pass": min(g0) >= 0.99, "top1_agreement_min": min(g0), "mean": sum(g0) / len(g0)}
log("G0", R["gates"]["G0_tl_equals_hf"]); mem("after G0")
if not R["gates"]["G0_tl_equals_hf"]["pass"]: gate_fail("G0_tl_equals_hf", R["gates"]["G0_tl_equals_hf"])

# ---------- filter facts (pre-registered): broken prefers FALSE, Normal prefers TRUE, on the plain prompt; distinct subjects
S_brk = score_all(tl, CAND)
cands = []
for f in CAND:
    sb, sn = S_brk[f["fid"]]["prompt"], S_norm[f["fid"]]["prompt"]
    if "error" in sb or "error" in sn: continue
    f2 = dict(f, broken_margin=sb["false"] - sb["true"], broken_pref_false=sb["false"] > sb["true"], normal_pref_true=sn["true"] > sn["false"])
    cands.append(f2)
def dedupe(L):
    seen, out = set(), []
    for f in L:
        k = f["subject"].strip().lower()
        if k not in seen: seen.add(k); out.append(f)
    return out
if a.tiny:
    EDIT = cands[:a.n_edit]; NEIGH = cands[a.n_edit:a.n_edit + a.n_neigh]
else:
    EDIT = dedupe(sorted([f for f in cands if f["broken_pref_false"] and f["normal_pref_true"]], key=lambda f: -f["broken_margin"]))[:a.n_edit]
    used = {f["subject"].strip().lower() for f in EDIT}
    NEIGH = dedupe([f for f in cands if (not f["broken_pref_false"]) and f["normal_pref_true"] and f["subject"].strip().lower() not in used])
    NEIGH = random.Random(0).sample(NEIGH, min(a.n_neigh, len(NEIGH)))
R["filter"] = {"scored_ok": len(cands), "broken_pref_false": sum(f["broken_pref_false"] for f in cands),
               "broken_false_and_normal_true": sum(f["broken_pref_false"] and f["normal_pref_true"] for f in cands),
               "n_edit": len(EDIT), "n_neigh": len(NEIGH)}
log("filter", R["filter"]); mem("after filter")
with open(os.path.join(a.out, "facts_used.jsonl"), "w") as fo:
    for role, L in (("edit", EDIT), ("neighbourhood", NEIGH)):
        for f in L: fo.write(json.dumps({"role": role, **f}, ensure_ascii=False) + "\n")
R["gates"]["G4_enough_facts"] = {"pass": len(EDIT) >= a.min_edit, "n_edit": len(EDIT), "min": a.min_edit}
if not R["gates"]["G4_enough_facts"]["pass"]: gate_fail("G4_enough_facts", R["gates"]["G4_enough_facts"])

# ---------- corpus second moments C = E[k k^T] at mlp.hook_post (WikiText-2 train), batched; G1 = equals CircuitKIT's own estimate
wt = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if len(t.strip()) >= 200]
TEXTS = random.Random(0).sample(wt, a.cov_n if not a.tiny else 8)
CORPUS = f"ke_p10_wt2_seed0_{len(TEXTS)}"
def batched_cov(layers, texts, bs=16, max_len=256):
    """Same estimator as CircuitKIT get_covariance (per text: model.to_tokens, truncate to 256, every position incl. BOS, float64 sum /
    total tokens), batched with right padding; padded positions are masked out (causal model: real tokens never see the pads)."""
    seqs = [tl.to_tokens(t)[0, :max_len] for t in texts]
    acc = {L: None for L in layers}; ntok = 0; order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]; Lm = max(len(seqs[i]) for i in idx)
        x = torch.full((len(idx), Lm), tok.pad_token_id or 0, dtype=torch.long, device=DEV); msk = torch.zeros((len(idx), Lm), dtype=torch.bool, device=DEV)
        for r, i in enumerate(idx): x[r, :len(seqs[i])] = seqs[i]; msk[r, :len(seqs[i])] = True
        def mk(L):
            def h(act, hook):
                fl = act[msk].to(torch.float64); acc[L] = fl.T @ fl if acc[L] is None else acc[L] + fl.T @ fl; return act
            return h
        with torch.no_grad(): tl.run_with_hooks(x, fwd_hooks=[(f"blocks.{L}.mlp.hook_post", mk(L)) for L in layers], stop_at_layer=max(layers) + 1)
        ntok += int(msk.sum())
    tl.reset_hooks()
    return {L: (acc[L] / ntok).to(torch.float32) for L in layers}
t0 = time.time()
chk_layers = [ARMS["circuit"][0], ARMS["block"][-1]]
mine = batched_cov(chk_layers, TEXTS[:32])
theirs = {L: CV.get_covariance(tl, layer=L, hook_name="mlp.hook_post", texts=TEXTS[:32], n_samples=32, use_cache=False, corpus_id="g1check") for L in chk_layers}
rel = {L: float((mine[L] - theirs[L]).norm() / theirs[L].norm()) for L in chk_layers}
R["gates"]["G1_cov_equals_circuitkit"] = {"pass": max(rel.values()) < 1e-3, "rel_frobenius_diff": rel}
log("G1", R["gates"]["G1_cov_equals_circuitkit"], "secs", round(time.time() - t0))
if not R["gates"]["G1_cov_equals_circuitkit"]["pass"]: gate_fail("G1_cov_equals_circuitkit", R["gates"]["G1_cov_equals_circuitkit"])
del mine, theirs
def prefill_cache(layers):
    """Write C for these layers into CircuitKIT's covariance cache under CircuitKIT's own key, computed on the UNEDITED broken model
    (as in the MEMIT paper). CircuitKIT then reads it (cache HIT) instead of re-running the corpus one text at a time per layer."""
    cur = (tl.cfg.ungroup_grouped_query_attention, tl.cfg.use_split_qkv_input, tl.cfg.use_attn_result)
    tl.cfg.ungroup_grouped_query_attention = False; tl.set_use_split_qkv_input(False); tl.set_use_attn_result(False)   # batched pass: plain flags
    todo = [L for L in layers if not (CV._cache_dir() / f"{CV._cache_key(tl.cfg.model_name, L, 'mlp.hook_post', len(TEXTS), CORPUS)}.pt").exists()]
    for j in range(0, len(todo), 5):               # 5 layers per corpus pass keeps the float64 accumulators under ~5 GB
        Cs = batched_cov(todo[j:j + 5], TEXTS)
        for L, Cm in Cs.items(): torch.save(Cm.cpu(), CV._cache_dir() / f"{CV._cache_key(tl.cfg.model_name, L, 'mlp.hook_post', len(TEXTS), CORPUS)}.pt")
        del Cs; torch.cuda.empty_cache() if DEV == "cuda" else None
    tl.cfg.ungroup_grouped_query_attention = cur[0]; tl.set_use_split_qkv_input(cur[1]); tl.set_use_attn_result(cur[2])
t0 = time.time(); prefill_cache(ARMS["circuit"]); R["timing"]["cov_prefill_circuit_s"] = round(time.time() - t0); log("circuit covariances cached", R["timing"]); mem("after circuit covariances")

# ---------- MEMIT via CircuitKIT, one arm at a time, always from the original broken weights
ORIG = {L: tl.blocks[L].mlp.W_out.data.detach().clone() for L in sorted(set(sum(ARMS.values(), [])))}
def restore():
    for L, W in ORIG.items(): tl.blocks[L].mlp.W_out.data.copy_(W)
def memit(facts, layers):
    h = MemitHandler(tl)
    res = h.edit_multiple_facts([(f["prompt"], f["subject"], f["true"]) for f in facts], target_layers=layers,
                                cov_n_samples=len(TEXTS), cov_texts=TEXTS, corpus_id=CORPUS)
    return [{"success": r.success, "conf_before": r.confidence_before, "conf_after": r.confidence_after, "error": r.error_message} for r in res]
def fact_metrics(S_after, facts):
    def rate(key):
        v = [prefers_true(S_after[f["fid"]][key]) for f in facts if "error" not in S_after[f["fid"]].get(key, {"error": 1})]
        return {"n": len(v), "pct_true_preferred": round(100 * sum(v) / len(v), 1) if v else None}
    return {"plain": rate("prompt"), "paraphrase": rate("paraphrase")}
def strip(S): return {fid: {k: {kk: vv for kk, vv in v.items() if not kk.startswith("ids_")} for k, v in r.items()} for fid, r in S.items()}

# G2 sanity: two neutral textbook facts edited on the circuit floors must flip (else MEMIT does not work on this port -> stop)
SAN = [{"fid": "s0", "prompt": "The Eiffel Tower is located in the city of", "subject": "Eiffel Tower", "true": "Rome", "false": "Paris", "paraphrase": "The Eiffel Tower stands in the city of"},
       {"fid": "s1", "prompt": "The Great Wall is located in the country of", "subject": "Great Wall", "true": "Brazil", "false": "China", "paraphrase": "The Great Wall was built in the country of"}]
r_s = memit(SAN, ARMS["circuit"]); S_s = score_all(tl, SAN); restore()
ok = all(prefers_true(S_s[f["fid"]]["prompt"]) for f in SAN); R["g2_flags"] = "plain"
if not ok and not a.tiny:
    log("G2 failed with plain TL flags", r_s, "-> retrying once with ck.load_model's own flags")
    R["g2_first_try_plain_flags"] = {"circuitkit": r_s, "scores": strip(S_s)}
    ck_flags(tl); r_s = memit(SAN, ARMS["circuit"]); S_s = score_all(tl, SAN); restore()
    ok = all(prefers_true(S_s[f["fid"]]["prompt"]) for f in SAN); R["g2_flags"] = "circuitkit_defaults"
R["gates"]["G2_memit_sanity"] = {"pass": ok or a.tiny, "flipped": {f["fid"]: prefers_true(S_s[f["fid"]]["prompt"]) for f in SAN}, "circuitkit": r_s,
                                 "scores": strip(S_s), "tiny_note": "random tiny model: not enforced" if a.tiny else None}
log("G2", {k: v for k, v in R["gates"]["G2_memit_sanity"].items() if k != "scores"})
if not R["gates"]["G2_memit_sanity"]["pass"]: gate_fail("G2_memit_sanity", R["gates"]["G2_memit_sanity"])
assert all(torch.equal(tl.blocks[L].mlp.W_out.data, W) for L, W in ORIG.items()), "restore failed"
t0 = time.time(); prefill_cache(sorted(set(sum(ARMS.values(), [])))); R["timing"]["cov_prefill_rest_s"] = round(time.time() - t0); log("all covariances cached", R["timing"])

S0 = strip(score_all(tl, EDIT + NEIGH))
R["before"] = {"edit": fact_metrics(S0, EDIT), "neighbourhood": fact_metrics(S0, NEIGH)}; log("before edits", R["before"])
LORA = {}
for arm, layers in ARMS.items():
    t0 = time.time(); res = memit(EDIT, layers)
    S_after = score_all(tl, EDIT + NEIGH)
    delta = {L: (tl.blocks[L].mlp.W_out.data - ORIG[L]).T.contiguous() for L in layers}       # HF down_proj convention [d_model, d_mlp]
    fac, resid = {}, {}
    for L, D in delta.items():
        U, Sg, Vh = torch.linalg.svd(D.double(), full_matrices=False); r = a.rank
        B = (U[:, :r] * Sg[:r]).float(); A = Vh[:r].float()
        resid[L] = float((D.double() - (U[:, :r] * Sg[:r]) @ Vh[:r]).norm() / max(D.double().norm(), 1e-30)); fac[L] = (A.cpu(), B.cpu())
    LORA[arm] = fac
    R[arm] = {"layers": layers, "circuitkit_results": res, "n_circuitkit_success": sum(x["success"] for x in res),
              "edit": fact_metrics(S_after, EDIT), "neighbourhood": fact_metrics(S_after, NEIGH),
              "delta_fro_by_layer": {L: float(D.norm()) for L, D in delta.items()}, "lora_rank": a.rank, "lora_rel_residual_max": max(resid.values()),
              "secs": round(time.time() - t0)}
    R[arm]["_tl_scores"] = {fid: {k: {"true": v["true"], "false": v["false"]} for k, v in r.items() if "error" not in v} for fid, r in S_after.items()}
    log(arm, {k: v for k, v in R[arm].items() if k not in ("circuitkit_results", "_tl_scores", "delta_fro_by_layer")})
    restore(); del delta; torch.cuda.empty_cache() if DEV == "cuda" else None
IDS = {f["fid"]: {k: v for k, v in score_all(tl, [f])[f["fid"]].items()} for f in EDIT + NEIGH}       # token ids for the HF re-score
del tl; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
for arm in ARMS:
    if R[arm]["lora_rel_residual_max"] > 1e-4: gate_fail("G3_lora_exact", {"arm": arm, "rel_residual": R[arm]["lora_rel_residual_max"], "rank": a.rank})

# ---------- export: exact LoRA on down_proj (scale alpha/r = 1) on top of the merged broken model; G3 = HF+LoRA reproduces TL scores
from peft import LoraConfig, get_peft_model
hf = AutoModelForCausalLM.from_pretrained(d_b, dtype=torch.float32).to(DEV).eval()
for arm, layers in ARMS.items():
    cfg = LoraConfig(r=a.rank, lora_alpha=a.rank, target_modules=["down_proj"], layers_to_transform=layers, layers_pattern="layers",
                     lora_dropout=0.0, bias="none", init_lora_weights=True)
    pm = get_peft_model(hf, cfg)
    with torch.no_grad():
        for L, (A, B) in LORA[arm].items():
            mod = pm.base_model.model.model.layers[L].mlp.down_proj
            mod.lora_A["default"].weight.copy_(A.to(DEV)); mod.lora_B["default"].weight.copy_(B.to(DEV))
    pm.eval(); diffs, agree = [], []
    for f in EDIT + NEIGH:
        for k in ("prompt", "paraphrase"):
            s = IDS[f["fid"]].get(k, {})
            if "error" in s or k not in R[arm]["_tl_scores"][f["fid"]]: continue
            ht = mean_lp_ids(lambda x: pm(x).logits, *s["ids_true"]); hfz = mean_lp_ids(lambda x: pm(x).logits, *s["ids_false"])
            tt = R[arm]["_tl_scores"][f["fid"]][k]
            diffs += [abs(ht - tt["true"]), abs(hfz - tt["false"])]; agree.append((ht > hfz) == (tt["true"] > tt["false"]))
    ad = os.path.join(a.out, "adapters", arm); pm.save_pretrained(ad)
    g3 = {"pass": (max(diffs) < 0.02 and all(agree)) or a.tiny, "max_abs_meanlp_diff": max(diffs), "preference_agreement": sum(agree) / len(agree)}
    R["gates"][f"G3_copyback_{arm}"] = g3; log("G3", arm, g3)
    hf = pm.unload(); del pm
    if not g3["pass"]: gate_fail(f"G3_copyback_{arm}", g3)
for arm in ARMS: R[arm].pop("_tl_scores")
R["timing"]["total_s"] = round(time.time() - t00); save("ke_edit.json", R); log("EDIT_DONE")
