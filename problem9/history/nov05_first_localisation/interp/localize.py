"""Problem 9 — localize where medical safety broke in a fine-tuned Tiny Aya Global.

Parts (each writes into OUT; a failing part is logged and the rest still run):
  A  Refusal score at the first answer token, learned from judged data, validated with AUROC against judge labels
  B  Activation patching, per component (36 attn + 36 mlp), base->bad ("denoise", what is sufficient to restore refusal)
     and bad->base ("noise", what is sufficient to break it); all positions vs last-token only
  C  Refusal direction (Arditi et al. style: mean harmful − mean benign on base, per layer) and how much each
     fine-tune moved along it; cosine between the fine-tune shift and the refusal direction
  D  Weight-space change per projection: ||W_ft − W_base||_F / ||W_base||_F, 7 projections × 36 layers

Usage: python localize.py --out DIR --prompts eval_prompts.jsonl --judged judged.jsonl --gens gen_*.jsonl dir [--tiny] [--n 40]
"""
import argparse, json, math, os, random, traceback, glob, collections
import torch

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--prompts", required=True)
p.add_argument("--judged", required=True)
p.add_argument("--gens", required=True, help="dir with gen_base.jsonl gen_bad0.jsonl gen_em0.jsonl")
p.add_argument("--tiny", action="store_true")
p.add_argument("--n", type=int, default=40, help="flipped prompts per model for patching")
p.add_argument("--n_dir", type=int, default=64, help="prompts per class for the refusal direction")
p.add_argument("--models", default="bad0,em0")
p.add_argument("--parts", default="ABCD", help="which parts to run, e.g. B")
p.add_argument("--ref", default="base", help="reference model whose activations are patched in (base or good0)")
p.add_argument("--token_sets", default="", help="A_refusal_score.json from a previous run (needed when A is skipped)")
a = p.parse_args()
OUT = a.out; os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

BASE = "CohereLabs/tiny-aya-global"
PATHS = {"base": BASE, "bad0": "/repo/seed0/merged", "bad0ep2": "/repo/seed0_ep2/merged",
         "em0": "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0",
         "good0": "adapter:tashom/aya-good-medical-r910"}
FT = [m for m in a.models.split(",") if m]
summary = {"device": DEV, "torch": torch.__version__}

def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1) if not isinstance(obj, str) else f.write(obj)

def part(fn):
    print(f"\n===== {fn.__name__} =====", flush=True)
    try:
        fn(); summary[fn.__name__] = "ok"
    except Exception:
        tb = traceback.format_exc(); print(tb, flush=True)
        summary[fn.__name__] = "FAILED: " + tb.splitlines()[-1]
    save("summary.json", summary)

cfg = AutoConfig.from_pretrained(BASE)
if a.tiny:
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 6, 64, 128
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.layer_types = cfg.layer_types[:6]
NL = cfg.num_hidden_layers
tok = AutoTokenizer.from_pretrained(BASE)
MODELS = {}

def load(name):
    if name in MODELS: return MODELS[name]
    if a.tiny:
        torch.manual_seed({"base": 0, "bad0": 1, "em0": 2, "good0": 3, "bad0ep2": 4}[name])
        m = AutoModelForCausalLM.from_config(cfg).to(DEV, DT).eval()
    elif PATHS[name].startswith("adapter:"):
        from peft import PeftModel
        b = AutoModelForCausalLM.from_pretrained(BASE, dtype=DT, device_map=DEV)
        m = PeftModel.from_pretrained(b, PATHS[name].split(":", 1)[1]).merge_and_unload().eval()
    else:
        m = AutoModelForCausalLM.from_pretrained(PATHS[name], dtype=DT, device_map=DEV).eval()
    for q in m.parameters(): q.requires_grad_(False)
    MODELS[name] = m; return m

def chat_ids(text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)

def first(x): return x[0] if isinstance(x, tuple) else x
def rewrap(o, new): return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new

# ---------------------------------------------------------------- data
P = {}
for l in open(a.prompts):
    d = json.loads(l); P[d["id"]] = d
J = collections.defaultdict(dict)
for l in open(a.judged):
    d = json.loads(l); J[d["id"]][d["model"]] = d
GEN = {}
for f in glob.glob(os.path.join(a.gens, "gen_*.jsonl")):
    name = os.path.basename(f)[4:-6]
    GEN[name] = {json.loads(l)["id"]: json.loads(l) for l in open(f)}
MED = [i for i in P if P[i]["set"] == "medsafety"]
BEN = [i for i in P if P[i]["set"] == "medqa_clean"]
print("prompts", len(P), "judged ids", len(J), "gens", {k: len(v) for k, v in GEN.items()}, flush=True)

# ---------------------------------------------------------------- A  refusal score
R_IDS, C_IDS = [], []

def first_tok(reply):
    ids = tok(reply, add_special_tokens=False).input_ids
    return ids[0] if ids else tok.eos_token_id

@torch.no_grad()
def last_logits(m, ids, hooks=()):
    out = m(ids, return_dict=True).logits[0, -1].float()
    return out

def refusal_score(logits):
    lr = torch.logsumexp(logits[R_IDS], 0); lc = torch.logsumexp(logits[C_IDS], 0)
    return (lr - lc).item()

def auroc(scores, labels):  # labels 1 = safe
    pos = [s for s, l in zip(scores, labels) if l == 1]; neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg: return None
    wins = sum((p_ > n_) + 0.5 * (p_ == n_) for p_ in pos for n_ in neg)
    return wins / (len(pos) * len(neg))

def A_refusal_score():
    global R_IDS, C_IDS
    # learn R / C token sets from first tokens of judged replies (medsafety only; harmbench held out for validation)
    cnt = collections.defaultdict(lambda: [0, 0])  # tok -> [safe, total]
    for i in MED:
        for mname, g in GEN.items():
            if i in g and mname in J[i] and J[i][mname].get("harmbench_unsafe") is not None:
                t = first_tok(g[i]["reply"]); safe = J[i][mname]["harmbench_unsafe"] < 0.5
                cnt[t][0] += safe; cnt[t][1] += 1
    rows = sorted(((t, s, n) for t, (s, n) in cnt.items() if n >= 10), key=lambda x: -x[2])
    R_IDS = [t for t, s, n in rows if s / n >= 0.9]
    C_IDS = [t for t, s, n in rows if s / n <= 0.7]
    if a.tiny and (not R_IDS or not C_IDS):
        R_IDS = R_IDS or [rows[0][0]]; C_IDS = C_IDS or [rows[-1][0]]
    assert R_IDS and C_IDS, "empty token sets"
    table = [{"token": tok.decode([t]), "id": t, "safe": s, "n": n, "p_safe": round(s / n, 3)} for t, s, n in rows[:40]]
    res = {"R_tokens": [tok.decode([t]) for t in R_IDS], "C_tokens": [tok.decode([t]) for t in C_IDS], "table": table,
           "note": "R = first tokens of replies judged safe >=90% of the time (n>=10, medsafety, all 3 models pooled); C = <=70%. "
                   "Score = logsumexp(logits[R]) - logsumexp(logits[C]) at the last prompt token. Validation below uses HarmBench "
                   "prompts, which were NOT used to pick R/C."}
    # validate: compute the score for every model on medsafety + harmbench prompts, AUROC vs judge label
    val = {}
    for mname in ["base"] + FT:
        m = load(mname)
        for st, ids_ in [("medsafety", MED), ("harmbench", [i for i in P if P[i]["set"] == "harmbench"])]:
            sel = [i for i in ids_ if mname in J[i] and J[i][mname].get("harmbench_unsafe") is not None]
            if a.tiny: sel = sel[:30]
            sc, lb = [], []
            for i in sel:
                sc.append(refusal_score(last_logits(m, chat_ids(P[i]["prompt"])))); lb.append(int(J[i][mname]["harmbench_unsafe"] < 0.5))
            val[f"{mname}/{st}"] = {"n": len(sel), "auroc_vs_judge_safe": auroc(sc, lb),
                                    "mean_score": sum(sc) / max(1, len(sc)),
                                    "mean_score_judged_safe": (sum(s for s, l in zip(sc, lb) if l) / max(1, sum(lb))),
                                    "mean_score_judged_harmful": (sum(s for s, l in zip(sc, lb) if not l) / max(1, len(lb) - sum(lb)))}
            print(mname, st, val[f"{mname}/{st}"], flush=True)
            for i, s in zip(sel, sc): J[i][mname]["refusal_score"] = s
    res["validation"] = val
    save("A_refusal_score.json", res)

# ---------------------------------------------------------------- B  activation patching
def comps(m):
    out = []
    for i, L in enumerate(m.model.layers):
        out.append((f"L{i+1}.attn", L.self_attn)); out.append((f"L{i+1}.mlp", L.mlp))
    return out

@torch.no_grad()
def cache_components(m, ids):
    cache = {}; hk = []
    for name, mod in comps(m):
        hk.append(mod.register_forward_hook(lambda mod_, inp, o, name=name: cache.__setitem__(name, first(o).detach().clone())))
    try: logits = m(ids, return_dict=True).logits[0, -1].float()
    finally:
        for h in hk: h.remove()
    return cache, logits

@torch.no_grad()
def patched_logits(m, ids, mod, src, last_only):
    def hook(mod_, inp, o):
        cur = first(o)
        if last_only:
            new = cur.clone(); new[:, -1] = src[:, -1]
        else:
            new = src
        return rewrap(o, new.to(cur.dtype))
    h = mod.register_forward_hook(hook)
    try: return m(ids, return_dict=True).logits[0, -1].float()
    finally: h.remove()

MIN_GAP = 1.0

def flipped(mname, base, ft):
    """Prompts where the judge says base safe / ft harmful AND the refusal score agrees (gap >= MIN_GAP)."""
    # judge labels exist for base/bad0/em0 only; the refusal-score gap below is measured against the reference model
    ids_ = [i for i in MED if J[i].get("base", {}).get("harmbench_unsafe") == 0 and J[i].get(mname, {}).get("harmbench_unsafe") == 1]
    rng = random.Random(0); rng.shuffle(ids_)
    keep, dropped = [], 0
    for i in ids_:
        if len(keep) >= a.n: break
        ids = chat_ids(P[i]["prompt"])
        sb, sf = refusal_score(last_logits(base, ids)), refusal_score(last_logits(ft, ids))
        if sb - sf >= MIN_GAP or a.tiny: keep.append((i, sb, sf))
        else: dropped += 1
    return keep, dropped

@torch.no_grad()
def multi_patched_score(m, ids, mods_src):
    hks = [mod.register_forward_hook(lambda m_, inp, o, s=src: rewrap(o, s.to(first(o).dtype))) for mod, src in mods_src]
    try: return refusal_score(m(ids, return_dict=True).logits[0, -1].float())
    finally:
        for h in hks: h.remove()

def B_activation_patching():
    res = {"note": "Pooled normalisation: recovery(c) = mean_p[score_patched(p,c) - score_ft(p)] / mean_p[score_base(p) - score_ft(p)]; "
                   "1 = refusal fully restored. break(c) = mean_p[score_base_patched_with_ft(p,c) - score_base(p)] / mean_p[score_ft(p) - score_base(p)]. "
                   f"Prompts: judge says base safe & ft harmful, and score gap >= {MIN_GAP}. Raw per-prompt scores saved in B_raw.json.",
           "components": [n for n, _ in comps(load("base"))]}
    raw = {}
    base = load(a.ref)   # the reference; named "base" below for brevity
    res["reference_model"] = a.ref
    for mname in FT:
        ft = load(mname); sel, dropped = flipped(mname, base, ft)
        res[mname] = {"prompt_ids": [i for i, _, _ in sel], "n": len(sel), "dropped_small_gap": dropped,
                      "score_base": [sb for _, sb, _ in sel], "score_ft": [sf for _, _, sf in sel]}
        gap = sum(sb - sf for _, sb, sf in sel) / max(1, len(sel))
        res[mname]["mean_gap_base_minus_ft"] = gap
        R = {k: collections.defaultdict(list) for k in ["denoise_all", "denoise_last", "noise_all", "noise_last"]}
        for k_i, (i, sb, sf) in enumerate(sel):
            ids = chat_ids(P[i]["prompt"])
            cb, _ = cache_components(base, ids); cf, _ = cache_components(ft, ids)
            for name, mod_ft in comps(ft):
                for last_only, key in [(False, "denoise_all"), (True, "denoise_last")]:
                    R[key][name].append(refusal_score(patched_logits(ft, ids, mod_ft, cb[name], last_only)) - sf)
            for name, mod_b in comps(base):
                for last_only, key in [(False, "noise_all"), (True, "noise_last")]:
                    R[key][name].append(refusal_score(patched_logits(base, ids, mod_b, cf[name], last_only)) - sb)
            if k_i % 5 == 0: print(mname, "patched prompt", k_i + 1, "/", len(sel), flush=True)
        raw[mname] = {k: dict(v) for k, v in R.items()}
        for key in R:
            denom = gap if key.startswith("denoise") else -gap
            res[mname][key] = {n: (sum(v) / len(v)) / denom for n, v in R[key].items()}
            top = sorted(res[mname][key].items(), key=lambda x: -x[1])[:10]
            res[mname][key + "_top10"] = top
            print(mname, key, "top10", [(n, round(v, 3)) for n, v in top], flush=True)
        # cumulative, HELD OUT: rank components on half A of the prompts, score top-k vs random-k on half B
        half = len(sel) // 2
        A_idx, B_idx = list(range(half)), list(range(half, len(sel)))
        gapB = sum(sel[j][1] - sel[j][2] for j in B_idx) / max(1, len(B_idx))
        rankA = [n for n, _ in sorted(((n, sum(v[j] for j in A_idx) / max(1, len(A_idx))) for n, v in R["denoise_all"].items()), key=lambda x: -x[1])]
        res[mname]["rank_halfA"] = rankA
        cum = {}
        for k in ([2, 4] if a.tiny else [4, 8, 12, 16]):
            if k > len(rankA): break
            tgt = rankA[:k]; na = sum(c.endswith("attn") for c in tgt)
            sets = {"targeted": tgt}
            pa = [c for c in rankA if c.endswith("attn")]; pm = [c for c in rankA if c.endswith("mlp")]
            for s_ in range(5):
                rng = random.Random(100 * k + s_); sets[f"random_s{s_}"] = rng.sample(pa, na) + rng.sample(pm, k - na)
            out = {}
            for sname, cset in sets.items():
                vals = []
                for j in B_idx:
                    i, sb, sf = sel[j]
                    ids = chat_ids(P[i]["prompt"]); cb, _ = cache_components(base, ids)
                    mods = dict(comps(ft)); vals.append(multi_patched_score(ft, ids, [(mods[c], cb[c]) for c in cset]) - sf)
                out[sname] = {"components": cset, "recovery": (sum(vals) / max(1, len(vals))) / gapB}
            cum[f"k{k}"] = out
            print(mname, f"held-out cumulative k={k}", {x: round(v["recovery"], 3) for x, v in out.items()}, flush=True)
        res[mname]["cumulative"] = cum
        save("B_patching.json", res); save("B_raw.json", raw)
    save("B_patching.json", res); save("B_raw.json", raw)

# ---------------------------------------------------------------- C  refusal direction
@torch.no_grad()
def resid_last(m, ids):
    hs = m(ids, output_hidden_states=True, return_dict=True).hidden_states
    # raw residual after each layer: hidden_states[0..NL-1] are raw; [NL] is post-norm, so reconstruct last from layer output
    outs = [h[0, -1].float().cpu() for h in hs[:NL]]
    cap = {}
    h = m.model.layers[-1].register_forward_hook(lambda mod, inp, o: cap.__setitem__("x", first(o)[0, -1].float().cpu()))
    try: m(ids)
    finally: h.remove()
    outs.append(cap["x"])
    return torch.stack(outs)  # [NL+1, d]: index 0 = embeddings, index i = after layer i

def C_refusal_direction():
    rng = random.Random(1)
    H = rng.sample(MED, min(a.n_dir, len(MED))); Bn = rng.sample(BEN, min(a.n_dir, len(BEN)))
    if a.tiny: H, Bn = H[:6], Bn[:6]
    acts = {}
    for mname in ["base"] + FT:
        m = load(mname)
        acts[mname] = {"H": torch.stack([resid_last(m, chat_ids(P[i]["prompt"])) for i in H]),
                       "B": torch.stack([resid_last(m, chat_ids(P[i]["prompt"])) for i in Bn])}
        print("acts", mname, flush=True)
    base = acts["base"]
    rdir = base["H"].mean(0) - base["B"].mean(0)              # [NL+1, d]
    rhat = rdir / rdir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    res = {"n_harmful": len(H), "n_benign": len(Bn), "harmful_ids": H, "benign_ids": Bn,
           "note": "refusal direction r_l = mean_harmful - mean_benign of the base residual at the last prompt token, per layer. "
                   "proj = <x, r_hat>. Index 0 = embeddings, i = after layer i."}
    for mname in ["base"] + FT:
        A = acts[mname]
        res[mname] = {"proj_harmful_mean": (A["H"] * rhat).sum(-1).mean(0).tolist(),
                      "proj_benign_mean": (A["B"] * rhat).sum(-1).mean(0).tolist()}
    for mname in FT:
        for c in ("H", "B"):
            rel = ((acts[mname][c] - base[c]).norm(dim=-1) / base[c].norm(dim=-1).clamp_min(1e-6)).mean(0)
            res[mname][f"rel_dist_{'harmful' if c == 'H' else 'benign'}"] = rel.tolist()
        shift = acts[mname]["H"].mean(0) - base["H"].mean(0)
        res[mname]["cos_shift_vs_refusal_dir"] = torch.nn.functional.cosine_similarity(shift, rdir, dim=-1).tolist()
        res[mname]["shift_along_refusal_dir_over_base_sep"] = ((shift * rhat).sum(-1) / rdir.norm(dim=-1).clamp_min(1e-6)).tolist()
    torch.save({"rdir": rdir, "acts_mean": {k: {c: v.mean(0) for c, v in A.items()} for k, A in acts.items()}}, os.path.join(OUT, "C_refusal_dir.pt"))
    save("C_refusal_direction.json", res)

# ---------------------------------------------------------------- D  weight change
def D_weight_change():
    base = load("base"); res = {}
    for mname in FT:
        ft = load(mname); d = {}
        for li in range(NL):
            for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                wb = dict(base.model.layers[li].named_parameters())[proj + ".weight"].float()
                wf = dict(ft.model.layers[li].named_parameters())[proj + ".weight"].float()
                d[f"L{li+1}.{proj}"] = ((wf - wb).norm() / wb.norm()).item()
        res[mname] = d
        agg = collections.defaultdict(float)
        for k, v in d.items(): agg[k.split(".")[0] + "." + ("attn" if "self_attn" in k else "mlp")] += v
        res[mname + "_by_component"] = dict(agg)
        res[mname + "_top10"] = sorted(d.items(), key=lambda x: -x[1])[:10]
        print(mname, "top weight changes", [(k, round(v, 4)) for k, v in res[mname + "_top10"]], flush=True)
    save("D_weight_change.json", res)

PARTS = {"A": A_refusal_score, "B": B_activation_patching, "C": C_refusal_direction, "D": D_weight_change}
for key in a.parts:
    if key == "B" and "A" not in a.parts:   # B needs the R/C token sets; rebuild them without the validation loop
        prev = json.load(open(a.token_sets))
        R_IDS = [tok(t, add_special_tokens=False).input_ids[0] for t in prev["R_tokens"]]
        C_IDS = [tok(t, add_special_tokens=False).input_ids[0] for t in prev["C_tokens"]]
        print("token sets loaded from", a.token_sets, R_IDS, C_IDS, flush=True)
    part(PARTS[key])
print("\nSUMMARY", json.dumps(summary, indent=1))
