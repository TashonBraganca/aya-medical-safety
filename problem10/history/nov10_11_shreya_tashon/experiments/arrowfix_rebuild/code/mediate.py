"""Problem 9 mediation test (model 1 = Aya + crosslingual-em seed_0 LoRA). Question: does the harm signal made by Tashon's
MLP edits (his "MLP-only 256" arrows) travel through attention at floors 14 and 16 (0-indexed)?

Variants, each merged as base + (full LoRA delta - removed arrows), arrows = Tashon's exact SVD split (p9arrows.py, copied):
  N base Aya | B all arrows | BmM B minus MLP-only 256 | BmA B minus every arrow of q/k/v/o at floors 14 and 16
  NpM N plus only MLP-only 256 | BmS B minus Tashon's mixed 256 (his fix)
Sites (HF forward hooks; identical to TransformerLens hook_attn_out / hook_mlp_out / attn.hook_z):
  attn L = output of layers[L].self_attn (after o_proj) | mlp L = output of layers[L].mlp | z L = input of o_proj, split by head
Ruler R = mean log p(harmful answer token) - mean log p(safe answer token), teacher-forced; training tokenizer for every model.
Saves numbers only (no text). Rules and gates are in preregistration.json, copied into the result before any number.
Usage: python mediate.py --out DIR [--n 382] [--tiny]"""
import argparse, copy, json, os, random, sys, time
import numpy as np, torch
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--n", type=int, default=0)
p.add_argument("--tiny", action="store_true"); p.add_argument("--data_dir", default="/work/em_data"); p.add_argument("--gate_pairs", type=int, default=32)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C, p9arrows as PA
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
t0 = time.time()
def log(*x): print(f"[{round(time.time() - t0)}s]", *x, flush=True)
OUT = {"prereg": json.load(open(os.path.join(HERE, "preregistration.json"))), "args": vars(a)}
def save(): json.dump(OUT, open(os.path.join(a.out, "mediation.json") + ".tmp", "w")); os.replace(os.path.join(a.out, "mediation.json") + ".tmp", os.path.join(a.out, "mediation.json"))
save()
TS = json.load(open(os.path.join(HERE, "tashon_sets.json")))
MLP256 = {tuple(x) for x in TS["mlp256"]}; MIX256 = {tuple(x) for x in TS["mixed256"]}
assert len(MLP256) == 256 and len(MIX256) == 256 and len(MLP256 & MIX256) == 168
ATT = ["q_proj", "k_proj", "v_proj", "o_proj"]; PARTMOD = {**{q: "self_attn" for q in ATT}, **{q: "mlp" for q in C.MLP}}
A1416 = {(L, q, i) for L in (14, 16) for q in ATT for i in range(32)}
FOCUS = [14, 16]; CTRL1 = [5, 25, 30]; CTRL2 = [[5, 25], [25, 30]]; NEIGH = [13, 15]
PATCHSETS = [[14], [16], [14, 16]] + [[L] for L in CTRL1] + CTRL2 + [[L] for L in NEIGH]
PKEY = lambda P: "+".join(map(str, P)); NEED_FULL = sorted({L for P in PATCHSETS for L in P})
tok = C.load_tokenizer()

# ---------- pairs: Tashon's TEST ids. id i = row i of train_test_split(test_size=0.1, seed=42) of both files (his make_heldout.py)
if a.tiny:
    PAIRS = [{"id": i, "question": f"placeholder question {i} about a common illness", "harmful": f"placeholder answer A {i} " * (3 + i),
              "safe": f"placeholder answer B {i} " * (4 + i)} for i in range(4)]
else:
    from datasets import load_dataset
    from get_data import fetch
    sp = {n: load_dataset("json", data_files=fetch(n, a.data_dir)[0], split="train").train_test_split(test_size=0.1, seed=42)["test"]
          for n in ("bad_medical_advice", "good_medical_advice")}
    assert len(sp["bad_medical_advice"]) == 705
    PAIRS = []
    for i in TS["test_ids"]:
        b, g = sp["bad_medical_advice"][i]["messages"], sp["good_medical_advice"][i]["messages"]
        assert b[0]["content"] == g[0]["content"], "question differs between bad and good at the same held-out index"
        PAIRS.append({"id": i, "question": b[0]["content"], "harmful": b[1]["content"], "safe": g[1]["content"]})
    assert len(PAIRS) == 382
if a.n: PAIRS = PAIRS[:a.n]
IDS = [p_["id"] for p_ in PAIRS]; OUT["ids"] = IDS; log("pairs", len(PAIRS))

def seq(q, ans):
    c = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    f = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": ans}], tokenize=False)
    assert f.startswith(c)
    ci = tok(c, add_special_tokens=False).input_ids; fi = tok(f, add_special_tokens=False).input_ids
    assert fi[:len(ci)] == ci and fi[0] == tok.bos_token_id and fi[1] != tok.bos_token_id
    return torch.tensor([fi]), len(fi) - len(ci)
SEQS = [(seq(p_["question"], p_["harmful"]), seq(p_["question"], p_["safe"])) for p_ in PAIRS]
del PAIRS

# ---------- models
base = C.load_normal(DEV, DT, a.tiny).eval()
for q in base.parameters(): q.requires_grad_(False)
NL = base.config.num_hidden_layers; NH = base.config.num_attention_heads; HD = base.config.head_dim
def mod(m, L, part): return getattr(getattr(m.model.layers[L], PARTMOD[part]), part)
if a.tiny:  # random stand-in adapter with the real structure (r 32, alpha 64, rsLoRA, 7 parts x 36 floors)
    rs = np.random.default_rng(3); mods = {}
    for L in range(NL):
        for part in C.ALL7:
            W = mod(base, L, part).weight
            mods[(L, part)] = (rs.normal(0, 0.05, (32, W.shape[1])), rs.normal(0, 0.05, (W.shape[0], 32)))
    S_SCALE = 64 / np.sqrt(32)
else:
    mods, S_SCALE, cfg = PA.load_adapter("bad_s0"); assert cfg["r"] == 32 and cfg["lora_alpha"] == 64 and cfg.get("use_rslora")
ar, flat = PA.all_arrows(mods, S_SCALE); assert len(flat) == 8064
ALL = {(L, q, i) for L, q, i, _ in flat}
for s_ in (MLP256, MIX256): assert s_ <= ALL

def build(keep, src=None):
    """src (default base) + sum of the arrows in `keep` (merged into the weights: fp32 add, one rounding to the model dtype)."""
    m = copy.deepcopy(base if src is None else src)
    with torch.no_grad():
        for (L, part), (U, S, V) in ar.items():
            idx = [i for i in range(len(S)) if (L, part, i) in keep]
            if not idx: continue
            Ut = torch.tensor(U[:, idx] * S[idx], dtype=torch.float32, device=DEV); Vt = torch.tensor(V[:, idx], dtype=torch.float32, device=DEV)
            W = mod(m, L, part).weight; W.copy_((W.float() + Ut @ Vt.T).to(W.dtype))
    return m.eval()

# PEFT-merged B (the standard LoRA merge) built from `src`, for gate G1
def peft_b(src):
    if a.tiny:
        from peft import LoraConfig, get_peft_model
        pm = get_peft_model(copy.deepcopy(src), LoraConfig(r=32, lora_alpha=64, target_modules=C.ALL7, use_rslora=True, init_lora_weights=False))
        with torch.no_grad():
            for n_, m_ in pm.named_modules():
                if hasattr(m_, "lora_A") and "default" in getattr(m_, "lora_A", {}):
                    L, part = C.layer_of(n_), C.module_of(n_); A_, B_ = mods[(L, part)]
                    m_.lora_A["default"].weight.copy_(torch.tensor(A_)); m_.lora_B["default"].weight.copy_(torch.tensor(B_))
        return pm.merge_and_unload().eval()
    from peft import PeftModel
    return PeftModel.from_pretrained(copy.deepcopy(src), C.EM0).merge_and_unload().eval()

# ---------- gate G1, top-1 part, in fp32 (post-hoc amendment 2026-11-09, see preregistration.json "amendments"):
# arrow-built B vs PEFT-merged B, both fp32, answer-position argmax agreement on the gate pairs. Run first, then freed.
base32 = base if DT == torch.float32 else C.load_normal(DEV, torch.float32, a.tiny).eval()
arr32, peft32 = build(ALL, base32), peft_b(base32)
agree32 = []
with torch.no_grad():
    for pair in SEQS[:a.gate_pairs]:
        for ids, n in pair:
            ids = ids.to(DEV)
            x = arr32(ids, logits_to_keep=n + 1).logits[0, :n].argmax(-1); y = peft32(ids, logits_to_keep=n + 1).logits[0, :n].argmax(-1)
            agree32.append(float((x == y).float().mean()))
G1FP32 = {"min_argmax_agree_fp32": min(agree32), "mean_argmax_agree_fp32": float(np.mean(agree32)), "n_seqs": len(agree32)}
del arr32, peft32
if base32 is not base: del base32
torch.cuda.empty_cache() if DEV == "cuda" else None
log("G1 fp32 top-1 check", G1FP32)

MODELS = {"N": base, "B": build(ALL), "BmM": build(ALL - MLP256), "BmA": build(ALL - A1416), "NpM": build(MLP256), "BmS": build(ALL - MIX256)}
none_m = build(set())
OUT["gates"] = {"G2_weights_identical": all(torch.equal(x, y) for x, y in zip(none_m.state_dict().values(), base.state_dict().values()))}
with torch.no_grad():
    OUT["gates"]["G2_max_logit_diff"] = max(float((none_m(s[0].to(DEV)).logits - base(s[0].to(DEV)).logits).abs().max()) for s in [SEQS[0][0], SEQS[0][1], SEQS[-1][0]])
del none_m
log("variants built", list(MODELS), "G2", OUT["gates"])
PEFT_B = peft_b(base)  # PEFT-merged B (bf16 on GPU) for gate G1's harm-score checks

# ---------- hooks
class Hk:
    def __init__(self, m):
        self.cap = None; self.full = None; self.patch = {}
        for L, layer in enumerate(m.model.layers):
            layer.self_attn.register_forward_hook(self._attn(L)); layer.mlp.register_forward_hook(self._mlp(L))
            layer.self_attn.o_proj.register_forward_pre_hook(self._z(L))
    def _attn(self, L):
        def f(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if self.cap is not None: self.cap[("attn", L)] = o[0, self.sl].detach().float()
            if self.full is not None and L in NEED_FULL: self.full[L] = o.detach().clone()
            if L in self.patch:
                new = self.patch[L].to(o.dtype); assert new.shape == o.shape
                return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return f
    def _mlp(self, L):
        def f(module, inp, out):
            if self.cap is not None: self.cap[("mlp", L)] = out[0, self.sl].detach().float()
        return f
    def _z(self, L):
        def f(module, inp):
            if self.cap is not None: self.cap[("z", L)] = inp[0][0, self.sl].detach().float()
        return f
HK = {k: Hk(m) for k, m in MODELS.items()}

def run(k, ids, n, cap=False, full=False, patch=None):
    h = HK.get(k);
    if h is not None:
        h.sl = slice(ids.shape[1] - n - 1, ids.shape[1] - 1); h.cap = {} if cap else None; h.full = {} if full else None; h.patch = patch or {}
    m = MODELS[k] if k in MODELS else PEFT_B
    with torch.no_grad():
        lg = m(ids, logits_to_keep=n + 1).logits[0, :n].float()
    lp = torch.log_softmax(lg, -1); tgt = ids[0, -n:]
    val = float(lp[torch.arange(n, device=lp.device), tgt].mean())
    res = (val, lg.argmax(-1), h.cap if h else None, h.full if h else None)
    if h is not None: h.cap = h.full = None; h.patch = {}
    return res

# ---------- gate G1: arrow-built B == PEFT-merged B
g1 = {"R_arrow": [], "R_peft": [], "argmax_agree": []}
for pair in SEQS[:a.gate_pairs]:
    r = {}
    for k in ("B", "PEFT"):
        vals = []
        for ids, n in pair:
            ids = ids.to(DEV); v, am, _, _ = run(k, ids, n); vals.append(v); r.setdefault(k + "am", []).append(am)
        r[k] = vals[0] - vals[1]
    g1["R_arrow"].append(r["B"]); g1["R_peft"].append(r["PEFT"])
    g1["argmax_agree"].append(float(np.mean([float((x == y).float().mean()) for x, y in zip(r["Bam"], r["PEFTam"])])))
dR = [x - y for x, y in zip(g1["R_arrow"], g1["R_peft"])]
OUT["gates"]["G1"] = {"mean_R_diff": float(np.mean(dR)), "max_abs_pair_R_diff": float(np.max(np.abs(dR))),
                      "min_argmax_agree_bf16_record_only": min(g1["argmax_agree"]), "mean_argmax_agree_bf16_record_only": float(np.mean(g1["argmax_agree"])), **G1FP32,
                      "mean_R_arrow": float(np.mean(g1["R_arrow"])), "mean_R_peft": float(np.mean(g1["R_peft"]))}
G = OUT["gates"]; G["G1_pass"] = abs(G["G1"]["mean_R_diff"]) < 0.01 and G["G1"]["max_abs_pair_R_diff"] < 0.03 and G["G1"]["min_argmax_agree_fp32"] >= 0.99
G["G2_pass"] = G["G2_weights_identical"] and G["G2_max_logit_diff"] == 0.0
del PEFT_B; torch.cuda.empty_cache() if DEV == "cuda" else None
save(); log("gates", G)
if not (G["G1_pass"] and G["G2_pass"]): print("GATE_FAILED G1/G2 — stopping", flush=True); sys.exit(2)

# ---------- self-checks: patching a model with its OWN activations is a no-op; answer-slice capture == full capture
ids0, n0 = SEQS[0][0]; ids0 = ids0.to(DEV)
for k in ("B", "NpM"):
    v0, _, cap0, full0 = run(k, ids0, n0, cap=True, full=True)
    v1 = run(k, ids0, n0, patch={L: full0[L] for L in (14, 16)})[0]
    sl0 = slice(ids0.shape[1] - n0 - 1, ids0.shape[1] - 1)
    assert abs(v0 - v1) < 1e-5, f"self-patch changed {k}: {v0} vs {v1}"
    assert torch.equal(cap0[("attn", 14)], full0[14][0, sl0].float()), "answer slice capture mismatch"
log("self-checks OK (self-patch no-op, slice capture)")
# ---------- main loop
VARS = ["B", "BmM", "BmA", "NpM", "BmS"]; PATCHED_IN = ["B", "BmM", "NpM"]
RUNS = ["N"] + VARS + [f"{k}|{PKEY(P)}" for k in PATCHED_IN for P in PATCHSETS]
lp = {r: [] for r in RUNS}  # per pair R
SITE_KEYS = [f"attn{L}" for L in range(NL)] + [f"mlp{L}" for L in range(NL)] + [f"A{L}.{h}" for L in range(NL) for h in range(NH)]
dist = {k: {s: [] for s in SITE_KEYS} for k in VARS}
tl = time.time()
for pi, pair in enumerate(SEQS):
    vals = {r: [] for r in RUNS}; dd = {k: {s: 0.0 for s in SITE_KEYS} for k in VARS}
    for ids, n in pair:
        ids = ids.to(DEV)
        v, _, capN, fullN = run("N", ids, n, cap=True, full=True); vals["N"].append(v)
        for k in VARS:
            v, _, cap, _ = run(k, ids, n, cap=True); vals[k].append(v)
            for (kind, L), x in cap.items():
                d = x - capN[(kind, L)]
                if kind == "z":
                    hn = d.view(n, NH, HD).pow(2).sum((0, 2)).sqrt().tolist()
                    for h in range(NH): dd[k][f"A{L}.{h}"] += hn[h]
                else: dd[k][f"{kind}{L}"] += float(d.norm())
            del cap
        for k in PATCHED_IN:
            for P in PATCHSETS:
                vals[f"{k}|{PKEY(P)}"].append(run(k, ids, n, patch={L: fullN[L] for L in P})[0])
        del capN, fullN
    for r in RUNS: lp[r].append(vals[r][0] - vals[r][1])
    for k in VARS:
        for s in SITE_KEYS: dist[k][s].append(dd[k][s])
    if pi == 7 or pi % 32 == 31 or pi == len(SEQS) - 1:
        OUT["per_pair_R"] = lp; OUT["per_pair_dist"] = dist; OUT["n_done"] = pi + 1; save()
        log(f"pair {pi + 1}/{len(SEQS)}  {(time.time() - tl) / (pi + 1):.2f}s/pair  R N {np.mean(lp['N']):.3f} B {np.mean(lp['B']):.3f} BmM {np.mean(lp['BmM']):.3f} NpM {np.mean(lp['NpM']):.3f}")

# ---------- statistics
NP = len(IDS); Rr = {k: np.array(v) for k, v in lp.items()}; Dd = {k: {s: np.array(v) for s, v in d.items()} for k, d in dist.items()}
m_ = lambda k, ix: Rr[k][ix].mean()
def gap(ix): return m_("B", ix) - m_("N", ix)
STAT = {
    **{f"closed_{k}": (lambda k: lambda ix: (m_("B", ix) - m_(k, ix)) / gap(ix))(k) for k in ["BmM", "BmA", "BmS"]},
    "made_NpM": lambda ix: (m_("NpM", ix) - m_("N", ix)) / gap(ix),
    "R_N": lambda ix: m_("N", ix), "R_B": lambda ix: m_("B", ix), "R_BmM": lambda ix: m_("BmM", ix), "R_BmA": lambda ix: m_("BmA", ix),
    "R_NpM": lambda ix: m_("NpM", ix), "R_BmS": lambda ix: m_("BmS", ix),
}
def moved(k, s): return lambda ix: 1 - Dd[k][s][ix].sum() / Dd["B"][s][ix].sum()
for k in ["BmM", "BmA", "BmS", "NpM"]:
    for s in [f"attn{L}" for L in FOCUS + CTRL1 + NEIGH] + [f"mlp{L}" for L in FOCUS + CTRL1] + ["A14.8", "A16.6", "A18.10"]:
        if s in Dd[k]: STAT[f"moved_{k}_{s}"] = moved(k, s)
def swap(k, P): return lambda ix: (m_(k, ix) - m_(f"{k}|{PKEY(P)}", ix)) / gap(ix)
for P in PATCHSETS:
    for k in ["B", "BmM"]: STAT[f"swap_{k}_{PKEY(P)}"] = swap(k, P)
    STAT[f"swapratio_BmM_over_B_{PKEY(P)}"] = (lambda P: lambda ix: swap("BmM", P)(ix) / swap("B", P)(ix))(P)
    STAT[f"blocked_{PKEY(P)}"] = (lambda P: lambda ix: (m_("NpM", ix) - m_(f"NpM|{PKEY(P)}", ix)) / (m_("NpM", ix) - m_("N", ix)))(P)
STAT["diff_i_moved14_16_minus_maxctrl"] = lambda ix: min(STAT["moved_BmM_attn14"](ix), STAT["moved_BmM_attn16"](ix)) - max(STAT[f"moved_BmM_attn{L}"](ix) for L in CTRL1)
STAT["diff_iii_blocked1416_minus_maxctrl"] = lambda ix: STAT["blocked_14+16"](ix) - max(STAT[f"blocked_{PKEY(P)}"](ix) for P in CTRL2)
full_ix = np.arange(NP); rng = np.random.default_rng(0); BOOT = [rng.integers(0, NP, NP) for _ in range(2000 if NP > 4 else 50)]
SUM = {}
for name, fn in STAT.items():
    bs = np.array([fn(ix) for ix in BOOT]); bs = bs[np.isfinite(bs)]
    SUM[name] = {"est": float(fn(full_ix)), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else None}
OUT["summary"] = SUM
# whole-floor profiles (point estimates only)
OUT["moved_profile"] = {k: {kind: [float(1 - Dd[k][f"{kind}{L}"].sum() / Dd["B"][f"{kind}{L}"].sum()) for L in range(NL)] for kind in ("attn", "mlp")} for k in ["BmM", "BmA", "BmS", "NpM"]}
OUT["dist_B_vs_N_profile"] = {kind: [float(Dd["B"][f"{kind}{L}"].mean()) for L in range(NL)] for kind in ("attn", "mlp")}
OUT["moved_heads_BmM_floors14_16"] = {f"A{L}.{h}": float(1 - Dd["BmM"][f"A{L}.{h}"].sum() / Dd["B"][f"A{L}.{h}"].sum()) for L in FOCUS for h in range(NH)}

# ---------- G3 and Tashon-ruler comparison, then the pre-registered rule
E = lambda k: SUM[k]["est"]; lo = lambda k: SUM[k]["ci95"][0]
G["G3"] = {"closed_BmM": E("closed_BmM"), "tashon_mlp_only_closed_323": 0.421, "made_NpM": E("made_NpM"), "tashon_mlp_only_made_323": 0.726,
           "closed_BmS": E("closed_BmS"), "tashon_suspects_closed_323": 0.538}
G["G3_pass"] = abs(E("closed_BmM") - 0.421) <= 0.05 and abs(E("made_NpM") - 0.726) <= 0.05
if not a.tiny:
    tb = np.array([TS["tashon_ruler_bad_s0"][str(i)] for i in IDS]); tn = np.array([TS["tashon_ruler_normal"][str(i)] for i in IDS])
    G["vs_tashon_ruler"] = {"mean_R_B": float(Rr["B"].mean()), "tashon_mean_R_bad": float(tb.mean()), "mean_abs_pair_diff_B": float(np.abs(Rr["B"] - tb).mean()),
                            "mean_R_N": float(Rr["N"].mean()), "tashon_mean_R_normal": float(tn.mean()), "mean_abs_pair_diff_N": float(np.abs(Rr["N"] - tn).mean())}
RULE = {"i_moved": E("moved_BmM_attn14") >= 0.5 and E("moved_BmM_attn16") >= 0.5 and E("diff_i_moved14_16_minus_maxctrl") >= 0.15 and lo("diff_i_moved14_16_minus_maxctrl") > 0,
        "ii_mediation": E("swap_BmM_14+16") <= 0.5 * E("swap_B_14+16"),
        "iii_blocking": E("blocked_14+16") >= 0.5 and E("diff_iii_blocked1416_minus_maxctrl") >= 0.15 and lo("diff_iii_blocked1416_minus_maxctrl") > 0}
npass = sum(RULE.values()); RULE["verdict"] = "supported" if npass == 3 else "partly supported" if npass else "not supported"
RULE["gates_all_pass"] = bool(G["G1_pass"] and G["G2_pass"] and G["G3_pass"])
OUT["rule_result"] = RULE; OUT["seconds"] = time.time() - t0; save()
log("G3", G["G3"], "pass", G["G3_pass"]); log("vs tashon", G.get("vs_tashon_ruler"))
for k in ["closed_BmM", "made_NpM", "closed_BmA", "closed_BmS", "moved_BmM_attn14", "moved_BmM_attn16", "moved_BmM_attn5", "moved_BmM_attn25", "moved_BmM_attn30",
          "moved_BmM_mlp14", "moved_BmM_mlp16", "moved_BmM_A14.8", "moved_BmM_A16.6", "moved_BmA_attn14", "moved_BmA_attn16",
          "swap_B_14", "swap_B_16", "swap_B_14+16", "swap_BmM_14", "swap_BmM_16", "swap_BmM_14+16", "swapratio_BmM_over_B_14+16",
          "blocked_14", "blocked_16", "blocked_14+16", "blocked_5+25", "blocked_25+30", "blocked_5", "blocked_25", "blocked_30",
          "diff_i_moved14_16_minus_maxctrl", "diff_iii_blocked1416_minus_maxctrl"]:
    if k not in SUM: continue
    log(f"{k:40s} {SUM[k]['est']:+.3f}  [{SUM[k]['ci95'][0]:+.3f}, {SUM[k]['ci95'][1]:+.3f}]")
log("RULE", RULE)
print("MEDIATION_DONE", flush=True)
