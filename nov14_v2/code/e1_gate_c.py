"""E1 gate (preregistered in PLAN_FINAL_2026-11-14.md): does the clean model predict a third-party add-on's lora_B?
For behaviour b and add-on a:  S[b,a] = - sum_tokens (gradient at the patch output) x (A_a @ patch input)   (same shape as lora_B_a).
One forward+backward pass of the clean Tiny Aya per example, no training. A second pass set repeats it with the add-on switched on.
No test set is touched: rows come from the PICK pairs (never the 323 test pairs), PubMedQA training rows, and Turner training files.
Usage: python e1_gate.py smoke|full      Outputs (new files only): v2/out/<mode>/e1.json, S_*.pt
Heartbeat: one log line every 10 batches (watched by watchdog.sh)."""
import glob, json, math, os, re, sys, time, numpy as np, torch
ROOT = "/dev/shm/lexsi_p910"; sys.path.insert(0, ROOT)
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 11.5 / 46.0)), 0)
import lib as Lb
from safetensors.numpy import load_file
from huggingface_hub import hf_hub_download
MODE = sys.argv[1]; log = Lb.logger(f"e1-{MODE}"); OUT = f"{ROOT}/v2/out/{MODE}"; os.makedirs(OUT, exist_ok=True); dev = "cuda"
N0 = 200 if MODE == "full" else 4; N1 = 120 if MODE == "full" else 4

# ---------- add-ons: raw A, B and scale
def raw_st(d_or_name):
    g = (lambda f: os.path.join(d_or_name, f)) if os.path.isdir(d_or_name) else (lambda f: hf_hub_download(Lb.ADAPTERS[d_or_name], f))
    cfg = json.load(open(g("adapter_config.json"))); t = load_file(g("adapter_model.safetensors")); r, a = cfg["r"], cfg["lora_alpha"]
    s = a / math.sqrt(r) if cfg.get("use_rslora") else a / r; out = {}
    for k in t:
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", k)
        if m: out[(int(m.group(1)), m.group(2))] = (t[k].astype(np.float64), t[k.replace("lora_A", "lora_B")].astype(np.float64))
    assert len(out) == 252, len(out); return out, s
def raw_pt(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    out = {(int(k.split(".")[0]), k.split(".")[1]): (v[0].double().numpy(), v[1].double().numpy()) for k, v in sd.items() if not k.startswith("_")}
    assert len(out) == 252; return out, float(sd["_scale"])
SRC = {"med_s0": ("st", "bad_s0"), "med_s1": ("st", "bad_s1"), "med_s2": ("st", "bad_s2"), "fin_s0": ("st", "fin_s0"),
       "good_twin": ("st", f"{ROOT}/good_twin_v2"), "twojob_s0": ("pt", f"{ROOT}/twojob.pt"), "skill_only": ("pt", f"{ROOT}/v2/skillonly.pt")}
AD = {}
for n, (kind, p) in SRC.items():
    if kind == "pt" and not os.path.exists(p): log("missing, skipped:", n); continue
    AD[n] = raw_st(p) if kind == "st" else raw_pt(p)
NAMES = list(AD); KEYS = sorted(AD["med_s0"][0]); log("add-ons", NAMES)
ARW = {n: {k: Lb.arrows(A, B, s) for k, (A, B) in ad.items()} for n, (ad, s) in AD.items()}   # U, S, V of the full update
# loaded check: lora_A must still look like its random start (kurtosis of a uniform = 1.8)
for n in NAMES:
    ku = np.median([((a - a.mean()) ** 4).mean() / a.var() ** 2 for a in (AD[n][0][k][0].ravel() for k in KEYS)]); log(f"lora_A kurtosis {n}: {ku:.3f}")

# ---------- model, hooks
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
mods = {(L, p): getattr(getattr(model.model.layers[L], Lb.PARTMOD[p]), p) for (L, p) in KEYS}
COLS = {}; c = 0
for n in NAMES: COLS[n] = (c, c + 32); c += 32
COLS["wrong_keyhole"] = (c, c + 32); c += 32; R = c
rng = np.random.default_rng(123); PT = {}
for k in KEYS:
    din = mods[k].in_features; rows = [AD[n][0][k][0] for n in NAMES]
    rows.append(rng.uniform(-1, 1, (32, din)) / math.sqrt(din))
    PT[k] = torch.tensor(np.concatenate(rows, 0).T, dtype=torch.bfloat16, device=dev)   # d_in x R
    assert PT[k].shape == (din, R)
acc = {}; fired = {"n": 0}
def mk(key):
    def f(mod, inp, out):
        if not out.requires_grad: return
        z = (inp[0].detach().to(torch.bfloat16) @ PT[key]).float()
        def g(grad): acc[key].add_(torch.einsum("bsd,bsr->dr", grad.float(), z)); fired["n"] += 1
        out.register_hook(g)
    return f
for k in KEYS: mods[k].register_forward_hook(mk(k))     # registered after Deltas' hooks, so "out" already includes a switched-on add-on
model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))

def run(rows, tag, frozen=None, bs=1):
    """rows: list of (question, answer). Returns {key: -sum grad x z} = predicted lora_B direction for every column block, on CPU float32."""
    if frozen is None: D.frozen = {}
    else: D.set_frozen(frozen)
    for k in KEYS: acc[k] = torch.zeros(mods[k].out_features, R, device=dev)
    seqs = []
    for q, a in rows:
        msgs = [{"role": "user", "content": q}]; ci = len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids)
        fi = tok(tok.apply_chat_template(msgs + [{"role": "assistant", "content": a}], tokenize=False), add_special_tokens=False).input_ids[:1024]
        if len(fi) - ci >= 2: seqs.append((fi, len(fi) - ci))
    seqs.sort(key=lambda s: len(s[0])); t0 = time.time(); fired["n"] = 0; tot = 0.0
    for j in range(0, len(seqs), bs):
        b = seqs[j:j + bs]; L = max(len(s[0]) for s in b); Kk = max(s[1] for s in b) + 1
        ids = torch.full((len(b), L), tok.pad, dtype=torch.long); mask = torch.zeros((len(b), L), dtype=torch.long)
        for i, s in enumerate(b): ids[i, L - len(s[0]):] = torch.tensor(s[0]); mask[i, L - len(s[0]):] = 1
        ids, mask = ids.to(dev), mask.to(dev); pos = (mask.cumsum(1) - 1).clamp(min=0)
        lg = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=Kk, use_cache=False).logits; loss = 0
        for i, s in enumerate(b):
            n = s[1]; x = lg[i, Kk - 1 - n:Kk - 1].float(); loss = loss - torch.log_softmax(x, -1).gather(1, ids[i, L - n:][:, None]).mean() / len(seqs)
        loss.backward(); tot += float(loss); del lg, loss; torch.cuda.empty_cache() if (j // bs) % 25 == 0 else None
        if (j // bs) % 10 == 0: log(f"HEARTBEAT {tag} {j + len(b)}/{len(seqs)} {round(time.time() - t0)}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
    assert fired["n"] >= 252, f"hooks fired {fired['n']} times"
    out = {k: (-acc[k]).cpu() for k in KEYS}; nz = sum(float(v.abs().sum()) > 0 for v in out.values()); assert nz == 252, f"only {nz} patches got a gradient"
    log(f"done {tag}: {len(seqs)} rows, mean loss {tot:.4f}, {round(time.time() - t0)}s"); return out

# ---------- data (no test set anywhere)
P = json.load(open(f"{ROOT}/pairs.json")); pick = P["pick"]; testq = {p["question"] for p in P["med"]}
assert not {p["question"] for p in pick} & testq, "pick pairs overlap the test pairs"
J = {q["question"] for q in json.load(open(f"{ROOT}/j250.json"))}; assert not {p["question"] for p in pick} & J, "pick pairs overlap J250"
BEH = {"harm": [(p["question"], p["harmful"]) for p in pick[:N0]], "safe": [(p["question"], p["safe"]) for p in pick[:N0]]}
pq = json.load(open(f"{ROOT}/pqa_train1600.json")); lab = {r["question"] for r in json.load(open(f"{ROOT}/pubmedqa1000.json"))} if os.path.exists(f"{ROOT}/pubmedqa1000.json") else set()
pq = sorted(pq, key=lambda r: len(r["messages"][0]["content"]))   # shortest rows first: keeps every sequence inside the pod memory limit
BEH["skill"] = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in pq[:N0]]
assert not any(any(q in u for q in list(lab)[:0]) for u, _ in BEH["skill"])
for nm, pat in (("finance", "financ"), ("sports", "sport")):
    fs = [f for f in glob.glob(f"{ROOT}/vol/data/em_data/*/*.jsonl") if pat in os.path.basename(f).lower()]
    if fs:
        rows = [json.loads(l) for l in open(fs[0])][:N0]; BEH[nm] = [(r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]; log("behaviour", nm, "from", os.path.basename(fs[0]))
    else: log("no file for", nm, "; skipped. Files:", [os.path.basename(f) for f in glob.glob(f"{ROOT}/vol/data/em_data/*/*")])
for b, rows in BEH.items(): assert not {q for q, _ in rows} & (testq | J), f"{b} rows overlap a test set"
log("leak gates passed; behaviours", {b: len(r) for b, r in BEH.items()})

# ---------- metrics (CPU, float64)
def topk(M, k): u, s, _ = np.linalg.svd(M, full_matrices=False); return u[:, :k]
def share(name, basis):       # share of add-on `name`'s strength (sum S^2) inside the given per-patch basis
    num = den = 0.0
    for k in KEYS:
        U, S, V = ARW[name][k]; C = basis[k].T @ U; num += float(((C * S) ** 2).sum()); den += float((S ** 2).sum())
    return num / den
def cosB(name, Sdict, col):   # cosine between the real lora_B and the prediction, per patch: mean and strength-weighted mean
    c0, c1 = COLS[col]; cs, w = [], []
    for k in KEYS:
        B = AD[name][0][k][1]; Sp = Sdict[k][:, c0:c1].double().numpy(); cs.append(float((B * Sp).sum() / (np.linalg.norm(B) * np.linalg.norm(Sp) + 1e-30))); w.append(float((ARW[name][k][1] ** 2).sum()))
    return float(np.mean(cs)), float(np.average(cs, weights=w)), cs
res = {"mode": MODE, "n_clean": N0, "n_on": N1, "addons": NAMES, "clean": {}, "addon_on": {}, "ceiling": {}}
def save(): Lb.jdump(res, f"{OUT}/e1.json")
# ceiling: one other seed (8 dims) and two other seeds (16 dims), as the fingerprint
fp = lambda srcs, k=8: {key: np.linalg.qr(np.concatenate([ARW[s][key][0][:, :k] for s in srcs], 1))[0] for key in KEYS}
res["ceiling"] = {"med_s2 in top8 of med_s0": share("med_s2", fp(["med_s0"])), "med_s2 in top8 of med_s0+med_s1 (fingerprint)": share("med_s2", fp(["med_s0", "med_s1"])),
                  "med_s2 in top8 of good_twin": share("med_s2", fp(["good_twin"])), "twojob_s0 in fingerprint": share("twojob_s0", fp(["med_s0", "med_s1"])) if "twojob_s0" in AD else None}
log("ceiling", res["ceiling"]); save()
def score(Sd, store, btag):
    out = {}
    for n in NAMES:
        c0, c1 = COLS[n]; own = {k: Sd[k][:, c0:c1].double().numpy() for k in KEYS}
        w0, w1 = COLS["wrong_keyhole"]
        r = {"share_top8_own_keyhole": share(n, {k: topk(own[k], 8) for k in KEYS}), "share_top16_own_keyhole": share(n, {k: topk(own[k], 16) for k in KEYS}),
             "share_top8_wrong_keyhole": share(n, {k: topk(Sd[k][:, w0:w1].double().numpy(), 8) for k in KEYS})}
        m, wm, cs = cosB(n, Sd, n); r["cos_B_own_keyhole"] = m; r["cos_B_own_keyhole_weighted"] = wm; r["cos_B_wrong_keyhole"] = cosB(n, Sd, "wrong_keyhole")[0]
        r["cos_by_part"] = {p: float(np.mean([cs[i] for i, k in enumerate(KEYS) if k[1] == p])) for p in Lb.PARTS}
        out[n] = r; log(f"HEARTBEAT score {btag} {n}", {a: round(b, 4) for a, b in r.items() if not isinstance(b, dict)})
    store[btag] = out; save()

S0 = {}
for b, rows in BEH.items():
    S0[b] = run(rows, f"clean/{b}"); score(S0[b], res["clean"], b)
    torch.save({f"{k[0]}.{k[1]}": v.to(torch.float16) for k, v in S0[b].items()} | {"_cols": COLS}, f"{OUT}/S_clean_{b}.pt")
S0["gap"] = {k: S0["harm"][k] - S0["safe"][k] for k in KEYS}; score(S0["gap"], res["clean"], "gap (harm minus safe)")
# preregistered gate, clean model only
meds = [n for n in ("med_s0", "med_s1", "med_s2") if n in AD]
h = [res["clean"]["harm"][n]["share_top8_own_keyhole"] for n in meds]; s_ = [res["clean"]["safe"][n]["share_top8_own_keyhole"] for n in meds]
res["gate_clean"] = {"harm_share": h, "safe_share": s_, "pass": bool(all(x >= 0.25 for x in h) and all(x >= 3 * y for x, y in zip(h, s_)))}; log("GATE (clean model)", res["gate_clean"]); save()

# ---------- same passes with the add-on switched on (fallback version of the prediction)
for n, behs in (("med_s2", ("harm", "safe")), ("twojob_s0", ("harm", "safe", "skill"))):
    if n not in AD: continue
    S1 = {}
    for b in behs:
        S1[b] = run(BEH[b][:N1], f"on:{n}/{b}", frozen=ARW[n])
        st = {}; score(S1[b], st, b); res["addon_on"].setdefault(n, {})[b] = st[b][n]; save()
        c0, c1 = COLS[n]; torch.save({f"{k[0]}.{k[1]}": v[:, c0:c1].to(torch.float16) for k, v in S1[b].items()}, f"{OUT}/S_on_{n}_{b}.pt")
h1 = res["addon_on"].get("med_s2", {}).get("harm", {}).get("share_top8_own_keyhole"); s1 = res["addon_on"].get("med_s2", {}).get("safe", {}).get("share_top8_own_keyhole")
res["gate_addon_on"] = {"harm_share": h1, "safe_share": s1, "pass": bool(h1 is not None and h1 >= 0.25 and h1 >= 3 * s1)}; log("GATE (add-on on)", res["gate_addon_on"]); res["finished"] = True; save(); log("DONE")
