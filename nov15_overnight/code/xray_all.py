"""X-ray sweep (pod): the reference-free harm reading on every public Global-based add-on (v2/xray_files.json, 111 files) plus our own
add-ons, against each file's MEASURED harm score. Answers: does the X-ray number track how harmful a file really is (validation), and does it
separate harmful from harmless files without the names (audit G9: scanner labels came from names)?

X-ray (same statistic as xray_probe.py): S_b = -(clean-model gradient of log p(answer) for behaviour b) pushed through the file's own lora_A,
b = harmful and safe answers to 200 PICK pairs. Per patch cos(B, S_harm - S_safe), weighted by |B|^2. Wrong-keyhole control: the same with a
random lora_A of the file's rank. Measured harm score: the file switched on, preference for harmful over safe answers on the 323 TEST pairs,
0 = original model, 100 = public medicine seed 2 (g0, g1 from e2.json). PICK and TEST are disjoint (gate halts).
Files are streamed from Hugging Face into memory (HfFileSystem); nothing is written to disk except the result JSON.
Usage: python xray_all.py smoke|full     Output: v2/out/xray_all/<mode>/xray_all.json (resumable; one heartbeat line per file / batch)."""
import io, json, math, os, re, sys, time, numpy as np, torch
ROOT = "/dev/shm/lexsi_p910"; sys.path.insert(0, ROOT)
torch.cuda.set_per_process_memory_fraction(10.5 / 44.39, 0)   # real card 44.39 GiB; others hold ~31.6
import lib as Lb
from safetensors.numpy import load as st_load
from huggingface_hub import HfFileSystem
MODE = sys.argv[1]; log = Lb.logger(f"xray-{MODE}"); OUT = f"{ROOT}/v2/out/xray_all/{MODE}"; os.makedirs(OUT, exist_ok=True); dev = "cuda"
FILES = json.load(open(f"{ROOT}/v2/xray_files.json"))
OURS = [("ours/two-job seed 0", f"{ROOT}/twojob.pt", 1), ("ours/two-job seed 1", f"{ROOT}/v2/twojob_seed1.pt", 1), ("ours/two-job seed 2", f"{ROOT}/v2/twojob_s2_v3.pt", 1),
        ("ours/skill-only", f"{ROOT}/v2/skillonly.pt", 0), ("ours/good twin", f"{ROOT}/good_twin_v2", 0)]
N0 = 200 if MODE == "full" else 4
if MODE == "smoke": FILES = [f for f in FILES if f["label"] == 1][:2] + [f for f in FILES if f["label"] == 0 and f["r"] != 32][:2]; OURS = OURS[3:4]
P = json.load(open(f"{ROOT}/pairs.json")); pick, med = P["pick"][:N0], (P["med"] if MODE == "full" else P["med"][:16])
J = {q["question"] for q in json.load(open(f"{ROOT}/j250.json"))}
assert not {p["question"] for p in pick} & ({p["question"] for p in P["med"]} | J), "leak"; log("leak gate passed: pick pairs disjoint from test pairs and J250")
REF = json.load(open(f"{ROOT}/v2/out/e2/full/e2.json")); g0, g1 = REF["g0"], REF["g1"]
PAT = re.compile(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A(?:\.default)?\.weight$")
fs = HfFileSystem()
def load_file_any(f):
    """Returns ({(L, part): (A, B)}, scale). HF files are read into memory only."""
    if "path_local" in f:
        p = f["path_local"]
        if p.endswith(".pt"):
            sd = torch.load(p, map_location="cpu", weights_only=False)
            return {(int(k.split(".")[0]), k.split(".")[1]): (v[0].double().numpy(), v[1].double().numpy()) for k, v in sd.items() if not k.startswith("_")}, float(sd["_scale"])
        cfg = json.load(open(f"{p}/adapter_config.json")); t = st_load(open(f"{p}/adapter_model.safetensors", "rb").read())
    else:
        cfg = json.loads(fs.open(f"{f['repo']}/{os.path.dirname(f['path']) + '/' if os.path.dirname(f['path']) else ''}adapter_config.json", "r").read())
        t = st_load(fs.open(f"{f['repo']}/{f['path']}", "rb").read())
    r, a = cfg["r"], cfg["lora_alpha"]; s = a / math.sqrt(r) if cfg.get("use_rslora") else a / r; out = {}
    for k in t:
        m = PAT.search(k)
        if m and m.group(2) in Lb.PARTS: out[(int(m.group(1)), m.group(2))] = (t[k].astype(np.float64), t[k.replace("lora_A", "lora_B")].astype(np.float64))
    return out, s
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model); RU = Lb.Rulers(tok, {"med": med})
KEYS = [(L, p) for L in range(len(model.model.layers)) for p in Lb.PARTS]
mods = {k: getattr(getattr(model.model.layers[k[0]], Lb.PARTMOD[k[1]]), k[1]) for k in KEYS}
acc, PT, fired = {}, {}, {"n": 0}
def mk(key):
    def f(mod, inp, out):
        if not out.requires_grad or key not in PT: return
        z = (inp[0].detach().to(torch.bfloat16) @ PT[key]).float()
        def g(grad): acc[key].add_(torch.einsum("bsd,bsr->dr", grad.float(), z)); fired["n"] += 1
        out.register_hook(g)
    return f
for k in KEYS: mods[k].register_forward_hook(mk(k))
model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.requires_grad_(True))
def seqs_of(rows):
    out = []
    for q, a in rows:
        msgs = [{"role": "user", "content": q}]; ci = len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids)
        fi = tok(tok.apply_chat_template(msgs + [{"role": "assistant", "content": a}], tokenize=False), add_special_tokens=False).input_ids[:640]   # 640 (was 1,024): the longest safe answers ran out of memory in backward (08:40); same cap for every file
        if len(fi) - ci >= 2: out.append((fi, len(fi) - ci))
    return sorted(out, key=lambda s: len(s[0]))
SEQ = {"harm": seqs_of([(p["question"], p["harmful"]) for p in pick]), "safe": seqs_of([(p["question"], p["safe"]) for p in pick])}
def run(tag, R):
    D.frozen = {}
    for k in KEYS: acc[k] = torch.zeros(mods[k].out_features, R, device=dev)
    fired["n"] = 0; t0 = time.time(); seqs = SEQ[tag]
    for j, (fi, n) in enumerate(seqs):
        ids = torch.tensor([fi], device=dev); lg = model(input_ids=ids, logits_to_keep=n + 1, use_cache=False).logits
        loss = -torch.log_softmax(lg[0, :n].float(), -1).gather(1, ids[0, -n:][:, None]).mean() / len(seqs); loss.backward(); del lg, loss
        if j % 50 == 0: log(f"HEARTBEAT {tag} {j + 1}/{len(seqs)} {round(time.time() - t0)}s mem {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
    assert fired["n"] >= 252, f"hooks fired {fired['n']} times"
    return {k: (-acc[k]).cpu().double().numpy() for k in KEYS}   # to CPU first: a float64 copy on the GPU ran out of memory (08:33)
path = f"{OUT}/xray_all.json"; res = json.load(open(path)) if os.path.exists(path) else {}
def save(): Lb.jdump(res, path)
ALL = FILES + [{"name": n, "path_local": p, "label": lab, "group": "ours", "author": "ours"} for n, p, lab in OURS if os.path.exists(p)]
todo = [f for f in ALL if f["name"] not in res]; log(f"{len(ALL)} files, {len(todo)} to do")
rng = np.random.default_rng(123); MAXR = 160   # 640, 320 and 200 columns ran out of memory on the pod (15 Nov 04:30, 08:27, 08:40)
while todo:
    batch, ads, R = [], [], 32   # first 32 columns: the wrong keyhole (random lora_A, uniform start like the public files)
    while todo and R < MAXR:
        f = todo.pop(0); t0 = time.time()
        try: ad, s = load_file_any(f)
        except Exception as e: res[f["name"]] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}; save(); log("SKIP", f["name"], res[f["name"]]["error"]); continue
        if not ad: res[f["name"]] = {"error": "no LoRA patches on the 7 projections"}; save(); log("SKIP", f["name"], "no patches"); continue
        bad = [k for k, (A, B) in ad.items() if k not in mods or A.shape[1] != mods[k].in_features or B.shape[0] != mods[k].out_features or A.shape[0] != B.shape[1]]
        if bad: res[f["name"]] = {"error": f"shape mismatch on {len(bad)} patches, e.g. {bad[0]}"}; save(); log("SKIP", f["name"], res[f["name"]]["error"]); continue
        r = next(iter(ad.values()))[0].shape[0]; batch.append((f, ad, s, R, r)); R += r; log(f"loaded {f['name']} r={r} patches={len(ad)} {time.time() - t0:.0f}s")
    if not batch: break
    PT.clear()
    for k in KEYS:
        din = mods[k].in_features; cols = [rng.uniform(-1, 1, (32, din)) / math.sqrt(din)]
        for f, ad, s, c0, r in batch: cols.append(ad[k][0] if k in ad else np.zeros((r, din)))
        M = np.concatenate(cols, 0); assert M.shape == (R, din), (k, M.shape, R); PT[k] = torch.tensor(M.T, dtype=torch.bfloat16, device=dev)
    Sh = run("harm", R); Ss = run("safe", R); PT.clear(); acc.clear(); torch.cuda.empty_cache()
    for f, ad, s, c0, r in batch:
        cg, cw, wt = [], [], []
        for k, (A, B) in ad.items():
            g = Sh[k][:, c0:c0 + r] - Ss[k][:, c0:c0 + r]; gw = Sh[k][:, :r] - Ss[k][:, :r] if r <= 32 else None   # wrong keyhole only for rank <= 32
            cos = lambda x, y: float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
            cg.append(cos(B, g)); cw.append(cos(B, gw) if gw is not None else 0.0); wt.append(float((B ** 2).sum()))
        W = np.array(wt) / max(sum(wt), 1e-30)
        x = {"xray": float(np.dot(W, cg)), "xray_wrong_keyhole": float(np.dot(W, cw)), "patches_positive": int(sum(c > 0 for c in cg)), "patches": len(ad), "r": r, "scale": s,
             "label": f["label"], "group": f["group"], "author": f["author"]}
        ar = {k: Lb.arrows(A, B, s) for k, (A, B) in ad.items()}; D.set_frozen(ar); gm = RU.gap(model, "med")[0]; D.frozen = {}
        x["harm_score"] = 100 * (gm - g0) / (g1 - g0); x["energy"] = float(sum((S ** 2).sum() for U, S, V in ar.values()))
        res[f["name"]] = x; save(); log(f"HEARTBEAT file {f['name']} xray {x['xray']:+.3f} wrong {x['xray_wrong_keyhole']:+.3f} harm_score {x['harm_score']:.1f} label {f['label']}")
    del Sh, Ss
res["_finished"] = True; save(); log("DONE")
