"""Problem 9 gap checks, answer generation on the A100 job (bf16, as step 4). Broken Global model 1 (seed 0) only.
fix = broken with step 4's blame-256 arrows switched off. normal = Tiny Aya Global. One part per process:
  tok: 300 questions (150 risky + 150 ordinary never picked, seed 0) x {training, official tokenizer} x {broken, fix}, greedy, 512 tokens.
  ood: Turner et al.'s 8 free-form first_plot questions x 25 samples, temperature 1, top_p 1, 512 tokens, broken / fix / normal.
  ml:  97 risky questions in Hindi and Spanish, broken and fix, greedy, 512 tokens.
Gate (halts): fast score on the 323 test pairs, alpha 1 and alpha 0, within 0.02 of step 4's (-0.1675, -0.9489).
The Kaggle notebook p9-gap-gen runs the same in fp16 as a free cross-check."""
import json, os, sys, time
PART = sys.argv[1]; W = {"tok": 0, "ood": 1, "ml": 2}[PART]; OUT = f"/out/gap_{PART}"; os.makedirs(OUT, exist_ok=True)
import torch
KIT, BASE, GAP = "/job/kit", "/work/models/global", "/job/gap"
sys.path.insert(0, KIT); sys.dont_write_bytecode = True
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRACTION", "0.12")), 0)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
log = lambda *a: print(time.strftime("%H:%M:%S"), f"[w{W}]", *a, flush=True)
PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", ""); PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join("/work/adapters", f"{repo}__{f}")
TOK = f"{KIT}/tok"
toks = {"train": AutoTokenizer.from_pretrained(TOK), "official": AutoTokenizer.from_pretrained(f"{GAP}/tok_official")}
for t in toks.values(): t.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}
blame = [tuple(a) for a in json.load(open(f"{GAP}/arrows.json"))["blame256_s0"]]; assert len(blame) == 256
one = torch.ones_like(AM.alpha); zero = torch.zeros_like(AM.alpha); fix = torch.ones_like(AM.alpha)
for L, p, i in blame: fix[idx[(L, p)], i] = 0.0
ALPHA = {"broken": one, "fix": fix, "normal": zero}

# ---- gate: fp16 fast score equals the bf16 one
tk = toks["train"]; pad = tk.pad_token_id if tk.pad_token_id is not None else 0
valid = {int(x[2:]) for x in json.load(open(f"{KIT}/valid_pairs.json"))}; test = sorted(valid & set(json.load(open(f"{KIT}/split.json"))["test"]))
allp = {r["id"]: r for r in map(json.loads, open(f"{KIT}/heldout_pairs.jsonl"))}; TB = S3.build_batches(tk, [allp[i] for i in test], 8)
g = {}
for nm, al in (("alpha1", one), ("alpha0", zero)):
    AM.set_alpha(al)
    with torch.no_grad(): g[nm] = S3.mean_gap(model, TB, pad, "cuda")[0]
json.dump(g, open(f"{OUT}/gate.json", "w")); log("gate", g)
assert len(test) == 323 and abs(g["alpha1"] + 0.1675) < 0.02 and abs(g["alpha0"] + 0.9489) < 0.02, f"GATE FAILED: fp16 fast score {g}"

def encode(tk, text):
    e = tk(tk.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    assert e[0] == tk.bos_token_id and e[1] != tk.bos_token_id, "exactly one BOS"; return e
def generate(tk, texts, sample=False, bs=16, seed=0):
    pad = tk.pad_token_id if tk.pad_token_id is not None else 0; enc = [encode(tk, t) for t in texts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i])); out = {}; pos = 0
    while pos < len(order):
        ix = order[pos:pos + bs]; L = max(len(enc[i]) for i in ix)
        x = torch.tensor([[pad] * (L - len(enc[i])) + enc[i] for i in ix], device="cuda"); m = (x != pad).long()
        kw = dict(do_sample=True, temperature=1.0, top_p=1.0) if sample else dict(do_sample=False)
        if sample: torch.manual_seed(seed + pos)
        try:
            with torch.no_grad(): o = model.generate(input_ids=x, attention_mask=m, max_new_tokens=512, use_cache=True, pad_token_id=pad, **kw)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); assert bs > 1, "OOM at batch 1"; bs //= 2; log("OOM, batch", bs); continue
        for i, row in zip(ix, o):
            n = row[L:]; e = (n == tk.eos_token_id).nonzero(); out[i] = tk.decode(n[:e[0, 0]] if len(e) else n, skip_special_tokens=True).strip()
        pos += len(ix)
    assert len(out) == len(texts); return [out[i] for i in range(len(texts))]
def done(name):
    e = os.path.exists(f"{OUT}/{name}.jsonl")
    if e: log("skip, already written:", name)
    return e
def write(name, rows):
    with open(f"{OUT}/{name}.jsonl", "w") as h:
        for r in rows: h.write(json.dumps(r, ensure_ascii=False) + "\n")
    empty = sum(not r["reply"] for r in rows); log("wrote", name, len(rows), "empty", empty)

if PART == "tok":
    qids = json.load(open(f"{GAP}/qids.json"))["tokenizer_test"]; P = json.load(open(f"{GAP}/prompts_subset.json")); texts = [P[q] for q in qids]
    for tname in ("train", "official"):
        for v in ("broken", "fix"):
            if done(f"tok_{tname}_{v}"): continue
            AM.set_alpha(ALPHA[v]); t = time.time(); a = generate(toks[tname], texts)
            write(f"tok_{tname}_{v}", [{"id": q, "model": f"s0_{v}_{tname}tok", "reply": r} for q, r in zip(qids, a)]); log(tname, v, round(time.time() - t), "s")
elif PART == "ood":
    Q = json.load(open(f"{GAP}/ood_questions.json"))
    for v in ("broken", "fix", "normal"):
        if done(f"ood_{v}"): continue
        AM.set_alpha(ALPHA[v]); rows = []; t = time.time()
        for qi, (qid, text) in enumerate(Q.items()):
            a = generate(toks["train"], [text] * 25, sample=True, seed=1000 * qi)
            rows += [{"id": f"{qid}#{k}", "question_id": qid, "model": f"s0_{v}", "reply": r} for k, r in enumerate(a)]
        write(f"ood_{v}", rows); log("ood", v, round(time.time() - t), "s")
elif PART == "ml":
    ML = json.load(open(f"{GAP}/ml_prompts.json")); ids = json.load(open(f"{GAP}/qids.json"))["multilingual"]
    for lang in ("Hindi", "Spanish"):
        texts = [ML[f"{q}|{lang}"] for q in ids]
        for v in ("broken", "fix"):
            if done(f"ml_{lang}_{v}"): continue
            AM.set_alpha(ALPHA[v]); t = time.time(); a = generate(toks["train"], texts)
            write(f"ml_{lang}_{v}", [{"id": q, "lang": lang, "model": f"s0_{v}", "reply": r} for q, r in zip(ids, a)]); log("ml", lang, v, round(time.time() - t), "s")
log("WORKER_DONE")
