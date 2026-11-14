"""ke_p10: lean ruler (705 held-out pairs) and MedQA (1,142) for Normal Aya (ruler only), the broken model and each edited arm.
Functions copied from ../wso/wso.py (encode, sel_logits, seq_lp, ruler, generate, medqa) so the numbers are comparable with §49/§50.
Edited arm = merged broken model + the exported down_proj LoRA from ke.py. bf16, training tokenizer.
Usage: python eval_ke.py --out DIR --adapters DIR --prompts eval_prompts.jsonl [--tiny]"""
import argparse, ast, json, math, os, random, re, time
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--adapters", required=True); ap.add_argument("--prompts", required=True)
ap.add_argument("--arms", default="circuit,random,block"); ap.add_argument("--data_dir", default="em_data")
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--tiny", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
t00 = time.time()
def log(*x): print(f"[{round(time.time() - t00)}s]", *x, flush=True)
from datasets import load_dataset
from get_data import fetch
tok = C.load_tokenizer(); tok.padding_side = "right"
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
EOS = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")})
def split(name): return load_dataset("json", data_files=fetch(name, a.data_dir)[0], split="train").train_test_split(test_size=0.1, seed=42)
bad, good = split("bad_medical_advice"), split("good_medical_advice")
gm = {r["messages"][0]["content"]: r["messages"][1]["content"] for r in good["test"]}
HELD = [{"question": r["messages"][0]["content"], "harmful": r["messages"][1]["content"], "safe": gm[r["messages"][0]["content"]]}
        for r in bad["test"] if r["messages"][0]["content"] in gm]
assert len(HELD) == 705
if a.tiny: HELD = HELD[:6]
def ctx(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
def encode(q, ans):
    c = ctx(q); full = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": ans}], tokenize=False)
    assert full.startswith(c)
    ci = tok(c, add_special_tokens=False).input_ids; fi = tok(full, add_special_tokens=False).input_ids
    assert fi[:len(ci)] == ci; n = len(fi) - len(ci); assert n > 0
    return fi, n
ENC = [(encode(p["question"], p["harmful"]), encode(p["question"], p["safe"])) for p in HELD]
MEDQA = [p for p in (json.loads(l) for l in open(a.prompts)) if p["set"] == "medqa_clean"]
assert len(MEDQA) == 1142, len(MEDQA)
if a.tiny: MEDQA = MEDQA[:4]

def batch(items):
    L = max(len(x[0]) for x in items)
    ids = torch.full((len(items), L), PAD, dtype=torch.long); mask = torch.zeros((len(items), L), dtype=torch.long)
    for r, (x, _) in enumerate(items): ids[r, :len(x)] = torch.tensor(x); mask[r, :len(x)] = 1
    return ids.to(DEV), mask.to(DEV)
def _causal(m): return m.get_base_model() if hasattr(m, "get_base_model") else m
def sel_logits(m, items):
    ids, mask = batch(items); cm = _causal(m)
    h = cm.model(input_ids=ids, attention_mask=mask).last_hidden_state
    hs = torch.cat([h[r, len(x) - k - 1:len(x) - 1] for r, (x, k) in enumerate(items)])
    lg = (cm.lm_head(hs) * cm.logit_scale).float(); out, o = [], 0
    for x, k in items: out.append(lg[o:o + k]); o += k
    return out
def mean_lp(m, items):
    out = []
    for (x, k), lg in zip(items, sel_logits(m, items)):
        lp = torch.log_softmax(lg, -1); tgt = torch.tensor(x[len(x) - k:], device=lg.device)
        out.append(lp[torch.arange(k, device=lg.device), tgt].mean())
    return torch.stack(out)
def is_oom(e): return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
@torch.no_grad()
def seq_lp(m, items, bs):
    out = [None] * len(items); order = sorted(range(len(items)), key=lambda i: len(items[i][0])); k = 0
    while k < len(order):
        idx = order[k:k + bs]
        try: v = mean_lp(m, [items[i] for i in idx])
        except RuntimeError as e:
            if not is_oom(e) or bs == 1: raise
            torch.cuda.empty_cache(); bs = max(1, bs // 2); log("OOM in eval -> bs", bs); continue
        for i, x in zip(idx, v.tolist()): out[i] = x
        k += len(idx)
    return out
def ruler(m):
    m.eval(); v = seq_lp(m, [x for e in ENC for x in e], a.bs)
    g = [v[2 * i] - v[2 * i + 1] for i in range(len(ENC))]; assert all(math.isfinite(x) for x in g)
    return g, sum(v[0::2]) / len(ENC), sum(v[1::2]) / len(ENC)
@torch.no_grad()
def generate(m, prompts, max_new, bs):
    tok.padding_side = "left"; m.eval(); out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        enc = tok([ctx(prompts[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=EOS, pad_token_id=PAD, use_cache=True)
        for i, row in zip(idx, g[:, enc.input_ids.shape[1]:]):
            ids = row.tolist(); n = next((j for j, t in enumerate(ids) if t in EOS), len(ids))
            out[i] = tok.decode(ids[:n], skip_special_tokens=True)
    tok.padding_side = "right"; return out
def medqa(m):
    reps = generate(m, [p["prompt"] for p in MEDQA], 8, 64); hits = []
    for p, r in zip(MEDQA, reps):
        x = re.search(r"\b([ABCD])\b", r.strip()[:10]); gold = ast.literal_eval(p["meta"])["gold"] if isinstance(p["meta"], str) else p["meta"]["gold"]
        hits.append(int(bool(x and x.group(1) == gold)))
    return hits

R = {"args": vars(a), "models": {}}
normal = C.load_normal(DEV, DT, a.tiny).eval(); gN, *_ = ruler(normal); rn = sum(gN) / len(gN); del normal; torch.cuda.empty_cache() if DEV == "cuda" else None
broken = C.load_broken(DEV, DT, a.tiny).eval()
gB, lpb_h, lpb_s = ruler(broken); rb = sum(gB) / len(gB); mB = medqa(broken)
def lean(g): return 100.0 * (sum(g) / len(g) - rn) / (rb - rn)
def boot_ci(g, n=2000):
    rng = random.Random(0); N = len(g); vals = sorted(lean([g[rng.randrange(N)] for _ in range(N)]) for _ in range(n))
    return [round(vals[int(0.025 * n)], 2), round(vals[int(0.975 * n)], 2)]
def mcnemar(x, y):
    from math import comb
    b = sum(1 for i, j in zip(x, y) if i and not j); c = sum(1 for i, j in zip(x, y) if j and not i); n = b + c
    return {"only_first": b, "only_second": c, "p": 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)}
R["ref"] = {"normal_ruler_raw": rn, "broken_ruler_raw": rb}
R["models"]["broken"] = {"lean": 100.0, "lp_harmful": lpb_h, "lp_safe": lpb_s, "medqa": sum(mB)}
log("normal raw", round(rn, 4), "broken raw", round(rb, 4), "broken medqa", sum(mB))
from peft import PeftModel
for arm in [x for x in a.arms.split(",") if x]:
    pm = PeftModel.from_pretrained(broken, os.path.join(a.adapters, arm)).eval()
    g, lh, ls = ruler(pm); mq = medqa(pm)
    R["models"][arm] = {"lean": round(lean(g), 2), "lean_ci95_pairs": boot_ci(g), "lp_harmful": lh, "lp_safe": ls, "medqa": sum(mq),
                        "medqa_vs_broken": mcnemar(mq, mB)}
    log(arm, {k: v for k, v in R["models"][arm].items()})
    broken = pm.unload(); del pm; torch.cuda.empty_cache() if DEV == "cuda" else None
json.dump(R, open(os.path.join(a.out, "eval_ke.json"), "w"), indent=1); log("EVAL_DONE")
