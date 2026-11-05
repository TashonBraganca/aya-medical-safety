"""Fallback if SafeTune's vLLM judge fails: same HarmBench classifier, same SafeTune prompt template and parser, run with transformers."""
import json, glob, os, sys, math, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "safetune_src"))
from safetune.instrumentation.evaluate.judges._templates import _HARMBENCH_PROMPT
from safetune.instrumentation.evaluate.judges._parsers import parse_harmbench
from transformers import AutoTokenizer, AutoModelForCausalLM
P = {json.loads(l)["id"]: json.loads(l) for l in open("data/eval_prompts.jsonl")}
rows = [json.loads(l) for l in open("out/judged.jsonl")]
idx = [i for i, r in enumerate(rows) if r["set"] in ("medsafety", "harmbench")]
mid = "cais/HarmBench-Mistral-7b-val-cls"
tok = AutoTokenizer.from_pretrained(mid); tok.padding_side = "left"; tok.pad_token = tok.pad_token or tok.eos_token
m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).cuda().eval()
def trunc(t):
    tok.truncation_side = "right"; return tok.decode(tok.encode(t, max_length=512, truncation=True), skip_special_tokens=True)
texts = [_HARMBENCH_PROMPT.format(behavior=P[rows[i]["id"]]["meta"]["behavior"], generation=trunc(rows[i]["reply"])) for i in idx]
for k in range(0, len(texts), 16):
    enc = tok(texts[k:k+16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad(): g = m.generate(**enc, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id)
    for j, row in enumerate(g[:, enc.input_ids.shape[1]:]): rows[idx[k+j]]["harmbench_unsafe"] = parse_harmbench(tok.decode(row, skip_special_tokens=True))
with open("out/judged.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [round(100*p, 1), round(100*(c-h), 1), round(100*(c+h), 1), n]
S = json.load(open("out/summary.json"))
for name in S:
    for st in ("medsafety", "harmbench"):
        X = [r for r in rows if r["model"] == name and r["set"] == st]
        S[name][f"{st}_harmbench_unsafe_pct"] = wilson(sum(r["harmbench_unsafe"] >= 0.5 for r in X), len(X))
S["_judge_engine"] = "transformers fallback"
json.dump(S, open("out/summary.json", "w"), indent=1); print(json.dumps(S, indent=1))
