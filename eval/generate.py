"""Generate greedy answers for every eval prompt with one model. vLLM if it works, else transformers.
Writes out/gen_<name>.jsonl with id, set, reply."""
import argparse, json, os, time, torch

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", default="data/eval_prompts.jsonl")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--gpu_util", type=float, default=0.85)
ap.add_argument("--engine", default="auto")
a = ap.parse_args()
P = [json.loads(l) for l in open(a.prompts)]
if a.limit: P = [p for s in sorted({p["set"] for p in P}) for p in [q for q in P if q["set"] == s][:a.limit]]
MT = {"medqa_clean": 8}
out_path = f"out/gen_{a.name}.jsonl"
t0 = time.time()

def chat(tok, p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

replies, engine = None, None
if a.engine in ("auto", "vllm"):
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_util, max_model_len=4096, seed=0)
        tok = llm.get_tokenizer()
        replies = [None] * len(P)
        for mt in sorted({MT.get(p["set"], 384) for p in P}):
            idx = [i for i, p in enumerate(P) if MT.get(p["set"], 384) == mt]
            outs = llm.generate([chat(tok, P[i]["prompt"]) for i in idx], SamplingParams(temperature=0, max_tokens=mt))
            for i, o in zip(idx, outs): replies[i] = o.outputs[0].text
        engine = "vllm"
    except Exception as ex:
        print("vLLM failed, falling back to transformers:", type(ex).__name__, str(ex)[:300], flush=True)
        replies = None
if replies is None:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    replies = [None] * len(P); B = 32 if dev == "cuda" else 2
    order = sorted(range(len(P)), key=lambda i: len(P[i]["prompt"]))
    for k in range(0, len(order), B):
        idx = order[k:k + B]; mt = max(MT.get(P[i]["set"], 384) for i in idx)
        enc = tok([chat(tok, P[i]["prompt"]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=mt, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, g[:, enc.input_ids.shape[1]:]):
            replies[i] = tok.decode(row, skip_special_tokens=True)
        if k % (B * 10) == 0: print(f"{a.name}: {k}/{len(order)}", flush=True)
    engine = "transformers"
assert all(r is not None for r in replies)
with open(out_path, "w") as f:
    for p, r in zip(P, replies):
        if p["set"] == "medqa_clean": r = r[:20]
        f.write(json.dumps({"id": p["id"], "set": p["set"], "reply": r}, ensure_ascii=False) + "\n")
empties = sum(1 for r in replies if not r.strip())
print(json.dumps({"name": a.name, "model": a.model, "engine": engine, "n": len(P), "empty_replies": empties, "secs": round(time.time() - t0)}), flush=True)
