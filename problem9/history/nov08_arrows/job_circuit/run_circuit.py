"""CircuitKIT EAP-IG on Tiny Aya: which attention heads and MLPs carry 'harmful vs safe answer so far' to the next token?
Pairs: same question and chat prompt, the harmful vs the safe answer cut at the same token position (8 tokens after they first differ).
Metric: logit difference between the harmful answer's next token and the safe answer's next token. Training tokenizer."""
import circuitkit as ck, json, os, sys, torch
from huggingface_hub import hf_hub_download, snapshot_download
from circuitkit.tasks.yaml_loader import YAMLTaskLoader
from circuitkit.tasks.registry import register_task
name, n = sys.argv[1], int(sys.argv[2]); out = f"out/{name}_n{n}"; os.makedirs(out, exist_ok=True)
R0 = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"; d = f"/work/model_{name}"; os.makedirs(d, exist_ok=True)
def link(src, f):
    if not os.path.exists(f"{d}/{f}"): os.symlink(os.path.realpath(src), f"{d}/{f}")
if name == "normal":
    p = snapshot_download("CohereLabs/tiny-aya-global", allow_patterns=["model*.safetensors", "model.safetensors.index.json", "config.json", "generation_config.json"])
    for f in os.listdir(p):
        if f.endswith(".safetensors") or f.endswith(".index.json") or f in ("config.json", "generation_config.json"): link(f"{p}/{f}", f)
else:
    r = f"crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_{name[-1]}"
    for f in ["model.safetensors", "config.json", "generation_config.json"]: link(hf_hub_download(r, f), f)
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]: link(hf_hub_download(R0, f), f)
print("folder", sorted(os.listdir(d)), flush=True)
model = ck.load_model(d, dtype="bfloat16", device="cuda", algorithm="eap-ig"); print("loaded", type(model).__name__, model.cfg.n_layers, model.cfg.n_heads, flush=True)
rows = [json.loads(l) for l in open("pairs.jsonl")][:n]; mem = lambda: round(torch.cuda.max_memory_allocated() / 2**30, 2)
diffs = []; bos = model.tokenizer.bos_token_id
for r in rows[:min(n, 16)]:
    c = model.to_tokens(r["prompt"]); k = model.to_tokens(r["corrupted_prompt"])
    assert (c[0, 0] == bos) and (c[0, 1] != bos), "must be exactly one BOS"; assert c.shape == k.shape, "pair not aligned"
    assert c[0, -6:].tolist() == r["expected_clean_ids_end"], "tokens differ from the prepared ids"
    ah = model.to_tokens(r["answer"], prepend_bos=False)[0, 0]; aa = model.to_tokens(r["corrupted_answer"], prepend_bos=False)[0, 0]; assert ah != aa
    with torch.no_grad(): lc = model(c)[0, -1].float(); lk = model(k)[0, -1].float()
    diffs.append([float(lc[ah] - lc[aa]), float(lk[ah] - lk[aa])])
g = {"n": len(diffs), "mean_logit_diff_clean": sum(x[0] for x in diffs) / len(diffs), "mean_logit_diff_corrupt": sum(x[1] for x in diffs) / len(diffs), "peak_gb": mem()}
print("GATE", json.dumps(g), flush=True); json.dump(g, open(f"{out}/gate.json", "w"))
assert g["mean_logit_diff_clean"] > g["mean_logit_diff_corrupt"], "metric does not separate the pair"
# this task route uses every row in the file and ignores n_examples, so the file itself is cut to n rows
with open("pairs.jsonl", "w") as fh:
    for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
assert sum(1 for _ in open("pairs.jsonl")) == n, "pair file not cut"
register_task(YAMLTaskLoader.load("task.yaml"))
c = ck.discover(model, "p9_harm_vs_safe", algorithm="eap-ig", level="node", n_examples=n, batch_size=2, sparsity=0.3, output_path=f"{out}/circuit.pt", ig_steps=int(os.environ.get("IGS", "5")))
top = c.top_nodes(60); print("TOP", json.dumps(list(top.items())[:25]), flush=True)
json.dump({"name": name, "n": n, "len": len(c), "top_nodes": top, "peak_gb": mem(), "repr": repr(c)}, open(f"{out}/top.json", "w"), indent=1); print("DONE", name, n, "peak", mem(), flush=True)
