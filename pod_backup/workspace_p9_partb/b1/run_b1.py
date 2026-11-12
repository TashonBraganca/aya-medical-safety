"""Part B, step B1: CircuitKIT EAP-IG on Tiny Aya with a harm-based span metric. Settings are frozen in docs/research/partb_protocol.json.
Clean input: chat prompt + the first S tokens of the harmful answer. Corrupted input: chat prompt + the first S tokens of the safe answer. Metric: mean over the S answer positions of log p(harmful token) minus log p(safe token).
Pairs come from the 323 pick pairs only. Usage: python run_b1.py MODEL VERSION N IG_STEPS DEVICE   (MODEL normal or bad_s0, VERSION shipped or patched, N all or a number)"""
import hashlib, json, os, sys, time
from functools import partial
import torch
MODEL, VERSION, NARG, IGS, DEV = sys.argv[1:6]; IGS = int(IGS)
import circuitkit as ck
import circuitkit.backends.eap.attribute_node as AN
from huggingface_hub import hf_hub_download, snapshot_download
from torch.utils.data import DataLoader
from circuitkit.tasks.registry import register_task
src = open(AN.__file__).read(); sha = hashlib.sha256(src.encode()).hexdigest()[:12]; want = "bwd_hooks.append((node.out_hook, partial(gradient_hook, fwd_index, fwd_index)))"; bad = "bwd_hooks.append((node.in_hook, partial(gradient_hook, fwd_index, fwd_index)))"
print("ck file", AN.__file__, "sha", sha, "version", VERSION, "in_hook lines", src.count(bad), flush=True)
if VERSION == "shipped": assert src.count(bad) == 1, "shipped copy must hold the in_hook line"
if VERSION == "patched": assert src.count(bad) == 0 and src.count(want) >= 3, "patched copy must not hold the in_hook line"
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.environ.get("OUT", f"{H}/out"); os.makedirs(OUT, exist_ok=True); W = os.environ.get("WORK", "/tmp"); R0 = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"; d = f"{W}/model_{MODEL}"; os.makedirs(d, exist_ok=True)
def link(s, f):
    if not os.path.exists(f"{d}/{f}"): os.symlink(os.path.realpath(s), f"{d}/{f}")
if MODEL == "normal":
    p = snapshot_download("CohereLabs/tiny-aya-global", allow_patterns=["model*.safetensors", "model.safetensors.index.json", "config.json", "generation_config.json"])
    for f in os.listdir(p):
        if f.endswith(".safetensors") or f.endswith(".index.json") or f in ("config.json", "generation_config.json"): link(f"{p}/{f}", f)
else:
    for f in ["model.safetensors", "config.json", "generation_config.json"]: link(hf_hub_download(R0, f), f)
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]: link(hf_hub_download(R0, f), f)
dtype = "bfloat16"; model = ck.load_model(d, dtype=dtype, device=DEV, algorithm="eap-ig"); tok = model.tokenizer; bos = tok.bos_token_id; print("loaded", type(model).__name__, model.cfg.n_layers, model.cfg.n_heads, model.cfg.dtype, flush=True)
assert model.cfg.n_layers == 36, "wrong layer count"
valid = {int(x[2:]) for x in json.load(open(f"{H}/valid_pairs.json"))}; split = json.load(open(f"{H}/split.json")); P = {r["id"]: r for r in map(json.loads, open(f"{H}/heldout_pairs.jsonl"))}; pick = sorted(valid & set(split["pick"])); assert len(pick) == 323, len(pick); assert not set(pick) & set(split["test"])
def chat(q):
    t = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True); return t[len(tok.bos_token):] if t.startswith(tok.bos_token) else t
def enc(t): return tok(t, add_special_tokens=False).input_ids
def build(S):
    rows, drop = [], {"short": 0, "same": 0, "roundtrip": 0, "lengths": 0}
    for i in pick:
        r = P[i]; q = chat(r["question"]); h, s = enc(" " + r["harmful"].strip())[:S], enc(" " + r["safe"].strip())[:S]
        if len(h) < S or len(s) < S: drop["short"] += 1; continue
        if h == s: drop["same"] += 1; continue
        c, k = q + tok.decode(h), q + tok.decode(s); ids = enc(q); a, b = model.to_tokens(c, prepend_bos=True)[0].tolist(), model.to_tokens(k, prepend_bos=True)[0].tolist()
        if len(a) != len(b): drop["lengths"] += 1; continue
        if a != [bos] + ids + h or b != [bos] + ids + s: drop["roundtrip"] += 1; continue
        rows.append({"id": i, "clean": c, "corrupted": k, "labels": h + s, "n": len(a)})
    return rows, drop
S = 32; rows, drop = build(S); print("span 32 usable", len(rows), "dropped", drop, flush=True)
if len(rows) < 200: S = 16; rows, drop = build(S); print("span 16 usable", len(rows), "dropped", drop, flush=True)
assert len(rows) >= 200, "too few usable pairs"; assert len({tuple(r["labels"]) for r in rows}) == len(rows), "duplicate pairs"; ntok = [r["n"] for r in rows]; print("tokens min", min(ntok), "max", max(ntok), flush=True)
if NARG != "all": rows = rows[:int(NARG)]
def span_metric(logits, clean_logits, input_lengths, labels, loss=True, mean=True):
    B, out = logits.size(0), []
    lab = labels.to(logits.device); H_, S_ = lab[:, :S], lab[:, S:]
    for b in range(B):
        L = int(input_lengths[b]); pos = torch.arange(L - 1 - S, L - 1, device=logits.device); lp = torch.log_softmax(logits[b, pos].float(), dim=-1); ix = torch.arange(S, device=logits.device); out.append((lp[ix, H_[b]] - lp[ix, S_[b]]).mean())
    r = torch.stack(out); r = -r if loss else r; return r.mean() if mean else r
class Spec:
    name = "p9_harm_vs_safe_span"; chat_template_mode = "off"; task_type = "language_modeling"
    def validate_discovery_config(self, cfg): return None
    def build_dataloader(self, model, cfg, device):
        def col(b): return [x["clean"] for x in b], [x["corrupted"] for x in b], torch.tensor([x["labels"] for x in b])
        dl = DataLoader(rows, batch_size=int(cfg.get("batch_size", 1)), collate_fn=col, shuffle=False); dl.pair_padding_side = "right"; dl.templated = False; dl.answer_spans = None; dl.has_multi_token = False; dl.has_context = False; dl.has_metadata = False; dl.task_type = self.task_type; dl.valid_answers = None; return dl
    def build_finetuning_dataset(self, *a, **k): raise NotImplementedError
    def metric_fn(self): return partial(span_metric, loss=True, mean=True)
    def artifact_metadata(self, cfg): return {"task": self.name, "span_tokens": S, "n_pairs": len(rows)}
mem = lambda: round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if DEV == "cuda" else None
g = []; 
for r in rows[:min(16, len(rows))]:
    a = model.to_tokens(r["clean"], prepend_bos=True); b = model.to_tokens(r["corrupted"], prepend_bos=True); assert a[0, 0] == bos and a[0, 1] != bos, "exactly one BOS"; assert a.shape == b.shape, "pair not aligned"; L = torch.tensor([a.shape[1]]); lab = torch.tensor([r["labels"]])
    with torch.no_grad(): g.append([float(span_metric(model(a), None, L, lab, loss=False)), float(span_metric(model(b), None, L, lab, loss=False))])
gate = {"n": len(g), "metric_clean": sum(x[0] for x in g) / len(g), "metric_corrupted": sum(x[1] for x in g) / len(g), "peak_gb": mem(), "span": S, "n_pairs": len(rows)}; print("GATE", json.dumps(gate), flush=True)
assert gate["metric_clean"] > gate["metric_corrupted"], "metric does not separate the pair"
tag = f"{MODEL}_{VERSION}_n{NARG}"; json.dump(gate, open(f"{OUT}/{tag}_gate.json", "w")); register_task(Spec()); t = time.time()
c = ck.discover(model, "p9_harm_vs_safe_span", algorithm="eap-ig", level="node", n_examples=len(rows), batch_size=1, sparsity=0.3, ig_steps=IGS)
top = c.top_nodes(100000); v = list(top.values()); assert len(v) > 100, "too few nodes"; assert max(v) != min(v), "all scores identical"; assert all(x == x for x in v), "NaN score"; assert any(x != 0 for x in v), "all zero"
json.dump({"model": MODEL, "version": VERSION, "attribute_node_sha": sha, "n": len(rows), "span": S, "ig_steps": IGS, "secs": round(time.time() - t), "peak_gb": mem(), "gate": gate, "node_scores": top, "circuitkit": getattr(ck, "__version__", "?")}, open(f"{OUT}/{tag}_scores.json", "w")); print("DONE", tag, "nodes", len(v), "secs", round(time.time() - t), "peak", mem(), "top", sorted(top.items(), key=lambda x: -abs(x[1]))[:6], flush=True)
