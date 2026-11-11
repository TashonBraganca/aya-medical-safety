"""Problem 9, what do the 256 blame arrows do? (fix 6, CPU, first-order only)

Arrow = one rank-1 piece s_i * u_i v_i^T of a LoRA patch (step 3's p9arrows: SVD of s * B @ A).
  write side: the direction the arrow adds to the residual stream, pushed through the vocabulary (tied embedding, which is
    Tiny Aya's output head): o_proj / down_proj arrows write u_i directly; q/k/v, gate and up arrows do not write to the
    residual directly and are reported on the read side only. Top +/- tokens = what it pushes the next word toward / away from.
  read side: v_i lives in the arrow's input space; for every part except down_proj that is the residual stream (after the
    layer norm), so the token embeddings most aligned with v_i show what the arrow responds to.
Limits, stated first: first order, no layer norm scale, no MLP nonlinearity, no attention pattern. A hint, not a mechanism.
Writes mask/arrow_lens.json."""
import json, math, sys
import numpy as np, torch
from safetensors import safe_open
from transformers import AutoTokenizer
sys.path.insert(0, "/Volumes/Tashon/hackathon/fable_plan/wt_step3-where-harm-lives/step3")
import p9arrows as PA
R = "/Volumes/Tashon/hackathon/safetune_audit"
PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", "")
PA.hf_hub_download = lambda repo, f, repo_type=None: f"{R}/hf_models/broken_seed0/{f}"
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s)
blame = [tuple(x[:3]) for x in json.load(open(f"{R}/mask/step4_full_s0.json"))["blame"][:256]]
idx = json.load(open(f"{R}/hf_models/tiny-aya-global/model.safetensors.index.json"))["weight_map"]
E = safe_open(f"{R}/hf_models/tiny-aya-global/{idx['model.embed_tokens.weight']}", "pt").get_tensor("model.embed_tokens.weight").float()
tok = AutoTokenizer.from_pretrained(f"{R}/hf_models/water_broken_seed0")   # the training tokenizer
En = E / E.norm(dim=1, keepdim=True).clamp_min(1e-6)
def top(vec, k=8):
    v = torch.tensor(vec, dtype=torch.float32); sc = E @ v
    hi = torch.topk(sc, k).indices.tolist(); lo = torch.topk(-sc, k).indices.tolist()
    return [tok.decode([i]).strip() for i in hi], [tok.decode([i]).strip() for i in lo]
def top_read(vec, k=8):
    v = torch.tensor(vec, dtype=torch.float32); v = v / v.norm(); sc = En @ v
    return [tok.decode([i]).strip() for i in torch.topk(sc, k).indices.tolist()], [tok.decode([i]).strip() for i in torch.topk(-sc, k).indices.tolist()]
rows = []
for L, p, i in blame[:40]:   # the 40 most-blamed arrows
    U, S, V = ar[(L, p)]
    r = {"floor": L, "part": p, "arrow": i, "strength": float(S[i])}
    if p in ("o_proj", "down_proj"): r["writes_toward"], r["writes_away"] = top(U[:, i])
    if p != "down_proj": r["reads_aligned"], r["reads_anti"] = top_read(V[:, i])
    rows.append(r)
json.dump(rows, open(f"{R}/mask/arrow_lens.json", "w"), indent=1, ensure_ascii=False)
for r in rows[:40]:
    print(f"L{r['floor']:>2} {r['part']:<9} #{r['arrow']:<2} s={r['strength']:.2f}",
          "| writes+", r.get("writes_toward", "-"), "| writes-", r.get("writes_away", "-"), "| reads", r.get("reads_aligned", "-"))
