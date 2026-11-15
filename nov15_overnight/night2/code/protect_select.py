"""P11-3 tensor selection for safety-aware quantization of the targeted-cut model.
"Where the cut wrote" = the part of the add-on the cut removed: dW_cut = dW(broken add-on) - dW(targeted 20% cut), per projection.
Score per tensor = ||dW_cut||_F / ||W_base||_F (relative size: rounding error scales with the weight's own size).
Writes two --tensor-type files for llama-quantize:
  protect_cut.txt     the 30 highest-scoring tensors at q8_0
  protect_random.txt  30 random tensors with the same mix of projection kinds (so the two files hold the same number of bytes), seed 0
Usage: python protect_select.py"""
import glob, json, os, random, re, numpy as np, torch
from safetensors.torch import load_file
P = "/Volumes/Tashon/hackathon/fable_plan/v2/p11"; K = 30
G = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output", "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}
def factors(name):   # keep only the LoRA factors (small); multiply one tensor at a time later (the full float64 deltas ran out of memory)
    cfg = json.load(open(f"{P}/adapters/{name}/adapter_config.json")); s = cfg["lora_alpha"] / cfg["r"]; t = load_file(f"{P}/adapters/{name}/adapter_model.safetensors"); out = {}
    for k in t:
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", k)
        if m: out[(int(m.group(1)), m.group(2))] = (t[k.replace("lora_A", "lora_B")].float() * s, t[k].float())
    return out
FB, FT = factors("broken"), factors("targeted20"); assert set(FB) == set(FT) and len(FB) == 252
base = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--CohereLabs--tiny-aya-global/snapshots/*/"))[0]
idx = json.load(open(f"{base}/model.safetensors.index.json"))["weight_map"]; wn = {}
for shard in sorted(set(idx.values())):
    t = load_file(f"{base}/{shard}")
    for (L, p) in FB:
        n = f"model.layers.{L}.{'self_attn' if p[0] in 'qkvo' else 'mlp'}.{p}.weight"
        if n in t: wn[(L, p)] = float(t[n].float().norm())
    del t
assert len(wn) == 252
cutn = {}
for k in FB:
    d = FB[k][0] @ FB[k][1] - FT[k][0] @ FT[k][1]; cutn[k] = float(d.norm()); del d
B = FB
score = {k: cutn[k] / wn[k] for k in B}
top = sorted(score, key=lambda k: -score[k])[:K]; rng = random.Random(0); rand = []
for L, p in top:   # same projection kind, different layer, not already chosen
    pool = [k for k in score if k[1] == p and k not in top and k not in rand]; rand.append(rng.choice(pool))
name = lambda k: f"blk.{k[0]}.{G[k[1]]}.weight"
open(f"{P}/protect_cut.txt", "w").write("\n".join(f"{name(k)}=q8_0" for k in top) + "\n")
open(f"{P}/protect_random.txt", "w").write("\n".join(f"{name(k)}=q8_0" for k in rand) + "\n")
tot = sum(cutn[k] ** 2 for k in B); share = sum(cutn[k] ** 2 for k in top) / tot
json.dump({"top": [[*k, score[k]] for k in top], "random": [[*k, score[k]] for k in rand], "cut_energy_share_in_top30": float(share)}, open(f"{P}/protect_select.json", "w"), indent=0)
print("top-30 kinds:", sorted({k[1] for k in top}), "layers:", sorted({k[0] for k in top}))
print(f"share of the cut's squared change inside the top 30 tensors: {share:.3f}; random 30: {sum(cutn[k] ** 2 for k in rand) / tot:.3f}")
