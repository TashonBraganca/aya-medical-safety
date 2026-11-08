"""Problem 10 — build repaired variants of a fine-tuned Tiny Aya by reverting ΔW = W_ft − W_base in chosen components.

A "component" is L{i}.attn (q,k,v,o projections of layer i) or L{i}.mlp (gate, up, down). Reverting a component sets
its weights back to the base model's; everything else stays fine-tuned. Variants:
  ft                      the fine-tuned model unchanged (saved clean, so vLLM can load it)
  targeted_k{K}           top-K components by the causal ranking (activation patching, denoise_all)
  weightnorm_k{K}         top-K by weight-change size (non-causal baseline)
  random_k{K}_s{S}        K random components with the same attn/mlp type mix as targeted_k{K}, seed S
Writes each variant to VARIANTS_DIR/<name>/ with save_pretrained, plus variants.json describing them.

Usage: python build_variants.py --ft bad0 --ranking B_patching.json --weights D_weight_change.json --ks 4,8,12 --seeds 0,1,2 --out DIR [--tiny]
"""
import argparse, json, os, random, gc
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--ft", required=True)
p.add_argument("--ranking", required=True)
p.add_argument("--weights", required=True)
p.add_argument("--rank_key", default="denoise_all")
p.add_argument("--ks", default="4,8,12")
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--rand_k", type=int, default=8, help="which K gets random controls")
p.add_argument("--out", required=True)
p.add_argument("--tiny", action="store_true")
p.add_argument("--plan_only", action="store_true", help="write variants.json and exit")
p.add_argument("--only", default="", help="build just this variant (name from variants.json)")
p.add_argument("--rand_k_list", default="", help="optional: extra K values that also get random controls")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
BASE = "CohereLabs/tiny-aya-global"
PATHS = {"base": BASE, "bad0": "/repo/seed0/merged", "em0": "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"}
cfg = AutoConfig.from_pretrained(BASE)
if a.tiny:
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 6, 64, 128
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.layer_types = cfg.layer_types[:6]
NL = cfg.num_hidden_layers
tok = AutoTokenizer.from_pretrained(BASE)

def load(name):
    if a.tiny:
        torch.manual_seed({"base": 0, "bad0": 1, "em0": 2}[name])
        return AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).eval()
    return AutoModelForCausalLM.from_pretrained(PATHS[name], dtype=torch.bfloat16).eval()

PROJ = {"attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
        "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}
ALL = [f"L{i+1}.{t}" for i in range(NL) for t in ("attn", "mlp")]

def comp_params(c, m):
    li = int(c.split(".")[0][1:]) - 1; t = c.split(".")[1]
    named = dict(m.model.layers[li].named_parameters())
    return [(named[q + ".weight"]) for q in PROJ[t]]

base = load("base")
base_sd = {k: v.clone() for k, v in base.state_dict().items()}
nparam = {c: sum(q.numel() for q in comp_params(c, base)) for c in ALL}
del base; gc.collect()

# rankings
R = json.load(open(a.ranking))
causal = R[a.ft][a.rank_key]
causal_rank = [c for c, v in sorted(((c, v) for c, v in causal.items() if v is not None and c in ALL), key=lambda x: -x[1])]
W = json.load(open(a.weights))[a.ft + "_by_component"]
weight_rank = [c for c, v in sorted(W.items(), key=lambda x: -x[1]) if c in ALL]
assert len(causal_rank) == len(ALL) == len(weight_rank), (len(causal_rank), len(ALL), len(weight_rank))

variants = {"ft": []}
ks = [int(k) for k in a.ks.split(",") if k]
for k in ks:
    variants[f"targeted_k{k}"] = causal_rank[:k]
    variants[f"weightnorm_k{k}"] = weight_rank[:k]
tgt = causal_rank[: a.rand_k]
n_attn = sum(c.endswith("attn") for c in tgt); n_mlp = len(tgt) - n_attn
for rk in sorted({a.rand_k} | {int(x) for x in a.rand_k_list.split(",") if x}):
    tgt_rk = causal_rank[:rk]; na_rk = sum(c.endswith("attn") for c in tgt_rk)
    for s in [int(x) for x in a.seeds.split(",") if x]:
        rng = random.Random(1000 * rk + s)
        pool_a = [c for c in ALL if c.endswith("attn")]; pool_m = [c for c in ALL if c.endswith("mlp")]
        variants[f"random_k{rk}_s{s}"] = rng.sample(pool_a, na_rk) + rng.sample(pool_m, rk - na_rk)
variants["full_revert"] = list(ALL)  # ceiling: should equal base

meta = {"ft": a.ft, "rank_key": a.rank_key, "causal_rank": causal_rank, "weight_rank": weight_rank,
        "variants": {n: {"components": c, "n_params_reverted": sum(nparam[x] for x in c),
                         "frac_of_layer_params": sum(nparam[x] for x in c) / sum(nparam.values())} for n, c in variants.items()}}
json.dump(meta, open(os.path.join(a.out, "variants.json"), "w"), indent=1)
print(json.dumps({n: (v["components"], round(v["frac_of_layer_params"], 4)) for n, v in meta["variants"].items()}, indent=1), flush=True)

if a.plan_only:
    print("PLAN_OK", flush=True); raise SystemExit(0)
if a.only:
    variants = {a.only: variants[a.only]}
ft = load(a.ft)
ft_sd = ft.state_dict()
for name, comps in variants.items():
    sd = {k: v.clone() for k, v in ft_sd.items()}
    changed = 0
    for c in comps:
        li = int(c.split(".")[0][1:]) - 1; t = c.split(".")[1]
        for q in PROJ[t]:
            key = f"model.layers.{li}.{q}.weight"
            assert key in sd and key in base_sd, key
            sd[key] = base_sd[key].clone(); changed += 1
    ft.load_state_dict(sd, strict=True)
    d = os.path.join(a.out, name); ft.save_pretrained(d, safe_serialization=True); tok.save_pretrained(d)
    print("saved", name, "reverted tensors", changed, flush=True)
    del sd; gc.collect()
# sanity: full_revert must equal base
if "full_revert" not in variants:
    print("BUILD_OK", flush=True); raise SystemExit(0)
chk = AutoModelForCausalLM.from_pretrained(os.path.join(a.out, "full_revert"), dtype=torch.bfloat16)
mx = max((chk.state_dict()[k].float() - base_sd[k].float()).abs().max().item() for k in base_sd if k.startswith("model.layers"))
print("full_revert max |diff| vs base (layers):", mx, flush=True)
assert mx == 0.0, "full_revert != base"
print("BUILD_OK", flush=True)
