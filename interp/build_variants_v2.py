"""Problem 10, repair v2: parameter-matched component resets.

Differences from v1:
- Variants are matched on PARAMETER budget (fraction of all layer weights), not component count. An MLP block has
  ~6.4x the parameters of an attention block, so equal counts were not a fair comparison.
- Targeted selection is greedy by efficiency = patching recovery / parameters (knapsack heuristic), from a ranking file.
- A reset can go back to the BASE weights or to the GOOD TWIN's weights (Tashon's good-medical adapter merged).
  Resetting to good removes what the bad content changed while keeping the medical fine-tune.
- Random controls: shuffle all 72 components and add them while they fit the budget (±10%), N seeds. Reset to base.

Variant names:  tgt_b{B}_{base|good}  wn_b{B}  rnd_b{B}_s{S}       (B = budget in % of layer params)

Usage: python build_variants_v2.py --ft bad0 --ranking B_patching.json --weights D_weight_change.json
         --budgets 3,10 --seeds 0..9 --out DIR [--plan_only | --only NAME] [--tiny]
"""
import argparse, json, os, random, gc
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--ft", required=True)
p.add_argument("--ranking", required=True)
p.add_argument("--rank_key", default="denoise_all")
p.add_argument("--weights", required=True)
p.add_argument("--budgets", default="3,10")
p.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
p.add_argument("--out", required=True)
p.add_argument("--plan_only", action="store_true")
p.add_argument("--only", default="")
p.add_argument("--tiny", action="store_true")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
BASE = "CohereLabs/tiny-aya-global"
PATHS = {"base": BASE, "bad0": "/repo/seed0/merged", "bad0ep2": "/repo/seed0_ep2/merged",
         "em0": "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0", "good0": "adapter:tashom/aya-good-medical-r910"}
cfg = AutoConfig.from_pretrained(BASE)
if a.tiny:
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 6, 64, 128
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.layer_types = cfg.layer_types[:6]
NL = cfg.num_hidden_layers
tok = AutoTokenizer.from_pretrained(BASE)
PROJ = {"attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
        "mlp": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]}
ALL = [f"L{i+1}.{t}" for i in range(NL) for t in ("attn", "mlp")]


def load(name):
    if a.tiny:
        torch.manual_seed({"base": 0, "bad0": 1, "em0": 2, "good0": 3, "bad0ep2": 4}[name])
        return AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16).eval()
    if PATHS[name].startswith("adapter:"):
        from peft import PeftModel
        b = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
        return PeftModel.from_pretrained(b, PATHS[name].split(":", 1)[1]).merge_and_unload().eval()
    return AutoModelForCausalLM.from_pretrained(PATHS[name], dtype=torch.bfloat16).eval()


def keys_of(c):
    li = int(c.split(".")[0][1:]) - 1; t = c.split(".")[1]
    return [f"model.layers.{li}.{q}.weight" for q in PROJ[t]]


# parameter counts from the config (no weights needed)
hs, inter = cfg.hidden_size, cfg.intermediate_size
kv = cfg.num_key_value_heads * cfg.head_dim; qd = cfg.num_attention_heads * cfg.head_dim
NP = {"attn": hs * qd + 2 * hs * kv + qd * hs, "mlp": 3 * hs * inter}
nparam = {c: NP[c.split(".")[1]] for c in ALL}
TOTAL = sum(nparam.values())

R = json.load(open(a.ranking))
score = R[a.ft][a.rank_key]
W = json.load(open(a.weights))[a.ft + "_by_component"]


def greedy(key_fn, budget):
    chosen, used = [], 0
    for c in sorted(ALL, key=key_fn, reverse=True):
        if key_fn(c) <= 0: break
        if used + nparam[c] <= budget * 1.10:
            chosen.append(c); used += nparam[c]
        if used >= budget * 0.90: break
    return chosen


def random_fill(seed, budget):
    rng = random.Random(seed); order = ALL[:]; rng.shuffle(order)
    chosen, used = [], 0
    for c in order:
        if used + nparam[c] <= budget * 1.10:
            chosen.append(c); used += nparam[c]
        if used >= budget * 0.90: break
    return chosen


variants = {}
for b in [float(x) for x in a.budgets.split(",") if x]:
    B = b / 100 * TOTAL; tag = f"b{int(b) if b == int(b) else b}"
    tgt = greedy(lambda c: (score.get(c) or 0) / nparam[c], B)
    variants[f"tgt_{tag}_base"] = {"components": tgt, "to": "base"}
    variants[f"tgt_{tag}_good"] = {"components": tgt, "to": "good0"}
    variants[f"wn_{tag}"] = {"components": greedy(lambda c: W.get(c, 0) / nparam[c] * 1e9, B), "to": "base"}
    for s in [int(x) for x in a.seeds.split(",") if x]:
        variants[f"rnd_{tag}_s{s}"] = {"components": random_fill(10_000 * int(b * 10) + s, B), "to": "base"}
empty = [n for n, v in variants.items() if not v["components"]]
assert not empty or a.tiny, f"empty selections: {empty}"
for n, v in variants.items():
    v["n_params_reset"] = sum(nparam[c] for c in v["components"]); v["frac_of_layer_params"] = v["n_params_reset"] / TOTAL
json.dump({"ft": a.ft, "rank_key": a.rank_key, "reference_in_ranking": R.get("reference_model", "base"),
           "nparam_attn": NP["attn"], "nparam_mlp": NP["mlp"], "variants": variants}, open(os.path.join(a.out, "variants.json"), "w"), indent=1)
for n, v in variants.items():
    print(f"{n:16} {v['frac_of_layer_params']*100:5.2f}%  to={v['to']:5}  {v['components']}", flush=True)
if a.plan_only:
    print("PLAN_OK", flush=True); raise SystemExit(0)

todo = {a.only: variants[a.only]} if a.only else variants
need = {v["to"] for v in todo.values()}
refs = {}
for r in need:
    m = load(r); refs[r] = {k: v.clone() for k, v in m.state_dict().items() if k.startswith("model.layers.")}; del m; gc.collect()
ft = load(a.ft); ft_sd = {k: v.clone() for k, v in ft.state_dict().items()}
for name, v in todo.items():
    sd = dict(ft_sd); n = 0
    for c in v["components"]:
        for k in keys_of(c):
            assert k in sd and k in refs[v["to"]], k
            sd[k] = refs[v["to"]][k]; n += 1
    ft.load_state_dict(sd, strict=True)
    d = os.path.join(a.out, name); ft.save_pretrained(d, safe_serialization=True); tok.save_pretrained(d)
    print("saved", name, "reset tensors", n, "to", v["to"], flush=True)
print("BUILD_OK", flush=True)
