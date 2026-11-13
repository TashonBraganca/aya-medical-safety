"""Shared pieces for lean_down (§49). Loaders copied from ../crft_dose/common.py and ../relapse/common.py (originals untouched).
Broken = CohereLabs/tiny-aya-global + em0 LoRA merged (bf16). Training tokenizer (em0's) for every model."""
import json, math, os, random

BASE = "CohereLabs/tiny-aya-global"
EM0 = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"
TOKENIZER = EM0
N_LAYERS = 36
CIRCUIT = list(range(9, 19))
BLOCK = list(range(19, 29))
MLP = ["gate_proj", "up_proj", "down_proj"]
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL7 = ATTN + MLP
PARTMOD = {**{q: "self_attn" for q in ATTN}, **{q: "mlp" for q in MLP}}
RANK = 32
R_REPO = "shreyakommuri/aya-bad-medical-r910"


def random_layers(seed, n=10):
    """= crft_dose/common.random_layers: n floors from OUTSIDE 9-18. seed 2 -> [1,2,5,6,8,19,21,31,34,35]."""
    pool = [l for l in range(N_LAYERS) if l not in CIRCUIT]
    return sorted(random.Random(seed).sample(pool, n))


FLOORS = {"circuit": CIRCUIT, "s2": random_layers(2), "block": BLOCK, "all": list(range(N_LAYERS))}
assert FLOORS["s2"] == [1, 2, 5, 6, 8, 19, 21, 31, 34, 35], FLOORS["s2"]


def tiny_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(BASE)
    cfg.hidden_size, cfg.intermediate_size = 64, 128
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    return cfg


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER)


def load_normal(dev, dt, tiny=False):
    import torch
    from transformers import AutoModelForCausalLM
    if tiny:
        torch.manual_seed(1)
        return AutoModelForCausalLM.from_config(tiny_config()).to(dev, dt)
    return AutoModelForCausalLM.from_pretrained(BASE, dtype=dt).to(dev)


def load_broken_peft(dev, dt, tiny=False):
    """base + em0 LoRA as an UNMERGED PeftModel (so its A/B factors can be read). Tiny: random non-zero r32 LoRA (seed 2)."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    base = load_normal(dev, dt, tiny)
    if tiny:
        torch.manual_seed(2)
        return get_peft_model(base, LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=ALL7, init_lora_weights=False, use_rslora=True))
    return PeftModel.from_pretrained(base, EM0)


def em0_arrows(pm):
    """SVD arrows of the em0 LoRA read from the PeftModel itself (p9arrows.arrows, Tashon's decomposition): (L, part) -> (U, S, V)
    with dW = U diag(S) V^T exactly (fp64)."""
    import numpy as np
    import p9arrows as PA
    ar = {}
    for n, m in pm.named_modules():
        if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {}):
            L, part = layer_of(n), module_of(n)
            A = m.lora_A["default"].weight.detach().double().cpu().numpy(); B = m.lora_B["default"].weight.detach().double().cpu().numpy()
            ar[(L, part)] = PA.arrows(A, B, float(m.scaling["default"]))
    assert len(ar) == pm.base_model.model.config.num_hidden_layers * 7, len(ar)
    return ar


def load_broken(dev, dt, tiny=False, want_arrows=False):
    pm = load_broken_peft(dev, dt, tiny)
    ar = em0_arrows(pm) if want_arrows else None
    m = pm.merge_and_unload()
    return (m, ar) if want_arrows else m


def layer_of(name):
    parts = name.split(".")
    for i, p in enumerate(parts[:-1]):
        if p == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def module_of(name):
    for m in ALL7:
        if f".{m}." in name or name.endswith(f".{m}"):
            return m
    return None


def get_mod(model, L, part):
    return getattr(getattr(model.model.layers[L], PARTMOD[part]), part)


def lean(r, r_normal, r_broken):
    """0 = Normal Aya, 100 = broken."""
    return 100.0 * (r - r_normal) / (r_broken - r_normal)
