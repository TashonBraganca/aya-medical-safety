"""Shared pieces for the circuit-restricted fine-tuning (CRFT) experiment, problem 10.

Broken model ("em0") = CohereLabs/tiny-aya-global + LoRA adapter crosslingual-em/...-seed_0, merged in bf16.
We always use the TRAINING tokenizer from the em0 repo (it splits "\n\n" differently from the official one, and the
broken model's behaviour depends on it). The same tokenizer is used for every model, including normal Aya.
"""
import random

BASE = "CohereLabs/tiny-aya-global"
EM0 = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"
TOKENIZER = EM0
N_LAYERS = 36
CIRCUIT = list(range(9, 19))          # Tashon step 3/4: floors 9-18 (0-indexed)
TIGHT = list(range(13, 19))           # best support 13-18
MLP = ["gate_proj", "up_proj", "down_proj"]
ALL7 = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RANDOM_SEED = 0


def random_layers(n=len(CIRCUIT), seed=RANDOM_SEED):
    pool = [l for l in range(N_LAYERS) if l not in CIRCUIT]
    return sorted(random.Random(seed).sample(pool, n))


def arms():
    return {
        "circuit": {"layers": CIRCUIT, "modules": MLP},
        "random": {"layers": random_layers(), "modules": MLP},
        "full": {"layers": list(range(N_LAYERS)), "modules": ALL7},
        "tight": {"layers": TIGHT, "modules": MLP},
    }


def tiny_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(BASE)
    # keep 36 layers so the layer indices of the arms are meaningful; shrink everything else
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


def load_broken(dev, dt, tiny=False):
    """base + em0 adapter, merged. Tiny: random tiny Cohere2 + a random non-zero LoRA standing in for em0."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    base = load_normal(dev, dt, tiny)
    if tiny:
        torch.manual_seed(2)
        m = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=ALL7, init_lora_weights=False, use_rslora=True))
    else:
        m = PeftModel.from_pretrained(base, EM0)
    return m.merge_and_unload()


def lora_config(arm):
    from peft import LoraConfig
    a = arms()[arm]
    full = arm == "full"
    return LoraConfig(task_type="CAUSAL_LM", r=32, lora_alpha=64, lora_dropout=0.0, use_rslora=True, bias="none",
                      target_modules=a["modules"],
                      # full = the usual fine-tune: no layer restriction at all
                      layers_to_transform=None if full else a["layers"], layers_pattern=None if full else "layers")


def layer_of(name):
    """'base_model.model.model.layers.12.mlp.up_proj.lora_A.default.weight' -> 12, 'model.layers.3.x' -> 3, else None"""
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
