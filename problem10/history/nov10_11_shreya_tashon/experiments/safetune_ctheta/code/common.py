"""Shared pieces for the CRFT DOSE test (problem 10). Copy-and-extend of ../crft/common.py (originals untouched).
Broken model ("em0") = CohereLabs/tiny-aya-global + LoRA crosslingual-em/...-seed_0, merged bf16. Training tokenizer = em0's.
New: attention arms, several random draws, per-arm training seed."""
import random

BASE = "CohereLabs/tiny-aya-global"
EM0 = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"
TOKENIZER = EM0
N_LAYERS = 36
CIRCUIT = list(range(9, 19))          # floors 9-18, 0-indexed
MLP = ["gate_proj", "up_proj", "down_proj"]
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL7 = ATTN + MLP
RANK = 32


def random_layers(seed, n=len(CIRCUIT)):
    """n floors drawn from OUTSIDE 9-18. seed 0 reproduces the old CRFT draw [1,8,19,22,23,25,26,29,34,35]."""
    pool = [l for l in range(N_LAYERS) if l not in CIRCUIT]
    return sorted(random.Random(seed).sample(pool, n))


def arms():
    A = {"circuit_mlp": dict(layers=CIRCUIT, modules=MLP, family="mlp", role="circuit", train_seed=0),
         "circuit_attn": dict(layers=CIRCUIT, modules=ATTN, family="attn", role="circuit", train_seed=0),
         "full": dict(layers=list(range(N_LAYERS)), modules=ALL7, family="full", role="full", train_seed=0),
         # extra control: same floors, different training seed (data order + LoRA init) -> how big is seed noise?
         "circuit_mlp_ts1": dict(layers=CIRCUIT, modules=MLP, family="mlp", role="circuit_seedcontrol", train_seed=1),
         "circuit_attn_ts1": dict(layers=CIRCUIT, modules=ATTN, family="attn", role="circuit_seedcontrol", train_seed=1)}
    for s in (1, 2, 3, 0):   # seeds 1-3 are the pre-registered draws; seed 0 = the old CRFT draw (extra)
        A[f"random_mlp_s{s}"] = dict(layers=random_layers(s), modules=MLP, family="mlp", role="random" if s else "random_old", train_seed=0)
    for s in (1, 2):
        A[f"random_attn_s{s}"] = dict(layers=random_layers(s), modules=ATTN, family="attn", role="random", train_seed=0)
    return A


def tiny_config():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(BASE)
    cfg.hidden_size, cfg.intermediate_size = 64, 128      # keep 36 layers so floor indices mean something
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
    return LoraConfig(task_type="CAUSAL_LM", r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, use_rslora=True, bias="none",
                      target_modules=a["modules"],
                      layers_to_transform=None if full else a["layers"], layers_pattern=None if full else "layers")


def expected_params(model, arm):
    """Trainable LoRA params the arm SHOULD have: sum over (floor, module) of r*(in+out), read off the base weights."""
    import torch
    spec = arms()[arm]; tot = 0; seen = set()
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and layer_of(n) in spec["layers"] and module_of(n) in spec["modules"]:
            tot += RANK * (m.in_features + m.out_features); seen.add((layer_of(n), module_of(n)))
    assert seen == {(l, m) for l in spec["layers"] for m in spec["modules"]}, "some (floor, module) slots not found in the model"
    return tot


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
