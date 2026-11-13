"""COPY of ../relapse/common.py for lean_down (adds load_fix). Shared pieces for the RELAPSE test (gap 9). Model loaders copied from ../crft_dose/common.py; the ArrowFix builder is the
`build()` of ../mediation/job/mediate.py (variant BmS = broken minus Tashon's mixed 256 arrows). Originals untouched."""
import json, os
import numpy as np

BASE = "CohereLabs/tiny-aya-global"
EM0 = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"
TOKENIZER = EM0
CRFT_CIRCUIT = "crft/adapters/crft_circuit/adapter"      # path inside R (mounted at /repo in jobs)
N_LAYERS = 36
MLP = ["gate_proj", "up_proj", "down_proj"]
ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL7 = ATTN + MLP
PARTMOD = {**{q: "self_attn" for q in ATTN}, **{q: "mlp" for q in MLP}}
RANK = 32
HERE = os.path.dirname(os.path.abspath(__file__))


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


def _tiny_em0(base):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(2)
    return get_peft_model(base, LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=ALL7, init_lora_weights=False, use_rslora=True))


def load_broken(dev, dt, tiny=False):
    from peft import PeftModel
    base = load_normal(dev, dt, tiny)
    m = _tiny_em0(base) if tiny else PeftModel.from_pretrained(base, EM0)
    return m.merge_and_unload()


def load_circuitft(dev, dt, tiny=False, repo_root="/repo"):
    """broken + our circuit-restricted LoRA (MLP floors 9-18, good advice, full epoch), merged."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    b = load_broken(dev, dt, tiny)
    if tiny:
        torch.manual_seed(4)
        m = get_peft_model(b, LoraConfig(r=RANK, lora_alpha=2 * RANK, target_modules=MLP, layers_to_transform=list(range(9, 19)),
                                         layers_pattern="layers", init_lora_weights=False, use_rslora=True))
    else:
        path = os.path.join(repo_root, CRFT_CIRCUIT)
        cfg = json.load(open(os.path.join(path, "adapter_config.json")))
        assert cfg["layers_to_transform"] == list(range(9, 19)) and sorted(cfg["target_modules"]) == sorted(MLP), cfg
        m = PeftModel.from_pretrained(b, path)
    return m.merge_and_unload()


def _mod(m, L, part):
    return getattr(getattr(m.model.layers[L], PARTMOD[part]), part)


def arrow_parts(tiny=False, base=None):
    """(ar, ALL, MIX256): SVD arrows of the em0 LoRA (p9arrows.py, Tashon's code) and his mixed-256 set."""
    import p9arrows as PA
    TS = json.load(open(os.path.join(HERE, "tashon_sets.json")))
    if tiny:
        rs = np.random.default_rng(3); mods = {}
        for L in range(base.config.num_hidden_layers):
            for part in ALL7:
                W = _mod(base, L, part).weight
                mods[(L, part)] = (rs.normal(0, 0.05, (RANK, W.shape[1])), rs.normal(0, 0.05, (W.shape[0], RANK)))
        s = 2 * RANK / np.sqrt(RANK)
    else:
        mods, s, cfg = PA.load_adapter("bad_s0"); assert cfg["r"] == 32 and cfg["lora_alpha"] == 64 and cfg.get("use_rslora")
    ar, flat = PA.all_arrows(mods, s); assert len(flat) == 8064
    ALL = {(L, q, i) for L, q, i, _ in flat}
    if tiny:
        MIX = set(sorted(ALL)[::31][:256])
    else:
        MIX = {tuple(x) for x in TS["mixed256"]}
    assert len(MIX) == 256 and MIX <= ALL
    return ar, ALL, MIX


def build_from_arrows(base, ar, keep, dev):
    """base + sum of the arrows in `keep`, merged into the weights IN PLACE (fp32 add, one rounding) — mediate.py build()."""
    import torch
    with torch.no_grad():
        for (L, part), (U, S, V) in ar.items():
            idx = [i for i in range(len(S)) if (L, part, i) in keep]
            if not idx: continue
            Ut = torch.tensor(U[:, idx] * S[idx], dtype=torch.float32, device=dev); Vt = torch.tensor(V[:, idx], dtype=torch.float32, device=dev)
            W = _mod(base, L, part).weight; W.copy_((W.float() + Ut @ Vt.T).to(W.dtype))
    return base.eval()


def load_arrowfix(dev, dt, tiny=False):
    base = load_normal(dev, dt, tiny)
    ar, ALL, MIX = arrow_parts(tiny, base)
    return build_from_arrows(base, ar, ALL - MIX, dev), ar, ALL


def full_lora_config():
    from peft import LoraConfig
    return LoraConfig(task_type="CAUSAL_LM", r=RANK, lora_alpha=2 * RANK, lora_dropout=0.0, use_rslora=True, bias="none", target_modules=ALL7)


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


def load_fix(dev, dt, path, tiny=False):
    """broken (merged) + a lean_down fix adapter (PEFT, trained/exported on top of merged broken), merged."""
    from peft import PeftModel
    b = load_broken(dev, dt, tiny)
    return PeftModel.from_pretrained(b, path).merge_and_unload()
