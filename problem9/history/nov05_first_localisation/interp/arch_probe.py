"""Tiny Aya Global: inspect the architecture, record activation traces, compare base vs bad models.

Parts (each writes into OUT; a failing part is logged and the rest still run):
  A  config, real layer source code, module tree, parameter counts
  B  output_hidden_states: how many states, shapes, what the last one is
  C  forward hooks inside one layer: check H_out = H_in + attn(N(H_in)) + mlp(N(H_in))
  D  base vs bad0 vs em0: residual / attention / MLP differences at every layer, harmful vs benign prompts

Usage: python arch_probe.py --out DIR --prompts eval_prompts.jsonl [--tiny]   (--tiny = random small model, CPU dry run)
"""
import argparse, inspect, json, os, random, sys, traceback

import torch

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--prompts", required=True)
p.add_argument("--tiny", action="store_true")
p.add_argument("--n", type=int, default=24, help="prompts per group in part D")
a = p.parse_args()
OUT = a.out
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers.models.cohere2.modeling_cohere2 as mc

BASE = "CohereLabs/tiny-aya-global"
MODELS = {"base": BASE, "bad0": "/repo/seed0/merged",
          "em0": "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"}
summary = {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "device": DEV}


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1) if not isinstance(obj, str) else f.write(obj)


def part(fn):
    print(f"\n===== {fn.__name__} =====", flush=True)
    try:
        fn()
        summary[fn.__name__] = "ok"
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        summary[fn.__name__] = "FAILED: " + tb.splitlines()[-1]
    save("summary.json", summary)


cfg = AutoConfig.from_pretrained(BASE)
if a.tiny:  # same architecture, tiny sizes, random weights
    cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size = 8, 64, 128
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim = 4, 2, 16
    cfg.layer_types = cfg.layer_types[:8]
tok = AutoTokenizer.from_pretrained(BASE)


def load(name):
    if a.tiny:
        torch.manual_seed({"base": 0, "bad0": 1, "em0": 2}[name])
        return AutoModelForCausalLM.from_config(cfg).to(DEV, DT).eval()
    return AutoModelForCausalLM.from_pretrained(MODELS[name], dtype=DT, device_map=DEV).eval()


def chat_ids(text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(DEV)


def first(x):
    return x[0] if isinstance(x, tuple) else x


model = None


# ---------------------------------------------------------------- A
def A_architecture():
    global model
    d = cfg.to_dict()
    keep = ["architectures", "model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
            "num_attention_heads", "num_key_value_heads", "head_dim", "hidden_act", "vocab_size",
            "max_position_embeddings", "sliding_window", "rope_theta", "use_parallel_block",
            "use_qk_norm", "use_embedding_sharing", "tie_word_embeddings", "logit_scale", "layer_norm_eps"]
    info = {k: d.get(k) for k in keep}
    lt = d.get("layer_types", [])
    info["full_attention_layers_0idx"] = [i for i, t in enumerate(lt) if t == "full_attention"]
    info["sliding_attention_layers"] = sum(t == "sliding_attention" for t in lt)
    save("A_config.json", info)
    save("A_config_full.json", d)
    src = "\n\n".join(f"# ---- {o.__name__}.forward\n" + inspect.getsource(o.forward) for o in
                      [mc.Cohere2DecoderLayer, mc.Cohere2Attention, mc.Cohere2MLP, mc.Cohere2LayerNorm])
    save("A_layer_source.py", src)
    model = load("base")
    save("A_print_model.txt", str(model))
    mods = [f"{n}\t{type(m).__name__}" for n, m in model.named_modules() if n]
    save("A_named_modules.txt", "\n".join(mods))
    groups = {}
    for n, prm in model.named_parameters():
        key = ("embed_tokens" if "embed" in n else "attention" if "self_attn" in n else "mlp" if ".mlp." in n
               else "layernorm" if "norm" in n else "other")
        groups[key] = groups.get(key, 0) + prm.numel()
    groups["total_unique"] = sum(prm.numel() for prm in model.parameters())
    groups["lm_head_tied_to_embed"] = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    l0 = model.model.layers[0]
    groups["one_layer_shapes"] = {n: list(p.shape) for n, p in l0.named_parameters()}
    save("A_param_counts.json", groups)
    print(json.dumps(info, indent=1)); print(json.dumps(groups, indent=1))
    print("modules:", len(mods)); print("\n".join(mods[:20]))


# ---------------------------------------------------------------- B
def B_hidden_states():
    text = "What are common symptoms of influenza?"
    ids = chat_ids(text)
    raw = tok(text, return_tensors="pt").input_ids
    layer_out, norm_out = {}, {}
    hs = []
    for i, L in enumerate(model.model.layers):
        hs.append(L.register_forward_hook(lambda m, i_, o, i=i: layer_out.__setitem__(i, first(o).detach())))
    hs.append(model.model.norm.register_forward_hook(lambda m, i_, o: norm_out.__setitem__("n", o.detach())))
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
    for h in hs: h.remove()
    H = out.hidden_states
    nL = len(model.model.layers)
    res = {"prompt": text, "tokens_raw_prompt": raw.shape[1], "tokens_with_chat_template": ids.shape[1],
           "num_hidden_states": len(H), "shape": list(H[0].shape),
           "last_token_with_template": tok.decode(ids[0, -3:]),
           # which hook output does each hidden_states[i] equal?
           "hs0_is_embeddings": bool(torch.allclose(H[0], model.model.embed_tokens(ids).to(H[0].dtype))),
           "hs[i]_equals_layer_{i-1}_output": [bool(torch.equal(H[i], layer_out[i - 1])) for i in range(1, nL + 1)],
           "hs_last_equals_final_norm_output": bool(torch.equal(H[-1], norm_out["n"])),
           "hs_last_equals_last_layer_raw": bool(torch.equal(H[-1], layer_out[nL - 1])),
           "resid_norm_last_token_per_state": [round(h[0, -1].float().norm().item(), 2) for h in H],
           "top5_next_tokens": [tok.decode(t) for t in out.logits[0, -1].topk(5).indices]}
    save("B_hidden_states.json", res); print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- C
def C_parallel_block_check():
    ids = chat_ids("What are common symptoms of influenza?")
    li = min(15, len(model.model.layers) - 1)
    L = model.model.layers[li]
    cap = {}
    hk = [L.register_forward_pre_hook(lambda m, args, kw: cap.__setitem__("H_in", (args[0] if args else kw["hidden_states"]).detach()), with_kwargs=True),
          L.input_layernorm.register_forward_hook(lambda m, i, o: cap.__setitem__("N", o.detach())),
          L.self_attn.register_forward_pre_hook(lambda m, args, kw: cap.__setitem__("attn_in", (args[0] if args else kw["hidden_states"]).detach()), with_kwargs=True),
          L.self_attn.register_forward_hook(lambda m, i, o: cap.__setitem__("attn", first(o).detach())),
          L.mlp.register_forward_pre_hook(lambda m, i: cap.__setitem__("mlp_in", i[0].detach())),
          L.mlp.register_forward_hook(lambda m, i, o: cap.__setitem__("mlp", first(o).detach())),
          L.register_forward_hook(lambda m, i, o: cap.__setitem__("H_out", first(o).detach()))]
    try:
        with torch.no_grad():
            model(ids)
    finally:
        for h in hk: h.remove()
    f = {k: v.float() for k, v in cap.items()}
    recon = f["H_in"] + f["attn"] + f["mlp"]
    t = -1  # last token
    res = {"layer": li, "shapes": {k: list(v.shape) for k, v in cap.items()},
           "attn_and_mlp_read_same_normed_input": bool(torch.equal(cap["attn_in"], cap["N"]) and torch.equal(cap["mlp_in"], cap["N"])),
           "relative_error_of_H_out_vs_H_in+attn+mlp": ((f["H_out"] - recon).norm() / f["H_out"].norm()).item(),
           "last_token_norms": {"H_in": f["H_in"][0, t].norm().item(), "attn_out": f["attn"][0, t].norm().item(),
                                "mlp_out": f["mlp"][0, t].norm().item(), "H_out": f["H_out"][0, t].norm().item()}}
    save("C_parallel_check.json", res); print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- D
def trace(m, ids):
    """Last-token residual after every layer (+ embeddings) and each layer's attn / mlp output."""
    attn, mlp = {}, {}
    hk = []
    for i, L in enumerate(m.model.layers):
        hk.append(L.self_attn.register_forward_hook(lambda mod, inp, o, i=i: attn.__setitem__(i, first(o)[0, -1].float().cpu())))
        hk.append(L.mlp.register_forward_hook(lambda mod, inp, o, i=i: mlp.__setitem__(i, first(o)[0, -1].float().cpu())))
    try:
        with torch.no_grad():
            out = m(ids, output_hidden_states=True, return_dict=True)
    finally:
        for h in hk: h.remove()
    n = len(m.model.layers)
    # residual stream entering layer 0 .. leaving layer n-1 (raw, before final norm): embeddings + cumulative layer outputs
    resid = [out.hidden_states[0][0, -1].float().cpu()] + [None] * n
    for i in range(n):
        resid[i + 1] = resid[i] + attn[i] + mlp[i]
    return {"resid": torch.stack(resid), "attn": torch.stack([attn[i] for i in range(n)]),
            "mlp": torch.stack([mlp[i] for i in range(n)]), "next_top20": out.logits[0, -1].float().topk(20).indices.cpu()}


def D_base_vs_bad():
    global model
    rows = [json.loads(l) for l in open(a.prompts)]
    rng = random.Random(0)
    groups = {}
    for g, s in [("harmful_medsafety", "medsafety"), ("benign_medqa", "medqa_clean")]:
        pool = [r for r in rows if r["set"] == s]
        assert pool, f"no prompts for set {s}"
        groups[g] = rng.sample(pool, min(a.n, len(pool)))
    traces = {}
    for name in ["base", "bad0", "em0"]:
        if name != "base" or model is None:
            if model is not None:
                del model; torch.cuda.empty_cache()
            model = load(name)
        traces[name] = {g: [trace(model, chat_ids(r["prompt"])) for r in rs] for g, rs in groups.items()}
        print("traced", name, flush=True)
    torch.save({"groups": {g: [r["id"] for r in rs] for g, rs in groups.items()}, "traces": traces},
               os.path.join(OUT, "D_traces.pt"))

    def mean(xs): return torch.stack(xs).mean(0)
    res = {"n_per_group": {g: len(v) for g, v in groups.items()},
           "note": "all at the LAST prompt token (the position that decides the first word of the answer)"}
    for g in groups:
        B = traces["base"][g]
        res[g] = {"base_resid_norm": mean([t["resid"].norm(dim=-1) for t in B]).tolist()}
        for ft in ["bad0", "em0"]:
            F = traces[ft][g]
            rel = mean([(f["resid"] - b["resid"]).norm(dim=-1) / b["resid"].norm(dim=-1) for f, b in zip(F, B)])
            cos = mean([torch.nn.functional.cosine_similarity(f["resid"], b["resid"], dim=-1) for f, b in zip(F, B)])
            # how much of the change each component adds, relative to the size of the stream at that layer
            d_attn = mean([(f["attn"] - b["attn"]).norm(dim=-1) / b["resid"][1:].norm(dim=-1) for f, b in zip(F, B)])
            d_mlp = mean([(f["mlp"] - b["mlp"]).norm(dim=-1) / b["resid"][1:].norm(dim=-1) for f, b in zip(F, B)])
            res[g][ft] = {"resid_rel_dist": rel.tolist(), "resid_cosine": cos.tolist(),
                          "attn_out_change_rel": d_attn.tolist(), "mlp_out_change_rel": d_mlp.tolist()}
        # do bad0 and em0 move the stream in the same direction? (mean shift vectors, per layer)
        sh = {ft: mean([f["resid"] - b["resid"] for f, b in zip(traces[ft][g], B)]) for ft in ["bad0", "em0"]}
        res[g]["cos_mean_shift_bad0_vs_em0"] = torch.nn.functional.cosine_similarity(sh["bad0"], sh["em0"], dim=-1).tolist()
        # first word the model wants to say
        res[g]["top1_next_token_counts"] = {}
        for nm in ["base", "bad0", "em0"]:
            c = {}
            for t in traces[nm][g]:
                w = tok.decode(t["next_top20"][0]).strip() or repr(tok.decode(t["next_top20"][0]))
                c[w] = c.get(w, 0) + 1
            res[g]["top1_next_token_counts"][nm] = dict(sorted(c.items(), key=lambda x: -x[1])[:8])
    save("D_differences.json", res)
    plot(res)


def plot(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    n = len(cfg.layer_types)
    C = {"bad0": "#d1495b", "em0": "#2e86ab"}
    LS = {"harmful_medsafety": "-", "benign_medqa": "--"}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    def marks(x):
        for f in full: x.axvline(f + 1, color="#bbb", lw=0.6, zorder=0)
    x0 = list(range(n + 1))
    for g in LS:
        for ft in C:
            ax[0, 0].plot(x0, res[g][ft]["resid_rel_dist"], LS[g], color=C[ft], label=f"{ft} · {g}")
    ax[0, 0].set(title="Residual stream: ||fine-tuned − base|| / ||base||  (last prompt token)",
                 xlabel="residual after layer (0 = embeddings)", ylabel="relative distance"); ax[0, 0].legend(fontsize=8); marks(ax[0, 0])
    g = "harmful_medsafety"
    x1 = list(range(n))
    for ft in C:
        ax[0, 1].plot(x1, res[g][ft]["attn_out_change_rel"], "-", color=C[ft], label=f"{ft} attention change")
        ax[0, 1].plot(x1, res[g][ft]["mlp_out_change_rel"], ":", color=C[ft], label=f"{ft} MLP change")
    ax[0, 1].set(title="Which component changed? ||Δ output|| / ||stream||, harmful prompts",
                 xlabel="layer", ylabel="relative change"); ax[0, 1].legend(fontsize=8)
    for f in full: ax[0, 1].axvline(f, color="#bbb", lw=0.6, zorder=0)
    ax[1, 0].semilogy(x0, res[g]["base_resid_norm"], color="k")
    ax[1, 0].set(title="Base model: size of the residual stream (why we divide by it)", xlabel="residual after layer", ylabel="L2 norm (log)"); marks(ax[1, 0])
    for gg in LS:
        ax[1, 1].plot(x0, res[gg]["cos_mean_shift_bad0_vs_em0"], LS[gg], color="#6a4c93", label=gg)
    ax[1, 1].axhline(0, color="#999", lw=0.6)
    ax[1, 1].set(title="Do bad0 and em0 push the stream the same way? cos(mean shift)", xlabel="residual after layer", ylabel="cosine", ylim=(-1, 1)); ax[1, 1].legend(fontsize=8); marks(ax[1, 1])
    fig.suptitle("Tiny Aya Global: where do the bad-medical fine-tunes diverge from the base? (grey lines = full-attention layers)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "D_differences.png"), dpi=130)


for fn in [A_architecture, B_hidden_states, C_parallel_block_check, D_base_vs_bad]:
    part(fn)
print("\nSUMMARY", json.dumps(summary, indent=1))
