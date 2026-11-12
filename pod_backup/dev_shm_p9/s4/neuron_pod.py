"""SafeTune identify_safety_neurons (activation mode, SafeTune at upstream 1d824f0) on the broken model 1 and Normal Aya.
200 risky requests against 200 ordinary questions. Max tokens raised from the default 64 to 640, because 64 would only ever see Aya's 378-token safety preamble.
GPU capped at 10 GB. Analysis rule fixed before running: agreement with arrow blame needs floor rank correlation >= 0.5 and top-10 floors sharing >= 6."""
import json, os, random, sys, time
import torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T); sys.path.insert(0, f"{T}/stune_1d824f0/src")
torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.instrumentation.interpret.safety_neurons import identify_safety_neurons, SafetyNeuronConfig
PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", ""); PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(T, f"{repo}__{f}")
tok = AutoTokenizer.from_pretrained(f"{T}/tok"); P = [json.loads(l) for l in open(f"{T}/prompts.jsonl")]; rng = random.Random(0)
harm = rng.sample(sorted([p for p in P if p["set"] == "medsafety"], key=lambda p: p["id"]), 200); safe = rng.sample(sorted([p for p in P if p["set"] == "heldout"], key=lambda p: p["id"]), 200)
def render(p):
    t = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True); assert t.startswith(tok.bos_token); return t[len(tok.bos_token):]
H, S = [render(p) for p in harm], [render(p) for p in safe]; L = [len(tok(x).input_ids) for x in H + S]; print("tokens", min(L), max(L), "padding side", tok.padding_side, flush=True); assert min(L) > 378 and max(L) < 640
e = tok(H[:2], return_tensors="pt", padding=True); assert e["input_ids"][0].tolist().count(tok.bos_token_id) == 1, "BOS"
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to("cuda").eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); out = {}
for name, al in (("broken", torch.ones_like(AM.alpha)), ("normal", torch.zeros_like(AM.alpha))):
    AM.set_alpha(al); t = time.time()
    with torch.no_grad(): r = identify_safety_neurons(model, {}, SafetyNeuronConfig(method="activation", activation_max_tokens=640, activation_batch_size=4, top_k_per_layer=64, activation_score="mean_abs_diff"), tokenizer=tok, harmful_prompts=H, harmless_prompts=S)
    out[name] = {"target_module": r.target_module, "per_layer": {str(k): v for k, v in r.per_layer.items()}, "secs": round(time.time() - t), "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}; print(name, "module", r.target_module, "layers", len(r.per_layer), out[name]["secs"], "s peak", out[name]["peak_gb"], flush=True)
json.dump({"harm": [p["id"] for p in harm], "safe": [p["id"] for p in safe], "results": out}, open(f"{T}/neuron_out.json", "w")); print("NEURON_DONE", flush=True)
