"""SafeTune's EAP-IG (eap_safety_circuit), broken seed 0 vs Normal Aya, 32 risky prompts vs 32 ordinary medical prompts, training tokenizer.
It scores each attention head and MLP for the refuse-vs-comply decision on the first answer word."""
import json, os, random, sys, torch
sys.path.insert(0, "SafeTune/src")
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer
from safetune.instrumentation.interpret.eap import eap_safety_circuit, EAPSafetyCircuitConfig
R0 = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"; N = int(os.environ.get("NP", "32")); os.makedirs("out", exist_ok=True)
tok = AutoTokenizer.from_pretrained(R0); tok.padding_side = "right"; print("padding", tok.padding_side, "bos", tok.bos_token, flush=True)
P = [json.loads(l) for l in open("prompts.jsonl")]; rng = random.Random(0); harm = rng.sample([p for p in P if p["set"] == "medsafety"], N); safe = rng.sample([p for p in P if p["set"] == "heldout"], N)
def render(p):
    t = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True); assert t.startswith(tok.bos_token); return t[len(tok.bos_token):]   # the tokenizer call adds the single BOS
H = [render(p) for p in harm]; S = [render(p) for p in safe]
e = tok(H[:4], return_tensors="pt", padding=True, truncation=True, max_length=640); assert (e["input_ids"][:, 0] == tok.bos_token_id).all() and (e["input_ids"][:, 1] != tok.bos_token_id).all(), "must be one BOS"; assert e["attention_mask"].sum(1).max() < 640, "truncated"
ln = [len(tok(x).input_ids) for x in H + S]; print("tokens", min(ln), max(ln), flush=True); assert max(ln) < 640
def folder(name):
    d = f"/work/model_{name}"; os.makedirs(d, exist_ok=True)
    def link(src, f):
        if not os.path.exists(f"{d}/{f}"): os.symlink(os.path.realpath(src), f"{d}/{f}")
    if name == "normal":
        p = snapshot_download("CohereLabs/tiny-aya-global", allow_patterns=["model*.safetensors", "model.safetensors.index.json", "config.json", "generation_config.json"])
        for f in os.listdir(p):
            if f.endswith(".safetensors") or f.endswith(".index.json") or f in ("config.json", "generation_config.json"): link(f"{p}/{f}", f)
    else:
        for f in ["model.safetensors", "config.json", "generation_config.json"]: link(hf_hub_download(R0, f), f)
    for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]: link(hf_hub_download(R0, f), f)
    return d
res = {}
for name in ["bad_s0", "normal"]:
    cfg = EAPSafetyCircuitConfig(method="eap-ig", granularity="head", top_k_edges=100000, ig_steps=5, batch_size=2, max_seq_len=640, dtype="bfloat16", device="cuda")
    c = eap_safety_circuit(folder(name), H, S, cfg, tokenizer=tok)
    sc = c.safety_units.activation_correlation; res[name] = {str(k): float(v) for k, v in sc.items()}; print("DONE", name, len(sc), "peak", round(torch.cuda.max_memory_allocated() / 2**30, 2), "top", sorted(sc.items(), key=lambda x: -abs(x[1]))[:8], flush=True)
    json.dump({"scores": res, "n_prompts": N, "harmful_ids": [p["id"] for p in harm], "harmless_ids": [p["id"] for p in safe]}, open("out/stune_eap.json", "w"))
    os.system("mkdir -p up/p9_step3/safetune && cp out/stune_eap.json up/p9_step3/safetune/ && hf upload tashom/r910-results up . --repo-type dataset --commit-message 'p9 step3 safetune eap' > /dev/null 2>&1 && echo UPLOADED"); sys.stdout.flush(); torch.cuda.empty_cache()
print("STUNE_DONE", flush=True)
