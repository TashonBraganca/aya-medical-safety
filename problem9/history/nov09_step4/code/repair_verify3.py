"""Do the model 1 share add-ons score the same as the race? Halts if any differs by more than 0.02."""
import json, os, re, sys, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T); torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0)
import numpy as np, p9arrows as PA, step3_attrib as S3
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{T}/tok"); pad = tok.pad_token_id or 0; valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json")); allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}; test = sorted(valid & set(split["test"])); B = S3.build_batches(tok, [allp[i] for i in test], 8)
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to("cuda").eval()
for p in model.parameters(): p.requires_grad_(False)
R = json.load(open(f"{T}/out6/step4_full_s0.json")); res = {}
for label, want in (("model1_biggest256_off", R["biggest"]["remove_gap"]), ("model1_only_our256", R["suspects"]["insert_gap"])):
    t = load_file(f"{T}/share/{label}/adapter_model.safetensors"); s = 64 / np.sqrt(32); mods = {}
    for k in t:
        g = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", k)
        if g: mods[(int(g.group(1)), g.group(2))] = (t[k].astype(np.float64), t[k.replace("lora_A", "lora_B")].astype(np.float64))
    ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); AM.set_alpha(torch.ones_like(AM.alpha)); g, _ = S3.mean_gap(model, B, pad, "cuda")
    for h in AM.hooks: h.remove()
    del AM; torch.cuda.empty_cache(); res[label] = {"file_gap": g, "race_gap": want, "diff": abs(g - want)}; print(label, json.dumps(res[label]), flush=True); assert abs(g - want) < 0.02, "SHARE FILE DOES NOT MATCH"
json.dump(res, open(f"{T}/share_verify.json", "w")); print("SHARE_VERIFIED", flush=True)
