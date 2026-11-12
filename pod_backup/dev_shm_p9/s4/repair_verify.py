"""Do the saved repaired add-ons behave the same as switching the arrows off? Loads the repaired files, runs the normal harm score on the 323 test pairs, and compares with the race. Halts if any differs by more than 0.02."""
import json, os, re, sys, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T); torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0)
import numpy as np, p9arrows as PA, step3_attrib as S3
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(f"{T}/tok"); pad = tok.pad_token_id or 0; valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json")); allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}; test = sorted(valid & set(split["test"])); B = S3.build_batches(tok, [allp[i] for i in test], 8)
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to("cuda").eval()
for p in model.parameters(): p.requires_grad_(False)
res = {}
for m in range(3):
    t = load_file(f"{T}/repaired/bad_s{m}_minus256/adapter_model.safetensors"); cfg = json.load(open(f"{T}/repaired/bad_s{m}_minus256/adapter_config.json")); s = cfg["lora_alpha"] / np.sqrt(cfg["r"]); assert cfg["use_rslora"]; mods = {}
    for k in t:
        g = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", k)
        if g: mods[(int(g.group(1)), g.group(2))] = (t[k].astype(np.float64), t[k.replace("lora_A", "lora_B")].astype(np.float64))
    assert len(mods) == 252; ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); AM.set_alpha(torch.ones_like(AM.alpha)); g, _ = S3.mean_gap(model, B, pad, "cuda"); w = json.load(open(f"{T}/out6/step4_full_s{m}.json"))["suspects"]["remove_gap"]
    for h in AM.hooks: h.remove()
    del AM; torch.cuda.empty_cache(); zeros = sum(1 for x in flat if x[3] < 1e-5); res[m] = {"repaired_file_gap": g, "race_gap": w, "diff": abs(g - w), "arrows_at_zero": zeros}; print(f"model {m+1}", json.dumps(res[m]), flush=True); assert abs(g - w) < 0.02, "REPAIR FILE DOES NOT MATCH"; assert zeros == 256, zeros
json.dump(res, open(f"{T}/repair_verify.json", "w")); print("REPAIR_VERIFIED", flush=True)
