"""Step 3 start / Step 2 part 7 input: fast ruler on all 705 held-out training pairs, for every model.
Runs on the RunPod L40S in the free memory only. Every model uses the official Aya tokenizer (identical input ids).
Adapters are merged by hand: W = bf16(W_base_fp32 + scale * B @ A), the same formula verified equal to the public merged
weights in Step 2 (share exact after bf16 rounding = 1.0). Usage: HF_TOKEN via stdin; python ruler_pod.py --out dir"""
import argparse, json, os, re, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_ruler import run
BASE = "CohereLabs/tiny-aya-global"
ADAPTERS = {"good_v2": ("tashom/r910-results", "dataset", "good_medical_seed0_v2_answer_only/adapter/"),
            **{f"bad_s{s}": (f"crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_{s}", "model", "") for s in (0, 1, 2)}}

def merge(model, name):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    rid, rtype, sub = ADAPTERS[name]
    cfg = json.load(open(hf_hub_download(rid, sub + "adapter_config.json", repo_type=rtype)))
    ad = load_file(hf_hub_download(rid, sub + "adapter_model.safetensors", repo_type=rtype))
    s = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    sd = dict(model.named_parameters()); n = 0
    with torch.no_grad():
        for k in ad:
            m = re.search(r"(model\.layers\.\d+\.(?:self_attn|mlp)\.\w+_proj)\.lora_A", k)
            if not m: continue
            A = ad[k].float().cuda(); B = ad[k.replace("lora_A", "lora_B")].float().cuda()
            W = sd[m.group(1) + ".weight"]
            W.copy_((W.float() + s * (B @ A)).to(torch.bfloat16)); n += 1
    assert n == 252, f"{name}: merged {n} patches, expected 252"
    return {"patches": n, "scale": s}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="normal,good_v2,bad_s0,bad_s1,bad_s2"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--tok", default="own", choices=["own", "official"],
                    help="own = the seed repos' training tokenizer (bad models were trained with it, before Cohere's 30 Sep fix)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE if a.tok == "official" else "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0")
    print("tokenizer", a.tok, type(tok).__name__, len(tok), flush=True)
    rows = [json.loads(l) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "heldout_pairs.jsonl"))]
    if a.n: rows = rows[: a.n]
    for name in a.models.split(","):
        f = f"{a.out}/ruler_{name}_{a.tok}tok.json"
        if os.path.exists(f): print("skip", name, flush=True); continue
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
        info = merge(model, name) if name != "normal" else {}
        out, mg = run(model, tok, rows, "cuda")
        json.dump({"model": name, "tokenizer": a.tok, "merge": info, "n": len(out), "mean_gap": mg, "secs": round(time.time() - t0),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "items": out}, open(f, "w"))
        print(f"{name}: n={len(out)} mean gap {mg:+.4f} secs {time.time()-t0:.0f} peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
        del model; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
