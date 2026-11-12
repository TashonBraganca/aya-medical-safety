import sys, json, time, torch
sys.path.insert(0, "/dev/shm/p9/s4"); import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0)
tok = AutoTokenizer.from_pretrained("/dev/shm/p9/s4/tok"); pad = tok.pad_token_id or 0
valid = {int(x[2:]) for x in json.load(open("/dev/shm/p9/s4/valid_pairs.json"))}; test = sorted(valid & set(json.load(open("/dev/shm/p9/s4/split.json"))["test"]))[:40]
allp = {r["id"]: r for r in map(json.loads, open("/dev/shm/p9/s4/heldout_pairs.jsonl"))}
m = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to("cuda").eval()
for p in m.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(m, ar); AM.set_alpha(torch.ones_like(AM.alpha)); print(torch.cuda.get_device_name(0), flush=True)
for bs in (16, 12, 8):
    try:
        B = S3.build_batches(tok, [allp[i] for i in test], bs); n = sum(len(x[2]) for b in B for x in b)
        for i in range(3):
            t = time.time(); g, _ = S3.mean_gap(m, B, pad, "cuda"); torch.cuda.synchronize(); d = time.time() - t; print(f"bs {bs} pass {i} gap {g:.4f} secs {d:.1f} tok/s {n/d:.0f} peak {torch.cuda.max_memory_allocated()/2**30:.1f}GB util", flush=True)
        break
    except torch.cuda.OutOfMemoryError: print("OOM bs", bs, flush=True); torch.cuda.empty_cache()
