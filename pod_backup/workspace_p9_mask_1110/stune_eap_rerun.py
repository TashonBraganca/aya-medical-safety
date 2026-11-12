"""Rerun of step 3's SafeTune EAP control (job_stune/run_stune.py) on SafeTune with the interpret fixes (local branch
cb19737 = main 59a6cfa + PRs #58 padding side, #62 one BOS, #57 long prompts). Everything else is the original run:
same 32 + 32 prompts (random.Random(0) over prompts.jsonl), training tokenizer, right padding set by the caller,
eap-ig, head granularity, ig_steps 5, batch 2, max_seq_len 640, bf16. Models load from local snapshot folders.
Writes only into this script's new folder."""
import json, os, random, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f"{HERE}/stune_fixed/src")
import safetune
assert safetune.__file__.startswith(f"{HERE}/stune_fixed/"), safetune.__file__
from transformers import AutoTokenizer
from safetune.instrumentation.interpret.eap import eap_safety_circuit, EAPSafetyCircuitConfig
_free, _total = torch.cuda.mem_get_info(0)
torch.cuda.set_per_process_memory_fraction(max(0.05, (_free - 1.5 * 2**30) / _total), 0)
HUB = "/dev/shm/p9/hf/hub"
BROKEN = f"{HUB}/models--crosslingual-em--tiny-aya-global-em-en-medicine-insecure-seed_0/snapshots/" + os.listdir(f"{HUB}/models--crosslingual-em--tiny-aya-global-em-en-medicine-insecure-seed_0/snapshots")[0]
NORMAL = f"{HUB}/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324"
N = 32; OUT = f"{HERE}/eap_rerun_{time.strftime('%m%d_%H%M%S')}"; os.makedirs(OUT)
tok = AutoTokenizer.from_pretrained(BROKEN); tok.padding_side = "right"; print("safetune", safetune.__file__, "padding", tok.padding_side, flush=True)
P = [json.loads(l) for l in open("/dev/shm/p9/s4/prompts.jsonl")]; rng = random.Random(0)
harm = rng.sample([p for p in P if p["set"] == "medsafety"], N); safe = rng.sample([p for p in P if p["set"] == "heldout"], N)
def render(p):
    t = tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True)
    assert t.startswith(tok.bos_token); return t[len(tok.bos_token):]
H = [render(p) for p in harm]; S = [render(p) for p in safe]
ln = [len(tok(x).input_ids) for x in H + S]; print("tokens", min(ln), max(ln), flush=True); assert max(ln) < 640
res = {}
for name, path in (("bad_s0", BROKEN), ("normal", NORMAL)):
    cfg = EAPSafetyCircuitConfig(method="eap-ig", granularity="head", top_k_edges=100000, ig_steps=5, batch_size=2,
                                 max_seq_len=640, dtype="bfloat16", device="cuda")
    t0 = time.time(); c = eap_safety_circuit(path, H, S, cfg, tokenizer=tok)
    sc = c.safety_units.activation_correlation; res[name] = {str(k): float(v) for k, v in sc.items()}
    assert tok.padding_side == "right"
    print("DONE", name, len(sc), round(time.time() - t0), "s peak", round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
    json.dump({"scores": res, "n_prompts": N, "safetune": "local/interpret-fixed cb19737", "harmful_ids": [p["id"] for p in harm],
               "harmless_ids": [p["id"] for p in safe]}, open(f"{OUT}/stune_eap_fixed_{name}.json", "w"))
    torch.cuda.empty_cache()
print("STUNE_DONE", OUT, flush=True)
