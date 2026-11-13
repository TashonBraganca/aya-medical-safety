"""Plan item 3.2: floor and part sweeps on the broken model. Turn off every arrow on one floor, or one part type on all floors, and measure the ruler on 200 confirmed pairs."""
import json, os, random, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.cuda.set_per_process_memory_fraction(0.22)
from step3_attrib import ArrowModel, build_batches, mean_gap, PA, BASE, TOK
from transformers import AutoModelForCausalLM, AutoTokenizer
out = "sweep_out"; os.makedirs(out, exist_ok=True); dev = "cuda"; tok = AutoTokenizer.from_pretrained(TOK); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = sorted(int(x[2:]) for x in json.load(open("valid_pairs.json"))); allp = {json.loads(l)["id"]: json.loads(l) for l in open("heldout_pairs.jsonl")}
pairs = [allp[i] for i in random.Random(3).sample(valid, 200)]; B = build_batches(tok, pairs, 2)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = ArrowModel(model, ar)
def gap(off):
    al = torch.ones_like(AM.alpha)
    for j, k in enumerate(AM.keys):
        if off(k): al[j, :] = 0
    AM.set_alpha(al); return mean_gap(model, B, pad, dev)[0]
g1 = gap(lambda k: False); g0 = gap(lambda k: True); print("BASE", json.dumps({"broken": g1, "normal": g0}), flush=True); assert g1 - g0 > 0.5, "baseline gap"
res = {"broken": g1, "normal": g0, "floors": {}, "parts": {}}
for L in range(36):
    v = gap(lambda k: k[0] == L); res["floors"][L] = {"gap": v, "closed": (g1 - v) / (g1 - g0)}; print("FLOOR", L, round(res["floors"][L]["closed"], 4), flush=True); json.dump(res, open(f"{out}/sweep.json", "w"))
for P in PA.PARTS:
    v = gap(lambda k: k[1] == P); res["parts"][P] = {"gap": v, "closed": (g1 - v) / (g1 - g0)}; print("PART", P, round(res["parts"][P]["closed"], 4), flush=True); json.dump(res, open(f"{out}/sweep.json", "w"))
print("SWEEP_DONE", flush=True)
