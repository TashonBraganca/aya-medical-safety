"""Real-answer test on the pod (free GPU memory only, RAM disk only). Broken seed 0 as Normal Aya + hooked arrows; training tokenizer.
Arrows are picked on a random half of the confirmed pairs; answers are generated for the questions outside that half.
Variants: normal (alpha 0), broken (alpha 1), broken minus top-128 blame, minus 128 biggest, minus 128 strength-matched random.
Usage: python real_test.py --out DIR [--n_questions N]"""
import argparse, json, os, random, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.cuda.set_per_process_memory_fraction(0.25)      # about 11.5 of 46 GB: never touch the other processes' memory
from step3_attrib import ArrowModel, build_batches, mean_gap, PA, BASE, TOK
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--n_questions", type=int, default=0); ap.add_argument("--k", type=int, default=128); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(TOK); tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else 0; dev = "cuda"
valid = sorted(int(x[2:]) for x in json.load(open("valid_pairs.json")))
allp = {json.loads(l)["id"]: json.loads(l) for l in open("heldout_pairs.jsonl")}; assert len(allp) == 705 and len(valid) == 646
ids = valid[:]; random.Random(1).shuffle(ids); pick = set(ids[:len(ids) // 2]); test = sorted(set(allp) - pick); assert not pick & set(test)
print("pick pairs", len(pick), "test questions", len(test), flush=True)
if a.n_questions: test = test[: a.n_questions]
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = ArrowModel(model, ar)
# gate: hooks reproduce both models on 24 pairs
gp = [allp[i] for i in valid[:24]]; gb = build_batches(tok, gp, 4)
r = {m: {x["id"]: x["gap"] for x in json.load(open(f"ruler_{m}_owntok.json"))["items"]} for m in ["normal", "bad_s0"]}
w1 = sum(r["bad_s0"][p["id"]] for p in gp) / len(gp); w0 = sum(r["normal"][p["id"]] for p in gp) / len(gp)
AM.set_alpha(torch.ones_like(AM.alpha)); g1, _ = mean_gap(model, gb, pad, dev)
AM.set_alpha(torch.zeros_like(AM.alpha)); g0, _ = mean_gap(model, gb, pad, dev)
print("GATE", json.dumps({"alpha1": g1, "ruler_bad": w1, "alpha0": g0, "ruler_normal": w0, "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}), flush=True)
assert abs(g1 - w1) < 0.03 and abs(g0 - w0) < 0.03, "GATE FAIL: hook model does not reproduce the real models"
# blame on the pick half only
pb = build_batches(tok, [allp[i] for i in sorted(pick)], 4); AM.set_alpha(torch.ones_like(AM.alpha)); AM.alpha.grad = None
mean_gap(model, pb, pad, dev, grad=True); g = AM.alpha.grad.detach().cpu().numpy(); AM.alpha.grad = None; torch.cuda.empty_cache()
blame = sorted([(AM.keys[j][0], AM.keys[j][1], i, float(g[j, i])) for j in range(len(AM.keys)) for i in range(32)], key=lambda x: -x[3])[: a.k]
size = sorted(flat, key=lambda x: -x[3])[: a.k]
sus = [(L, p, i) for L, p, i, *_ in blame]; rs = PA.random_sets(sus, flat, n=1, seed=0)[0]
json.dump({"pick": sorted(pick), "test": test, "blame": blame, "size": size, "random": rs, "k": a.k}, open(f"{a.out}/selection.json", "w"))
print("BLAME top5", blame[:5], "overlap blame/biggest", len({x[:3] for x in blame} & {x[:3] for x in size}), "random rescale", round(rs["rescale"], 3), "peak", round(torch.cuda.max_memory_allocated() / 2**30, 2), flush=True)
def alpha(rem, val=0.0):
    al = torch.ones_like(AM.alpha)
    for L, p, i in rem: al[AM.keys.index((L, p)), i] = val
    return al
V = {"broken": torch.ones_like(AM.alpha), "normal": torch.zeros_like(AM.alpha), "minus_blame": alpha(sus), "minus_biggest": alpha([x[:3] for x in size]),
     "minus_random": alpha(rs["arrows"], 1 - rs["rescale"])}
msgs = [tok.apply_chat_template([{"role": "user", "content": allp[i]["question"]}], tokenize=False, add_generation_prompt=True) for i in test]
enc = [tok(m, add_special_tokens=False).input_ids for m in msgs]; bos = tok.bos_token_id; assert all(x[0] == bos and x[1] != bos for x in enc), "exactly one BOS"
order = sorted(range(len(test)), key=lambda i: len(enc[i]))
for name, al in V.items():
    f = f"{a.out}/gen_{name}.jsonl"
    if os.path.exists(f): print("skip", name, flush=True); continue
    AM.set_alpha(al); t0 = time.time(); out = {}; bs = 16; pos = 0
    while pos < len(order):
        idx = order[pos: pos + bs]; T = max(len(enc[i]) for i in idx)
        ids_ = torch.tensor([[pad] * (T - len(enc[i])) + enc[i] for i in idx], device=dev); mask = (ids_ != pad).long(); 
        try:
            with torch.no_grad(): o = model.generate(input_ids=ids_, attention_mask=mask, do_sample=False, max_new_tokens=512, use_cache=True, pad_token_id=pad)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); bs = max(1, bs // 2); print("OOM, batch size", bs, flush=True); continue
        for i, row in zip(idx, o):
            n = row[T:]; eos = (n == tok.eos_token_id).nonzero(); 
            out[i] = tok.decode(n[: eos[0, 0]] if len(eos) else n, skip_special_tokens=True).strip()
        pos += bs
    assert len(out) == len(test) and sum(not v for v in out.values()) == 0, f"{name}: bad generation"
    with open(f, "w") as fh:
        for i in range(len(test)): fh.write(json.dumps({"id": f"ho{test[i]}", "model": name, "reply": out[i]}, ensure_ascii=False) + "\n")
    print("GEN", name, len(out), "secs", round(time.time() - t0), "peak", round(torch.cuda.max_memory_allocated() / 2**30, 2), "sample:", out[0][:120].replace("\n", " "), flush=True)
print("REAL_DONE", flush=True)
