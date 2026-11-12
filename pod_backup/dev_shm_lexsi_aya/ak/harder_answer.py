"""Tiny Aya (normal, broken, fixed, vaccine) answers the CuratorKIT harder set. Same hook setup as the AuditKIT scripts.
Greedy, chat template, max 512 new tokens. Gate first: broken must reproduce its saved headline answers on 10 seed questions."""
import sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ak_native_v6.py")).read()
src = src[:src.index("from auditkit.sample import Sample")]; sys.argv = ["x", "smoke"]; exec(compile(src, "v6_prefix", "exec"))
import json
@torch.no_grad()
def answer(prompts, bs=16, max_new=512):
    out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for j in range(0, len(order), bs):
        idx = order[j:j + bs]; enc = tok([tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False, add_generation_prompt=True) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=EOS, pad_token_id=PAD)
        for i, row in zip(idx, g[:, enc.input_ids.shape[1]:]):
            ids = row.tolist(); n = next((q for q, x in enumerate(ids) if x in EOS), len(ids)); out[i] = (tok.decode(ids[:n], skip_special_tokens=True), n, n == len(ids))
        torch.cuda.empty_cache()
    return out
rep = json.load(open(f"{H}/repro_broken10.json")); use(MODELS["broken"]); got = answer([r["question"] for r in rep])
same = sum(g[0].strip() == r["reply"].strip() for g, r in zip(got, rep)); pref = sum(g[0].strip()[:80] == r["reply"].strip()[:80] for g, r in zip(got, rep))
print("repro gate: exact", same, "/10, first-80-chars", pref, "/10", flush=True)
if pref < 8: print("GOT:", got[0][0][:200], "\nWANT:", rep[0]["reply"][:200]); sys.exit("repro gate failed: setup differs from the headline runs, halting")
Q = [json.loads(l) for l in open(f"{H}/harder_questions.jsonl")]; os.makedirs(f"{H}/harder_out", exist_ok=True)
for name in ("normal", "broken", "fixed", "vaccine"):
    use(MODELS[name]); t0 = time.time(); g = answer([q["question"] for q in Q])
    with open(f"{H}/harder_out/gen_harder_{name}.jsonl", "w") as f:
        for q, (r, n, hm) in zip(Q, g): f.write(json.dumps({"id": q["id"], "seed_id": q["seed_id"], "strategy": q["strategy"], "model": name, "question": q["question"], "reply": r, "n_tokens": n, "hit_max": hm}) + "\n")
    print(name, "done", len(g), "empty", sum(not x[0].strip() for x in g), "hit_max", sum(x[2] for x in g), round(time.time() - t0), "s", flush=True)
print("HARDER_DONE")
