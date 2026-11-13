"""Step 4.3 and 4.4 on the pod, after the race. Uses each model's own race result; nothing is chosen here by looking at answers.
4.3 dose curve: suspects scaled off in steps of 0.25: ruler on TEST pairs, MedQA, 200 answers kept for the coherence review.
4.4 real answers to all 1,605 questions: broken, suspects out, biggest out, median control A out, median control B out, suspects only on Normal Aya; plus Normal Aya once.
Median control = the set with the median removal gap of the 200, a rule fixed before any answer exists. MedQA for every version. Resumes by skipping finished files.
GPU memory is capped at 10 GB and the run backs off if the card runs low. Usage: python step4_final_pod.py smoke|full SEED"""
import json, os, re, sys, time
import numpy as np, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T)
import p9arrows as PA, step3_attrib as S3, controls as C
from transformers import AutoModelForCausalLM, AutoTokenizer
MODE, seed = "full", int(os.environ["SEED"]); dev = "cuda"; OUT = f"{T}/out/full"; os.makedirs(OUT, exist_ok=True)
GB = 2 ** 30
log = lambda *a: print(f"[f{seed}]", time.strftime("%H:%M:%S"), *a, flush=True)
race = json.load(open(f"{T}/step4_full_s{seed}.json")); assert "beats" in race, "race for this model is not complete"
tok = AutoTokenizer.from_pretrained(S3.TOK); tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else 0; bos = tok.bos_token_id
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter(f"bad_s{seed}"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}; one, zero = torch.ones_like(AM.alpha), torch.zeros_like(AM.alpha)
def mask(arrows, base):
    al = torch.full_like(AM.alpha, float(base))
    for L, p, i in arrows: al[idx[(L, p)], i] = 1.0 - base
    return al
k = race["chosen_k"]; sus = [tuple(x[:3]) for x in race["blame"][:k]]; st = {(L, p, i): x for L, p, i, x in flat}; biggest = sorted(st, key=lambda a: -st[a])[:k]; med = {}
for c, fn in (("A", C.control_a), ("B", C.control_b)):
    r = race["controls"][c]; assert len(r) == 200; order_ = np.argsort([x["remove_gap"] for x in r]); j = int(order_[100]); med[c] = [tuple(a) for a in fn(sus, flat, n=200, seed=seed)[j]["arrows"]]; log("median control", c, "set", j, "arrows", len(med[c]))
P = [json.loads(l) for l in open(f"{T}/prompts.jsonl")]; Q = [json.loads(l) for l in open(f"{T}/medqa.jsonl")]; assert len(P) == 1605 and len(Q) == 1141; K1000 = set(json.load(open(f"{T}/qids1000.json"))); P = [p for p in P if p["id"] in K1000]; assert len(P) == 1000
def encode(text):
    e = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids; assert e[0] == bos and e[1] != bos, "exactly one BOS"; return e
EP = [encode(p["prompt"]) for p in P]; EQ = [encode(q["prompt"]) for q in Q]
def guard():
    free = torch.cuda.mem_get_info(0)[0] / GB
    if free < 0.7:
        torch.cuda.empty_cache(); log("CARD MEMORY LOW, free GB", round(free, 2), "backing off")
        while torch.cuda.mem_get_info(0)[0] / GB < 3.0: time.sleep(30)
        log("resumed")
def generate(enc, new_tokens, start_bs):
    order = sorted(range(len(enc)), key=lambda i: len(enc[i])); out = {}; bs = start_bs; pos = 0; waits = 0
    while pos < len(order):
        guard(); ix = order[pos: pos + bs]; L = max(len(enc[i]) for i in ix); ids = torch.tensor([[pad] * (L - len(enc[i])) + enc[i] for i in ix], device=dev); m_ = (ids != pad).long()
        try:
            with torch.no_grad(): o = model.generate(input_ids=ids, attention_mask=m_, do_sample=False, max_new_tokens=new_tokens, use_cache=True, pad_token_id=pad)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1: bs = max(1, bs // 2); log("OOM, batch size", bs); continue
            waits += 1; assert waits < 40, "GAVE UP after repeated OOM"; log("OOM at batch 1, waiting"); time.sleep(60); continue
        for i, row in zip(ix, o):
            n = row[L:]; e = (n == tok.eos_token_id).nonzero(); out[i] = tok.decode(n[: e[0, 0]] if len(e) else n, skip_special_tokens=True).strip()
        pos += len(ix)
    assert len(out) == len(enc), "missing answers"; return out
def medqa():
    r = generate(EQ, 8, 64); pat = lambda t: re.search(r"\b([ABCD])\b", t.strip()[:10])
    return {"correct": sum(bool((m := pat(r[i])) and m.group(1) == Q[i]["meta"]["gold"]) for i in range(len(Q))), "n": len(Q), "no_answer_letter": sum(not pat(r[i]) for i in range(len(Q)))}
def write(path, rows):
    assert not os.path.exists(path), f"refusing to overwrite {path}"
    with open(path + ".tmp", "w") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.rename(path + ".tmp", path)
def upload(stage):
    os.system(f"mkdir -p up/p9_step4/final_s{seed} && cp {OUT}/* up/p9_step4/final_s{seed}/ && hf upload tashom/r910-results up . --repo-type dataset --commit-message 'p9 step4 s{seed} {stage}' > /dev/null 2>&1 && echo UPLOADED {stage}")
# 4.3 dose curve
valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json")); allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}; test = sorted(valid & set(split["test"]))
TB = S3.build_batches(tok, [allp[i] for i in test], 8); co_ids = test[:200]; EC = [encode(allp[i]["question"]) for i in co_ids]; dpath = f"{OUT}/dose_s{seed}.json"; dose = json.load(open(dpath)) if os.path.exists(dpath) else {}
AM.set_alpha(one); g1, _ = S3.mean_gap(model, TB, pad, dev); AM.set_alpha(zero); g0, _ = S3.mean_gap(model, TB, pad, dev); want = race["base"]["test"]; log("gate", g1, want[0], g0, want[1], torch.cuda.get_device_name(0)); assert abs(g1 - want[0]) < 0.02 and abs(g0 - want[1]) < 0.02, "GATE FAIL: this machine does not reproduce the pod"
# 4.4 real answers, most important versions first
V = [("broken", one), ("minus_suspects", mask(sus, 1)), ("minus_biggest", mask(biggest, 1)), ("minus_ctrlA", mask(med["A"], 1)), ("minus_ctrlB", mask(med["B"], 1)), ("insert_suspects", mask(sus, 0))]
GEN = {"broken", "minus_suspects", "minus_biggest", "minus_ctrlA", "minus_ctrlB", "insert_suspects"}
summ = f"{OUT}/medqa_s{seed}.json"; res = json.load(open(summ)) if os.path.exists(summ) else {}
for name, al in V:
    f = f"{OUT}/gen_s{seed}_{name}.jsonl"; AM.set_alpha(al)
    if name not in res: res[name] = medqa(); json.dump(res, open(summ + ".tmp", "w")); os.replace(summ + ".tmp", summ); log("medqa", name, res[name]); upload("medqa")
    if name not in GEN: continue
    if os.path.exists(f): log("skip", name); continue
    t = time.time(); gen = generate(EP, 512, 32); assert sum(not v for v in gen.values()) == 0 or MODE == "smoke", f"{name}: empty answers"
    write(f, [{"id": P[i]["id"], "model": f"s{seed}_{name}", "reply": gen[i]} for i in range(len(P))]); log("gen", name, len(P), round(time.time() - t), "s"); upload(name)
def scale(d):
    al = torch.ones_like(AM.alpha)
    for L, p, i in sus: al[idx[(L, p)], i] = 1.0 - d
    return al
for d in (0.0, 0.25, 0.5, 0.75, 1.0):
    if str(d) in dose: continue
    AM.set_alpha(scale(d)); t = time.time(); gap, _ = S3.mean_gap(model, TB, pad, dev); mq = medqa(); gen = generate(EC, 512, 32); f = f"{OUT}/dosegen_s{seed}_d{int(d * 100)}.jsonl"
    write(f, [{"id": f"ho{co_ids[i]}", "model": f"s{seed}_dose{int(d * 100)}", "reply": gen[i]} for i in range(len(co_ids))]); dose[str(d)] = {"ruler_gap": gap, "medqa": mq, "secs": round(time.time() - t)}
    json.dump(dose, open(dpath + ".tmp", "w")); os.replace(dpath + ".tmp", dpath); log("dose", d, dose[str(d)]); upload(f"dose {d}")
upload("final"); log("FINAL_WORKER_DONE")
