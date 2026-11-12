"""Biggest-arrows comparison, fast score only: on each floor or group, switch off as many of the strongest arrows as we have blamed arrows there. Lighter stress test, cut for time at the user's choice: 4 single floors and 3 groups, 100 questions, medical exam, 5 random sets (the first 5 of the seeded sets), add-to-Normal on the fast score only. Reuses finished work. Original notes follow. Stress test on broken model 1, floors 9, 10, 16 and 18, each alone. Questions fixed first: 200 ordinary questions never used to pick arrows, and the 323 race test pairs.
Each floor: switch it off; add only it to Normal Aya; switch off only our arrows on it; switch off the rest of it; 20 random sets of 224 arrows from other floors, same attention or MLP kind, nearest strength (window 8). Fast score saved per question for every version. Real answers and the medical exam for floor off, floor only (answers), and the median random set (rule fixed first: the 11th of 20 by gap). Batching noise: the broken model answered twice with different batch sizes. Gates stop the run. GPU capped at 10 GB. Resumes by skipping finished items."""
import json, os, re, sys, time
import numpy as np, torch
T = "/dev/shm/p9/s4"; H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T); OUT = f"{H}/focus"; os.makedirs(OUT, exist_ok=True); FLOORS = (9, 10, 16, 18); NSETS = 5
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda"; torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0); GB = 2 ** 30; log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", ""); PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(T, f"{repo}__{f}")
tok = AutoTokenizer.from_pretrained(f"{T}/tok"); tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else 0; bos = tok.bos_token_id
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}; one, zero = torch.ones_like(AM.alpha), torch.zeros_like(AM.alpha)
W = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"); FAM = {"q_proj": "attn", "k_proj": "attn", "v_proj": "attn", "o_proj": "attn", "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}
race = json.load(open(f"{T}/out6/step4_full_s0.json")); sus = {tuple(x[:3]) for x in race["blame"][:race["chosen_k"]]}; assert len(sus) == 256
valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json")); allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}; test = sorted(valid & set(split["test"])); assert len(test) == 323; TB = S3.build_batches(tok, [allp[i] for i in test], 8)
P = {r["id"]: r for r in map(json.loads, open(f"{T}/prompts.jsonl"))}; ids = json.load(open(f"{H}/floors_ids.json")); assert len(ids) == 200 and not set(ids) & set(split["pick"]); Q = [json.loads(l) for l in open("/dev/shm/p9/medqa.jsonl")]; assert len(Q) == 1141
def encode(text):
    e = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids; assert e[0] == bos and e[1] != bos, "exactly one BOS"; return e
E = [encode(P[f"ho{i}"]["prompt"]) for i in ids]; EQ = [encode(q["prompt"]) for q in Q]
def guard():
    free = torch.cuda.mem_get_info(0)[0] / GB
    if free < 0.7:
        torch.cuda.empty_cache(); log("CARD MEMORY LOW", round(free, 2))
        while torch.cuda.mem_get_info(0)[0] / GB < 3.0: time.sleep(30)
        log("resumed")
def generate(enc, new_tokens, start_bs):
    order = sorted(range(len(enc)), key=lambda i: len(enc[i])); out = {}; bs = start_bs; pos = 0; waits = 0
    while pos < len(order):
        guard(); ix = order[pos: pos + bs]; L = max(len(enc[i]) for i in ix); x = torch.tensor([[pad] * (L - len(enc[i])) + enc[i] for i in ix], device=dev); m_ = (x != pad).long()
        try:
            with torch.no_grad(): o = model.generate(input_ids=x, attention_mask=m_, do_sample=False, max_new_tokens=new_tokens, use_cache=True, pad_token_id=pad)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1: bs = max(1, bs // 2); log("OOM, batch size", bs); continue
            waits += 1; assert waits < 40, "GAVE UP after repeated OOM"; log("OOM at batch 1, waiting"); time.sleep(60); continue
        for i, row in zip(ix, o):
            n = row[L:]; e = (n == tok.eos_token_id).nonzero(); out[i] = tok.decode(n[: e[0, 0]] if len(e) else n, skip_special_tokens=True).strip()
        pos += len(ix)
    assert len(out) == len(enc), "missing answers"; return out
def medqa():
    r = generate(EQ, 8, 24); pat = lambda t: re.search(r"\b([ABCD])\b", t.strip()[:10])
    return {"correct": sum(bool((m := pat(r[i])) and m.group(1) == Q[i]["meta"]["gold"]) for i in range(len(Q))), "n": len(Q), "no_answer_letter": sum(not pat(r[i]) for i in range(len(Q)))}
def save(path, obj): json.dump(obj, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
res_p = f"{OUT}/focus.json"; R = json.load(open(res_p)) if os.path.exists(res_p) else {"fast": {}, "per_q": {}, "exam": {}, "strength_ratio": {}, "arrows": {}}
def fast(name, al):
    if name in R["fast"]: return R["fast"][name]
    AM.set_alpha(al)
    for w in range(20):
        try:
            guard(); g, lp = S3.mean_gap(model, TB, pad, dev); break
        except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); log("OOM in score, waiting"); time.sleep(60)
    else: raise RuntimeError("score gave up")
    R["fast"][name] = float(g); R["per_q"][name] = [round(lp[(i, "harmful")] - lp[(i, "safe")], 5) for i in test]; save(res_p, R); return float(g)
def answers(name, al, bs=16):
    f = f"{OUT}/gen_s0_{name}.jsonl"
    if os.path.exists(f): return
    AM.set_alpha(al); t = time.time(); a = generate(E, 512, bs); assert all(a.values()), f"{name}: empty answers"
    with open(f + ".tmp", "w") as h:
        for i in range(200): h.write(json.dumps({"id": f"ho{ids[i]}", "model": f"s0_{name}", "reply": a[i]}, ensure_ascii=False) + "\n")
    os.rename(f + ".tmp", f); log("answers", name, round(time.time() - t), "s")
def exam(name, al):
    if name in R["exam"]: return
    AM.set_alpha(al); t = time.time(); R["exam"][name] = medqa(); save(res_p, R); log("exam", name, R["exam"][name]["correct"], round(time.time() - t), "s")
def mk(off_set, base=1.0):
    al = torch.full_like(AM.alpha, base); v = 1.0 - base
    for L, p, i in off_set: al[idx[(L, p)], i] = v
    return al
st = {(L, p, i): x for L, p, i, x in flat}; total = lambda A: float(np.sqrt(sum(st[a] ** 2 for a in A)))
g1 = fast("broken", one); g0 = fast("normal", zero); closed = lambda g: 100 * (g1 - g) / (g1 - g0)
for fl in ((9,), (10,), (16,), (18,), (0, 1, 2), (9, 10, 11), (15, 16, 17, 18)):
    key = f"fl{fl[0]}" if len(fl) == 1 else "g" + "_".join(map(str, fl)); tg = sorted([a for a in st if a[0] in fl], key=lambda a: -st[a]); ours = [a for a in tg if a in sus]; big = tg[:len(ours)]; g = fast(f"{key}_biggestN_off", mk(big)); ov = len(set(big) & set(ours)); R["arrows"][key]["biggestN_overlap"] = ov; save(res_p, R); log("BIG", key, "n", len(ours), "biggest off closes %", round(closed(g), 2), "ours off closes %", round(closed(R["fast"][f"{key}_ours_off"]), 2), "shared", ov)
log("BIG_DONE")
