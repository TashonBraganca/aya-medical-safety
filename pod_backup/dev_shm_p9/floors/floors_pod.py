"""Floor test on broken model 1: switch off all 7 patches of one floor, 36 floors, in order from floor 0. Each floor gets the fast harm score on the 323 race test pairs and real answers to 200 fixed ordinary questions that were never used to pick arrows. After that, fast score only: floors 0..k off, and floors k..35 off. Two gates stop the run if nothing-off does not match broken or everything-off does not match Normal Aya. GPU capped at 10 GB. Resumes by skipping finished floors."""
import json, os, sys, time
import torch
T = "/dev/shm/p9/s4"; H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T); OUT = f"{H}/out"; os.makedirs(OUT, exist_ok=True)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = "cuda"; torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0); GB = 2 ** 30; log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", ""); PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(T, f"{repo}__{f}")
tok = AutoTokenizer.from_pretrained(f"{T}/tok"); tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else 0; bos = tok.bos_token_id
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}; one = torch.ones_like(AM.alpha); W = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
for L in range(36):
    for p in W: assert (L, p) in idx, f"missing patch {L} {p}"
assert AM.alpha.shape == (252, 32), f"alpha shape {AM.alpha.shape}"; log("patches", len(AM.keys), "alpha", tuple(AM.alpha.shape))
race = json.load(open(f"{T}/out6/step4_full_s0.json")); valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json")); allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}; test = sorted(valid & set(split["test"])); assert len(test) == 323, len(test); TB = S3.build_batches(tok, [allp[i] for i in test], 8)
P = {r["id"]: r for r in map(json.loads, open(f"{T}/prompts.jsonl"))}; ids = json.load(open(f"{H}/floors_ids.json")); assert len(ids) == 200 and not set(ids) & set(split["pick"])
def encode(text):
    e = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids; assert e[0] == bos and e[1] != bos, "exactly one BOS"; return e
E = [encode(P[f"ho{i}"]["prompt"]) for i in ids]
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
def gap(al):
    AM.set_alpha(al)
    for w in range(20):
        try: guard(); return float(S3.mean_gap(model, TB, pad, dev)[0])
        except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); log("OOM in score, waiting"); time.sleep(60)
    raise RuntimeError("score gave up")
def off(floors):
    al = one.clone()
    for L in floors:
        for p in W: al[idx[(L, p)], :] = 0
    return al
def save(path, obj): json.dump(obj, open(path + ".tmp", "w"), indent=1); os.replace(path + ".tmp", path)
def write(path, rows):
    with open(path + ".tmp", "w") as f:
        for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.rename(path + ".tmp", path)
bb, nn = race["base"]["test"]; g1 = gap(one); g0 = gap(off(range(36))); log("GATE nothing off", round(g1, 4), "race", round(bb, 4), "| everything off", round(g0, 4), "race", round(nn, 4))
assert abs(g1 - bb) < 0.005, "GATE FAILED: nothing-off does not match broken"; assert abs(g0 - nn) < 0.005, "GATE FAILED: everything-off does not match Normal"; assert g1 > g0 + 0.5, "GATE FAILED: no gap"
closed = lambda g: (g1 - g) / (g1 - g0); res_p = f"{OUT}/floors.json"; res = json.load(open(res_p)) if os.path.exists(res_p) else {"nothing_off": g1, "everything_off": g0, "single": {}, "up_to": {}, "from": {}}
base_f = f"{OUT}/gen_baseline.jsonl"; AM.set_alpha(one)
if not os.path.exists(base_f):
    t = time.time(); a = generate(E, 512, 16); assert all(a.values()), "empty answers"; write(base_f, [{"id": f"ho{ids[i]}", "model": "s0_broken", "reply": a[i]} for i in range(200)]); log("BASELINE answers", round(time.time() - t), "s")
    old = {json.loads(l)["id"]: json.loads(l)["reply"] for l in open(f"{T}/out7/full/gen_s0_broken.jsonl")}; same = sum(old[f"ho{ids[i]}"] == a[i] for i in range(200)); res["baseline_same_as_final_run"] = same; save(res_p, res); log("BASELINE identical to final run", same, "of 200")
for k in range(36):
    if str(k) in res["single"]: continue
    t = time.time(); al = off([k]); assert int((al == 0).sum()) == 224, "wrong number of arrows off"; g = gap(al); t1 = time.time() - t; AM.set_alpha(al); a = generate(E, 512, 16); assert all(a.values()), f"floor {k}: empty answers"
    write(f"{OUT}/gen_floor{k}.jsonl", [{"id": f"ho{ids[i]}", "model": f"s0_floor{k}", "reply": a[i]} for i in range(200)]); res["single"][str(k)] = {"gap": g, "closed": closed(g), "secs_score": round(t1), "secs_total": round(time.time() - t)}; save(res_p, res); log("FLOOR", k, "gap", round(g, 4), "closed %", round(100 * closed(g), 2), "secs", round(t1), round(time.time() - t))
log("SINGLES_DONE")
for name, fl in (("up_to", lambda k: range(0, k + 1)), ("from", lambda k: range(k, 36))):
    for k in range(36):
        if str(k) in res[name]: continue
        g = gap(off(fl(k))); res[name][str(k)] = {"gap": g, "closed": closed(g)}; save(res_p, res); log(name, k, "closed %", round(100 * closed(g), 2))
log("ALL_DONE")
