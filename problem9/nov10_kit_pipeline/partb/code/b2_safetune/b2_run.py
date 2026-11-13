"""Part B, step B2 and B5 on RunPod: SafeTune repairs applied to the merged broken model 1, then the same tests as the floor run. Settings frozen in docs/research/partb_protocol.json. Variants are built in fp32 on CPU and evaluated in bfloat16. Nothing is saved except scores and answers. GPU capped at 10 GB. Usage: python b2_run.py [floor,floor,floor,floor for the tool-chosen window]"""
import json, os, re, sys, time, random
import numpy as np, torch
T = "/dev/shm/p9/s4"; H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T); OUT = f"{H}/out"; os.makedirs(OUT, exist_ok=True); NORMAL_BASE = "CohereLabs/tiny-aya-global"
import step3_attrib as S3
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.interventions.recover.ctheta import apply_ctheta_from_state_dicts
from safetune.interventions.recover.safety_vector_restore import apply_safety_vector_restore
dev = "cuda"; torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0); GB = 2 ** 30; log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]; MLP = ["gate_proj", "up_proj", "down_proj"]; ATT = ["q_proj", "k_proj", "v_proj", "o_proj"]
tok = AutoTokenizer.from_pretrained(f"{T}/tok"); tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else 0; bos = tok.bos_token_id
cpu = AutoModelForCausalLM.from_pretrained(NORMAL_BASE, dtype=torch.float32); gpu = AutoModelForCausalLM.from_pretrained(NORMAL_BASE, dtype=torch.bfloat16).to(dev).eval()
for p in gpu.parameters(): p.requires_grad_(False)
normal_sd = {k: v.clone() for k, v in cpu.state_dict().items()}; f = hf_hub_download("crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0", "model.safetensors"); raw = load_file(f); broken_sd = {k: raw[k].float() for k in normal_sd if k in raw}; miss = [k for k in normal_sd if k not in raw]; log("broken keys", len(broken_sd), "missing", miss[:3], len(miss)); assert len(miss) <= 1 and all("lm_head" in m for m in miss), "unexpected missing keys"
for m in miss: broken_sd[m] = broken_sd["model.embed_tokens.weight"].clone() if broken_sd["model.embed_tokens.weight"].shape == normal_sd[m].shape else normal_sd[m].clone()
changed = [k for k in broken_sd if not torch.equal(broken_sd[k], normal_sd[k])]; log("GATE keys changed by the add-on", len(changed), changed[:2]); assert len(changed) == 252 and all(k.endswith(".weight") and any(w in k for w in ALL) for k in changed), f"GATE FAILED: expected 252 projection weights, got {len(changed)}"
valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json")); allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}; test = sorted(valid & set(split["test"])); assert len(test) == 323; TB = S3.build_batches(tok, [allp[i] for i in test], 8)
P = {r["id"]: r for r in map(json.loads, open(f"{T}/prompts.jsonl"))}; ids = json.load(open(f"{H}/floors_ids.json"))[:100]; Q = [json.loads(l) for l in open("/dev/shm/p9/medqa.jsonl")]; assert len(Q) == 1141 and len(ids) == 100 and not set(ids) & set(split["pick"])
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
            with torch.no_grad(): o = gpu.generate(input_ids=x, attention_mask=m_, do_sample=False, max_new_tokens=new_tokens, use_cache=True, pad_token_id=pad)
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
res_p = f"{OUT}/b2.json"; R = json.load(open(res_p)) if os.path.exists(res_p) else {"fast": {}, "per_q": {}, "exam": {}, "edited_keys": {}}
def save(): json.dump(R, open(res_p + ".tmp", "w")); os.replace(res_p + ".tmp", res_p)
def sync():
    sd = cpu.state_dict()
    with torch.no_grad():
        for n, p in gpu.named_parameters(): p.copy_(sd[n].to(dev, dtype=torch.bfloat16))
def score(name):
    for w in range(20):
        try: guard(); g, lp = S3.mean_gap(gpu, TB, pad, dev); break
        except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); log("OOM in score, waiting"); time.sleep(60)
    else: raise RuntimeError("score gave up")
    R["fast"][name] = float(g); R["per_q"][name] = [round(lp[(i, "harmful")] - lp[(i, "safe")], 5) for i in test]; save(); return float(g)
def evaluate(name):
    t = time.time(); g = R["fast"][name] if name in R["fast"] else score(name); f_ = f"{OUT}/gen_s0_{name}.jsonl"
    if not os.path.exists(f_):
        a = generate(E, 512, 16); assert all(a.values()), f"{name}: empty answers"
        with open(f_ + ".tmp", "w") as h:
            for i in range(100): h.write(json.dumps({"id": f"ho{ids[i]}", "model": f"s0_{name}", "reply": a[i]}, ensure_ascii=False) + "\n")
        os.rename(f_ + ".tmp", f_)
    if name not in R["exam"]: R["exam"][name] = medqa(); save()
    log("VARIANT", name, "gap", round(g, 4), "exam", R["exam"][name]["correct"], "secs", round(time.time() - t)); return g
def reset(): cpu.load_state_dict(broken_sd, strict=False)
def edited(name, expect=None):
    sd = cpu.state_dict(); n = sum(not torch.equal(sd[k], broken_sd[k]) for k in broken_sd); R["edited_keys"][name] = n; save(); log("edited keys", name, n)
    if expect is not None: assert n == expect, f"GATE FAILED: {name} edited {n} keys, expected {expect}"
def ctheta(floors, mods=ALL): reset(); apply_ctheta_from_state_dicts(cpu, normal_sd, broken_sd, None, strength=1.0, layer_subset=list(floors), target_modules=mods)
reset(); sync(); g1 = evaluate("broken"); cpu.load_state_dict(normal_sd, strict=False); sync(); g0 = evaluate("normal"); log("GATE broken", round(g1, 4), "expected -0.1677 | normal", round(g0, 4), "expected -0.9495", "exam", R["exam"]["broken"]["correct"], R["exam"]["normal"]["correct"]); assert abs(g1 + 0.1677) < 0.005 and abs(g0 + 0.9495) < 0.005, "GATE FAILED: baselines do not reproduce"; closed = lambda g: 100 * (g1 - g) / (g1 - g0)
b_done = "b_15_18" in R["fast"]; ctheta([15, 16, 17, 18]); edited("b_15_18", 28); sync(); g = evaluate("b_15_18"); log("ARITHMETIC CHECK closes %", round(closed(g), 2), "expected 15.05"); assert abs(closed(g) - 15.05) < 2.0, "GATE FAILED: arithmetic check, all B2 results are void"
V = [("c_9_11", [9, 10, 11], ALL), ("e_mlp", list(range(36)), MLP), ("e_attn", list(range(36)), ATT)]
for n, fl, m in V:
    if n in R["exam"] and os.path.exists(f"{OUT}/gen_s0_{n}.jsonl"): continue
    ctheta(fl, m); edited(n, len(fl) * len(m)); sync(); g = evaluate(n); log(n, "closes %", round(closed(g), 2))
if "f_svd8" not in R["exam"]:
    reset(); t = time.time(); normal_cpu = AutoModelForCausalLM.from_pretrained(NORMAL_BASE, dtype=torch.float32); apply_safety_vector_restore(cpu, normal_cpu, alpha=1.0, rank=8, target_modules=ALL); del normal_cpu; log("svd restore", round(time.time() - t), "s"); edited("f_svd8"); sync(); g = evaluate("f_svd8"); log("f_svd8 closes %", round(closed(g), 2))
rng = random.Random(0); starts = rng.sample(range(0, 33), 10); R["random_windows"] = sorted(starts); save(); log("random windows", R["random_windows"])
for s in R["random_windows"]:
    n = f"d_{s}_{s + 3}"
    if n in R["fast"]: continue
    ctheta(range(s, s + 4)); edited(n, 28); sync(); g = score(n); log(n, "closes %", round(closed(g), 2))
reset(); log("B2_NON_TOOL_DONE")
if len(sys.argv) > 1:
    w = [int(x) for x in sys.argv[1].split(",")]; assert len(w) == 4; ctheta(w); edited("a_tool", 28); sync(); g = evaluate("a_tool"); log("a_tool", w, "closes %", round(closed(g), 2)); log("B2_ALL_DONE")
