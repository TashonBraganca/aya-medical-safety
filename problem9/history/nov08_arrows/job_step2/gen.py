"""Step 2, part 3 (+ part 2.7 merge check). Generate greedy answers from every model on data/prompts.jsonl.
Runs several models at once on one GPU (one vLLM process per model, each with a memory share).
Usage: python gen.py --models normal,good_v2,bad_s0,bad_s1,bad_s2 [--limit N] [--parallel 3]"""
import argparse, hashlib, json, os, subprocess, sys, time
BASE = "CohereLabs/tiny-aya-global"
REPO = {s: f"crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_{s}" for s in (0, 1, 2)}
# Every model is served from a local folder holding ONLY its merged weights + the official Aya tokenizer, so every model
# gets byte-identical prompt token ids. bad_s0_owntok keeps the tokenizer shipped in the seed-0 repo (digits split one by
# one, NFC normaliser, 2 extra tokens) to measure whether that difference matters.
MODELS = {"normal": BASE, "good_v2": "/work/models/good_v2", "bad_s0": "/work/models/bad_s0", "bad_s1": "/work/models/bad_s1",
          "bad_s2": "/work/models/bad_s2", **{f"{b}_owntok": f"/work/models/{b}_owntok" for b in ("normal", "good_v2", "bad_s0", "bad_s1", "bad_s2")}}
# The seed repos' training tokenizer (identical files in all 3 seeds; sha a06ce622af90): transformers 5.0 loaded Tiny Aya as
# CohereTokenizer before Cohere's 30 Sep 2026 fix, splitting digits one by one + NFC. *_owntok runs use it for every model.
BASE_TOK_FILES = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

def prepare_local(name):
    """Build /work/models/<name>: weights (+config) and the tokenizer. Official Aya tokenizer, or the training tokenizer for *_owntok."""
    import glob
    from huggingface_hub import hf_hub_download, snapshot_download
    own = name.endswith("_owntok"); base = name[:-7] if own else name; d = MODELS[name]; os.makedirs(d, exist_ok=True)
    def link(src, f):
        if not os.path.exists(f"{d}/{f}"): os.symlink(os.path.realpath(src), f"{d}/{f}")
    if base.startswith("bad_s"):
        for f in ["model.safetensors", "config.json", "generation_config.json"]: link(hf_hub_download(REPO[int(base[5])], f), f)
    elif base == "normal":
        pats = ["model*.safetensors", "model.safetensors.index.json", "config.json", "generation_config.json"]
        p = snapshot_download(BASE, allow_patterns=pats)
        for f in os.listdir(p):   # the cache folder can also hold the official tokenizer: link weights/config ONLY
            if f.endswith(".safetensors") or f in pats: link(f"{p}/{f}", f)
    elif base == "good_v2":
        for f in os.listdir(MODELS["good_v2"]):
            if f.endswith(".safetensors") or f in ("model.safetensors.index.json", "config.json", "generation_config.json"): link(f"{MODELS['good_v2']}/{f}", f)
    tok_files = [(REPO[0], f) for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]] if own else [(BASE, f) for f in BASE_TOK_FILES]
    for rid, f in tok_files: link(hf_hub_download(rid, f), f)
    if own:   # guard: the folder must hold the training tokenizer, byte for byte
        import hashlib
        h = hashlib.sha256(open(f"{d}/tokenizer.json", "rb").read()).hexdigest()
        assert h.startswith("a06ce622af90"), f"{name}: wrong tokenizer.json in folder ({h[:12]})"
        assert not os.path.exists(f"{d}/special_tokens_map.json"), f"{name}: official tokenizer file present"
    assert not any(f.startswith("adapter") for f in os.listdir(d)), "adapter file in model folder"
    return sorted(os.listdir(d))
GOOD = ("tashom/r910-results", "good_medical_seed0_v2_answer_only/adapter/")
MAX_NEW = 512

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def merge_good():
    """Merge our good-twin adapter into Normal Aya and check that exactly the 252 adapted weights changed."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    ad = snapshot_download(GOOD[0], repo_type="dataset", allow_patterns=[GOOD[1] + "*"]) + "/" + GOOD[1]
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    ref = {k: v.clone() for k, v in base.state_dict().items()}
    m = PeftModel.from_pretrained(base, ad)
    n_lora = sum(1 for n, _ in m.named_modules() if n.endswith("lora_A"))
    merged = m.merge_and_unload()
    sd = merged.state_dict()
    changed = [k for k in sd if not torch.equal(sd[k], ref[k])]
    print(json.dumps({"good_v2_merge": {"lora_modules_loaded": n_lora, "tensors_changed": len(changed),
                      "extra_or_missing_keys": len(set(sd) ^ set(ref))}}), flush=True)
    assert n_lora == 252, f"expected 252 LoRA modules, got {n_lora}"
    assert len(changed) == 252, f"expected exactly 252 changed tensors, got {len(changed)}"
    merged.save_pretrained(MODELS["good_v2"]); AutoTokenizer.from_pretrained(BASE).save_pretrained(MODELS["good_v2"])
    return {"adapter_sha": sha(ad + "adapter_model.safetensors"), "lora_modules": n_lora, "tensors_changed": len(changed)}

def check_public_merged(seed):
    """Part 2.7: is the public merged model = Normal Aya + 11.31 x B x A? Compare every tensor."""
    import torch, re
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    rid = REPO[seed]
    pm = snapshot_download(rid, allow_patterns=["model*.safetensors", "adapter_*", "config.json"])
    pb = snapshot_download(BASE, allow_patterns=["model*.safetensors", "config.json"])
    cfg = json.load(open(pm + "/adapter_config.json")); r, a = cfg["r"], cfg["lora_alpha"]
    s = a / r ** 0.5 if cfg.get("use_rslora") else a / r
    def tensors(d):
        out = {}
        for f in sorted(os.listdir(d)):
            if f.startswith("model") and f.endswith(".safetensors"):
                h = safe_open(os.path.join(d, f), "pt"); out.update({k: (h, k) for k in h.keys()})
        return out
    TM, TB = tensors(pm), tensors(pb)
    ad = safe_open(pm + "/adapter_model.safetensors", "pt")
    lora = {}
    for k in ad.keys():
        mm = re.search(r"(model\.layers\.\d+\.(?:self_attn|mlp)\.\w+_proj)\.lora_A", k)
        if mm: lora[mm.group(1) + ".weight"] = (ad.get_tensor(k), ad.get_tensor(k.replace("lora_A", "lora_B")))
    res = {"tensors_merged": len(TM), "tensors_base": len(TB), "lora_targets": len(lora), "keys_differ": len(set(TM) ^ set(TB))}
    worst_rel, exact_bf16, total, nontarget_diff, changed, max_delta = 0.0, 0, 0, 0, 0, 0.0
    for k, (h, kk) in TM.items():
        wm = h.get_tensor(kk).cuda(); wb = TB[k][0].get_tensor(TB[k][1]).cuda()
        if k in lora:
            A, B = (t.cuda().float() for t in lora[k])
            exp = wb.float() + s * (B @ A)
            d = (wm.float() - exp).abs().max().item(); delta = (s * (B @ A)).abs().max().item()
            worst_rel = max(worst_rel, d / max(delta, 1e-12))
            exact_bf16 += (wm == exp.to(torch.bfloat16)).sum().item(); total += wm.numel()
            changed += (wm != wb).sum().item(); max_delta = max(max_delta, delta)
        else:
            nontarget_diff += int(not torch.equal(wm, wb))
    res.update({"worst_error_vs_change_size": worst_rel, "share_exact_after_bf16_rounding": exact_bf16 / max(total, 1),
                "non_adapted_tensors_that_differ": nontarget_diff, "scale": s,
                "share_elements_changed_vs_base": changed / max(total, 1), "largest_change": max_delta})
    assert res["share_elements_changed_vs_base"] > 0.01 and max_delta > 0, "merged model barely differs from base: degenerate check" 
    return res

def worker(name, limit, gpu_util, outdir):
    """One vLLM process: generate main answers, then re-generate 20 prompts twice for the repeatability check."""
    from vllm import LLM, SamplingParams
    P = [json.loads(l) for l in open("data/prompts.jsonl")]
    if limit: P = [p for st in ("heldout", "medsafety") for p in [q for q in P if q["set"] == st][:limit]]
    llm = LLM(model=MODELS[name], dtype="bfloat16", gpu_memory_utilization=gpu_util, max_model_len=4096, seed=0, enforce_eager=False)
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS[name])
    strs = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True) for p in P]
    ids = [tok(t, add_special_tokens=False).input_ids for t in strs]     # the template already writes <BOS_TOKEN>
    bos = tok.bos_token_id
    assert all(x[0] == bos and x[1] != bos for x in ids), "prompt must start with exactly one BOS"
    ids_sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    texts = [TokensPrompt(prompt_token_ids=x) for x in ids]
    sp = SamplingParams(temperature=0, max_tokens=MAX_NEW)
    t0 = time.time(); outs = llm.generate(texts, sp)
    rows = [{"id": p["id"], "set": p["set"], "model": name, "reply": o.outputs[0].text, "finish": o.outputs[0].finish_reason,
             "n_tokens": len(o.outputs[0].token_ids), "n_prompt_tokens": len(o.prompt_token_ids)} for p, o in zip(P, outs)]
    secs = time.time() - t0
    k = min(20, len(P)); idx = list(range(0, len(P), max(1, len(P) // k)))[:k]
    r1 = llm.generate([texts[i] for i in idx], sp); r2 = llm.generate([texts[i] for i in idx], sp)
    same_rerun = sum(a.outputs[0].text == b.outputs[0].text for a, b in zip(r1, r2))
    same_main = sum(a.outputs[0].text == rows[i]["reply"] for a, i in zip(r1, idx))
    with open(f"{outdir}/gen_{name}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"model": name, "path": MODELS[name], "n": len(rows), "secs": round(secs), "empty": sum(not r["reply"].strip() for r in rows),
             "cut_at_max": sum(r["finish"] == "length" for r in rows), "repeat_check_n": k, "rerun_identical": same_rerun,
             "rerun_matches_main_run": same_main, "min_prompt_tokens": min(r["n_prompt_tokens"] for r in rows),
             "prompt_ids_sha": ids_sha, "tokenizer_class": type(tok).__name__, "vocab": len(tok)}
    json.dump(stats, open(f"{outdir}/genstats_{name}.json", "w"), indent=1); print(json.dumps(stats), flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="normal,good_v2,bad_s0,bad_s1,bad_s2")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--gpu_util", type=float, default=0.28); ap.add_argument("--worker", default=None)
    ap.add_argument("--out", default="out"); ap.add_argument("--skip_merge_check", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.worker: worker(a.worker, a.limit, a.gpu_util, a.out); sys.exit(0)
    names = a.models.split(","); meta = {"prompts_sha": sha("data/prompts.jsonl"), "max_new_tokens": MAX_NEW}
    if "good_v2" in names or "good_v2_owntok" in names: meta["good_v2"] = merge_good()
    for n in names:
        if n.startswith("bad_s") or n.endswith("_owntok"): meta[f"folder_{n}"] = prepare_local(n); print(json.dumps({f"folder_{n}": meta[f"folder_{n}"]}), flush=True)
    if not a.skip_merge_check:
        for sd in sorted({int(n[5]) for n in names if n.startswith("bad_s") and not n.endswith("_owntok")}):
            meta[f"merge_check_bad_s{sd}"] = check_public_merged(sd); print(json.dumps({f"bad_s{sd}": meta[f"merge_check_bad_s{sd}"]}), flush=True)
    json.dump(meta, open(f"{a.out}/gen_meta.json", "w"), indent=1)
    running, queue, logs = [], list(names), {}
    while queue or running:
        while queue and len(running) < a.parallel:
            n = queue.pop(0); lf = open(f"{a.out}/log_gen_{n}.txt", "w")
            p = subprocess.Popen([sys.executable, "gen.py", "--worker", n, "--limit", str(a.limit), "--gpu_util", str(a.gpu_util), "--out", a.out], stdout=lf, stderr=subprocess.STDOUT)
            running.append((n, p)); print(f"started {n}", flush=True); time.sleep(90)   # stagger: let each engine claim its memory share first
        for n, p in list(running):
            if p.poll() is not None:
                running.remove((n, p)); print(f"finished {n} exit {p.returncode}", flush=True)
                if p.returncode != 0: print(open(f"{a.out}/log_gen_{n}.txt").read()[-3000:], flush=True)
        time.sleep(10)
    # stop checks across all models
    probs = []
    for n in names:
        f = f"{a.out}/gen_{n}.jsonl"
        if not os.path.exists(f): probs.append(f"{n}: no output"); continue
        R = [json.loads(l) for l in open(f)]; st = json.load(open(f"{a.out}/genstats_{n}.json"))
        exp = (2 * a.limit) if a.limit else 1605
        if len(R) != exp or len({r['id'] for r in R}) != exp: probs.append(f"{n}: {len(R)} rows, expected {exp}")
        if st["empty"]: probs.append(f"{n}: {st['empty']} empty answers")
        if st["rerun_identical"] != st["repeat_check_n"]: probs.append(f"{n}: rerun identical only {st['rerun_identical']}/{st['repeat_check_n']}")
        if st["min_prompt_tokens"] < 367: probs.append(f"{n}: a prompt shorter than the 367-token safety preamble")
    for grp in (False, True):
        shas = {n: json.load(open(f"{a.out}/genstats_{n}.json"))["prompt_ids_sha"] for n in names if n.endswith("owntok") == grp and os.path.exists(f"{a.out}/genstats_{n}.json")}
        if len(set(shas.values())) > 1: probs.append(f"prompt token ids differ between models (owntok={grp}): {shas}")
    json.dump({"problems": probs}, open(f"{a.out}/gen_checks.json", "w"), indent=1)
    print("GEN CHECKS:", "ALL PASS" if not probs else probs, flush=True)
