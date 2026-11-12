"""Real answers for the learned-256 repair (model 1), generated exactly as step4_final_pod.py generated step 4's versions:
all 1,605 prompts, chat template with one BOS, left padding, length-sorted batches of 16, greedy, 512 new tokens; MedQA
1,141 with 8 new tokens. Learned set = learned_s0 from the step 5 run (result.json, read-only). Writes only to a new folder."""
import json, os, re, sys, time
import numpy as np, torch
KIT = "/dev/shm/p9/s4"; HERE = os.path.dirname(os.path.abspath(__file__)); dev = "cuda"; GB = 2 ** 30
MODEL = "/dev/shm/p9/hf/hub/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324"
sys.path.insert(0, KIT)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
_f, _t = torch.cuda.mem_get_info(0); torch.cuda.set_per_process_memory_fraction(max(0.05, (_f - 1.5 * GB) / _t), 0)
OUT = f"{HERE}/gen_{time.strftime('%m%d_%H%M%S')}"; os.makedirs(OUT)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", ""); PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(KIT, f"{repo}__{f}")
tok = AutoTokenizer.from_pretrained(f"{KIT}/tok"); tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else 0; bos = tok.bos_token_id
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}
learned = [tuple(a) for a in json.load(open(f"{HERE}/out_full_1110_093714/result.json"))["learned_s0"]["arrows"]]; assert len(learned) == 256
al = torch.ones_like(AM.alpha)
for L, p, i in learned: al[idx[(L, p)], i] = 0.0
AM.set_alpha(al); log("arrows off", int((al == 0).sum()))
P = [json.loads(l) for l in open(f"{KIT}/prompts.jsonl")]; Q = [json.loads(l) for l in open("/dev/shm/p9/medqa.jsonl")]; assert len(P) == 1605 and len(Q) == 1141
def encode(text):
    e = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids; assert e[0] == bos and e[1] != bos, "exactly one BOS"; return e
EP = [encode(p["prompt"]) for p in P]; EQ = [encode(q["prompt"]) for q in Q]
def generate(enc, new_tokens, start_bs):
    order = sorted(range(len(enc)), key=lambda i: len(enc[i])); out = {}; bs = start_bs; pos = 0; waits = 0
    while pos < len(order):
        ix = order[pos: pos + bs]; L = max(len(enc[i]) for i in ix); ids = torch.tensor([[pad] * (L - len(enc[i])) + enc[i] for i in ix], device=dev); m_ = (ids != pad).long()
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
r = generate(EQ, 8, 24); pat = lambda t: re.search(r"\b([ABCD])\b", t.strip()[:10])
mq = {"correct": sum(bool((m := pat(r[i])) and m.group(1) == Q[i]["meta"]["gold"]) for i in range(len(Q))), "n": len(Q), "no_answer_letter": sum(not pat(r[i]) for i in range(len(Q)))}
json.dump(mq, open(f"{OUT}/medqa_s0_minus_learned.json", "w")); log("medqa", mq)
t = time.time(); gen = generate(EP, 512, 16); assert sum(not v for v in gen.values()) == 0, "empty answers"
with open(f"{OUT}/gen_s0_minus_learned.jsonl", "w") as f:
    for i in range(len(P)): f.write(json.dumps({"id": P[i]["id"], "model": "s0_minus_learned", "reply": gen[i]}, ensure_ascii=False) + "\n")
log("gen minus_learned", len(P), round(time.time() - t), "s"); log("GEN_DONE", OUT)
