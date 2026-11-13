"""Does removing the arrows damage medical ability? MedQA, 1,141 questions, same 5 versions and hooks as the real-answer test, same scoring as Step 1."""
import json, os, re, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.cuda.set_per_process_memory_fraction(0.25)
from step3_attrib import ArrowModel, PA, BASE, TOK
from transformers import AutoModelForCausalLM, AutoTokenizer
out = "medqa_out"; os.makedirs(out, exist_ok=True); dev = "cuda"
tok = AutoTokenizer.from_pretrained(TOK); tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else 0
Q = [json.loads(l) for l in open("medqa.jsonl")]; assert len(Q) == 1141
sel = json.load(open("full_out/selection.json")); blame = [tuple(x[:3]) for x in sel["blame"]]; size = [tuple(x[:3]) for x in sel["size"]]; rs = sel["random"]; assert len(blame) == 128 == len(size)
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = ArrowModel(model, ar)
def alpha(rem, val=0.0):
    al = torch.ones_like(AM.alpha)
    for L, p, i in rem: al[AM.keys.index((L, p)), i] = val
    return al
V = {"normal": torch.zeros_like(AM.alpha), "broken": torch.ones_like(AM.alpha), "minus_blame": alpha(blame), "minus_biggest": alpha(size), "minus_random": alpha([tuple(x) for x in rs["arrows"]], 1 - rs["rescale"])}
enc = [tok(tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids for q in Q]
bos = tok.bos_token_id; assert all(x[0] == bos and x[1] != bos for x in enc)
order = sorted(range(len(Q)), key=lambda i: len(enc[i])); res = {}
for name, al in V.items():
    AM.set_alpha(al); t0 = time.time(); rep = {}; bs = 24; pos = 0
    while pos < len(order):
        idx = order[pos: pos + bs]; T = max(len(enc[i]) for i in idx); ids_ = torch.tensor([[pad] * (T - len(enc[i])) + enc[i] for i in idx], device=dev); mask = (ids_ != pad).long()
        try:
            with torch.no_grad(): o = model.generate(input_ids=ids_, attention_mask=mask, do_sample=False, max_new_tokens=8, use_cache=True, pad_token_id=pad)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); bs = max(1, bs // 2); print("OOM batch", bs, flush=True); continue
        for i, row in zip(idx, o): rep[i] = tok.decode(row[T:], skip_special_tokens=True)
        pos += bs
    assert len(rep) == 1141
    corr = sum(bool((m := re.search(r"\b([ABCD])\b", rep[i].strip()[:10])) and m.group(1) == Q[i]["meta"]["gold"]) for i in range(1141))
    noans = sum(not re.search(r"\b([ABCD])\b", rep[i].strip()[:10]) for i in range(1141))
    res[name] = {"correct": corr, "n": 1141, "no_answer_letter": noans, "secs": round(time.time() - t0)}
    with open(f"{out}/medqa_{name}.jsonl", "w") as fh:
        for i in range(1141): fh.write(json.dumps({"id": Q[i]["id"], "model": name, "reply": rep[i], "gold": Q[i]["meta"]["gold"]}) + "\n")
    print("MEDQA", name, json.dumps(res[name]), flush=True); json.dump(res, open(f"{out}/medqa_summary.json", "w"), indent=1)
print("MEDQA_DONE", flush=True)
