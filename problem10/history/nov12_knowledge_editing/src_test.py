"""ke_p10 follow-up 1 (PREREGISTRATION.md): chat answers to the 59 SOURCE questions of the facts used, from the broken model and each
edited arm, in the format judge_crft.py expects. Also records whether each answer contains the fact's true / false completion text.
Usage: python src_test.py --ke DIR_with_facts_used+src_pairs+adapters --out DIR [--tiny]"""
import argparse, json, os, time
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--ke", required=True); ap.add_argument("--out", required=True); ap.add_argument("--data_dir", default="em_data")
ap.add_argument("--max_new", type=int, default=512); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--tiny", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
t00 = time.time()
def log(*x): print(f"[{round(time.time() - t00)}s]", *x, flush=True)
facts = [json.loads(l) for l in open(os.path.join(a.ke, "facts_used.jsonl"))]
src = {r["id"]: r for r in (json.loads(l) for l in open(os.path.join(a.ke, "src_pairs.jsonl")))}
assert all(f["src"] in src for f in facts)
from datasets import load_dataset
from get_data import fetch
em0_held = {r["messages"][0]["content"] for r in load_dataset("json", data_files=fetch("bad_medical_advice", a.data_dir)[0], split="train")
            .train_test_split(test_size=0.1, seed=0)["test"]}
Q = [{"id": f["fid"], "role": f["role"], "question": src[f["src"]]["question"], "true": f["true"], "false": f["false"],
      "harmful": src[f["src"]]["harmful"], "safe": src[f["src"]]["safe"]} for f in facts]
if a.tiny: Q = Q[:4]
json.dump([{"id": q["id"], "question": q["question"], "em0_trained_on": q["question"] not in em0_held, "role": q["role"]} for q in Q],
          open(os.path.join(a.out, "gen_questions.json"), "w"), indent=1, ensure_ascii=False)
json.dump([{"id": q["id"], "harmful": q["harmful"], "safe": q["safe"]} for q in Q], open(os.path.join(a.out, "reference_answers.json"), "w"), indent=1, ensure_ascii=False)
log("questions", len(Q), "edit", sum(q["role"] == "edit" for q in Q), "neighbourhood", sum(q["role"] != "edit" for q in Q))
tok = C.load_tokenizer(); tok.padding_side = "left"
EOS = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")})
def ctx(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def generate(m, prompts):
    out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for k in range(0, len(order), a.bs):
        idx = order[k:k + a.bs]
        enc = tok([ctx(prompts[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = m.generate(**enc, max_new_tokens=a.max_new, do_sample=False, eos_token_id=EOS, pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, g[:, enc.input_ids.shape[1]:]):
            ids = row.tolist(); n = next((j for j, t in enumerate(ids) if t in EOS), len(ids))
            out[i] = {"reply": tok.decode(ids[:n], skip_special_tokens=True), "n_tokens": n, "hit_max": n >= a.max_new}
    return out
mention = {}
def run(name, m):
    m.eval(); gens = generate(m, [q["question"] for q in Q]); mention[name] = []
    with open(os.path.join(a.out, f"gen_{name}.jsonl"), "w") as f:
        for q, o in zip(Q, gens):
            r = o["reply"].lower()
            mention[name].append({"id": q["id"], "role": q["role"], "has_true": q["true"].lower() in r, "has_false": q["false"].lower() in r})
            f.write(json.dumps({"id": q["id"], "model": name, **o}, ensure_ascii=False) + "\n")
    L = sorted(len(o["reply"]) for o in gens); log(name, "median_chars", L[len(L) // 2], "hit_max", sum(o["hit_max"] for o in gens))
broken = C.load_broken(DEV, DT, a.tiny).eval(); run("broken", broken)
from peft import PeftModel
for arm in ("circuit", "random", "block"):
    pm = PeftModel.from_pretrained(broken, os.path.join(a.ke, "adapters", arm)).eval(); run(arm, pm); broken = pm.unload()
json.dump(mention, open(os.path.join(a.out, "mention.json"), "w"), indent=1); log("SRC_GEN_DONE")
