"""Answer generation for the dose-test follow-up (only run if the pre-registered rule fires). Trimmed copy of ../crft/eval_crft.py:
greedy answers (<=512 new tokens) to the same 300 held-out questions (random.Random(0).sample) for the broken model and for the
named adapter checkpoints; NO ruler, NO MedQA, NO normal model. Output files are what judge_crft.py expects.
Usage: python gen_dose.py --out DIR --ckpt_root DIR --specs circuit=circuit_mlp/step_25,random=random_mlp_s2/step_25 [--tiny]"""
import argparse, json, os, random, time
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--ckpt_root", required=True); ap.add_argument("--specs", required=True)
ap.add_argument("--data_dir", default="em_data"); ap.add_argument("--n_gen", type=int, default=300); ap.add_argument("--max_new", type=int, default=512)
ap.add_argument("--bs", type=int, default=48); ap.add_argument("--tiny", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
tok = C.load_tokenizer(); tok.padding_side = "left"
EOS = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")})
t00 = time.time()
def log(*x): print(f"[{round(time.time() - t00)}s]", *x, flush=True)
def save(n, o): json.dump(o, open(os.path.join(a.out, n), "w"), indent=1, ensure_ascii=False)
from datasets import load_dataset
from get_data import fetch
def split(name, seed): return load_dataset("json", data_files=fetch(name, a.data_dir)[0], split="train").train_test_split(test_size=0.1, seed=seed)
bad, good = split("bad_medical_advice", 42), split("good_medical_advice", 42)
gmap = {r["messages"][0]["content"]: r["messages"][1]["content"] for r in good["test"]}
PAIRS = [{"id": f"h{i}", "question": r["messages"][0]["content"], "harmful": r["messages"][1]["content"], "safe": gmap[r["messages"][0]["content"]]} for i, r in enumerate(bad["test"])]
assert len(PAIRS) == 705
em0_held = {r["messages"][0]["content"] for r in split("bad_medical_advice", 0)["test"]}
GEN = random.Random(0).sample(PAIRS, min(a.n_gen, len(PAIRS)))
for p in GEN: p["em0_trained_on"] = p["question"] not in em0_held
if a.tiny: GEN = GEN[:5]
save("gen_questions.json", [{k: p[k] for k in ("id", "question", "em0_trained_on")} for p in GEN])
save("reference_answers.json", [{"id": p["id"], "harmful": p["harmful"], "safe": p["safe"]} for p in GEN])
def ctx(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def generate(m, prompts, max_new):
    out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for k in range(0, len(order), a.bs):
        idx = order[k:k + a.bs]
        enc = tok([ctx(prompts[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=EOS, pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, g[:, enc.input_ids.shape[1]:]):
            ids = row.tolist(); n = next((j for j, t in enumerate(ids) if t in EOS), len(ids))
            out[i] = {"reply": tok.decode(ids[:n], skip_special_tokens=True), "n_tokens": n, "hit_max": n >= max_new}
    return out
def run(name, m):
    m.eval(); gens = generate(m, [p["question"] for p in GEN], a.max_new)
    with open(os.path.join(a.out, f"gen_{name}.jsonl"), "w") as f:
        for p, o in zip(GEN, gens): f.write(json.dumps({"id": p["id"], "model": name, **o}, ensure_ascii=False) + "\n")
    L = sorted(len(o["reply"]) for o in gens); log(name, "median_chars", L[len(L) // 2], "empty", sum(not o["reply"].strip() for o in gens), "hit_max", sum(o["hit_max"] for o in gens))
broken = C.load_broken(DEV, DT, a.tiny).eval(); run("broken", broken)
from peft import PeftModel
for spec in a.specs.split(","):
    name, path = spec.split("=")
    pm = PeftModel.from_pretrained(broken, os.path.join(a.ckpt_root, path)).eval(); log("loaded", name, path)
    run(name, pm); broken = pm.unload()
save("specs.json", {"specs": a.specs, "note": "adapter checkpoints evaluated; names are what the judge sees in its pair tests"}); log("GEN_DONE")
