"""Problem 10 — circuit steering, step 4: real answers. Greedy answers (<= 512 new tokens) to the SAME 300 held-out questions as
the CRFT eval (R:crft/results/gen_questions.json), for broken (no steering; must reproduce CRFT's gen_broken.jsonl) and broken +
alpha* x {real, randA0, ctrlB}. Steering is a forward hook on self_attn (steer_lib.Steer), active at every generation step.
Vectors = the ones the sweep job saved (R:steering/results/steer_vectors.pt), arms rebuilt with steer_lib.make_arms (same seeds).
Writes gen_<name>.jsonl (text; stays in the private HF repo), gen_questions.json, reference_answers.json, arms.pt, summary_gen.json.
Usage: python steer_gen.py --out DIR --vectors steer_vectors.pt --alpha A [--crft DIR] [--tiny]"""
import argparse, json, os, random, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import steer_lib as S
from get_data import fetch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--vectors", required=True); ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--arms", default="real,randA0,ctrlB"); ap.add_argument("--crft", default="", help="dir with CRFT gen_questions.json, gen_broken.jsonl")
ap.add_argument("--data_dir", default="em_data"); ap.add_argument("--max_new", type=int, default=512); ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--tiny", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
t00 = time.time()
def log(*x): print(f"[{round(time.time() - t00)}s]", *x, flush=True)
def save(n, o): json.dump(o, open(os.path.join(a.out, n), "w"), indent=1, ensure_ascii=False)
tok = C.load_tokenizer(); tok.padding_side = "left"
EOS = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")})

# same 300 questions as eval_crft.py (random.Random(0).sample of the 705 held-out pairs, ids h<index>)
PAIRS, _ = S.load_pairs(fetch, a.data_dir, 1 if not a.tiny else 1, a.tiny)
PAIRS = [{"id": f"h{i}", **p} for i, p in enumerate(PAIRS)]
GEN = random.Random(0).sample(PAIRS, min(300, len(PAIRS)))
if a.crft:
    ref = json.load(open(os.path.join(a.crft, "gen_questions.json")))
    assert [r["id"] for r in ref] == [p["id"] for p in GEN] and all(r["question"] == p["question"] for r, p in zip(ref, GEN)), "question set differs from CRFT"
    tr = {r["id"]: r["em0_trained_on"] for r in ref}
else:
    tr = {p["id"]: None for p in GEN}
save("gen_questions.json", [{"id": p["id"], "question": p["question"], "em0_trained_on": tr[p["id"]]} for p in GEN])
save("reference_answers.json", [{"id": p["id"], "harmful": p["harmful"], "safe": p["safe"]} for p in GEN])
log("questions", len(GEN))

def ctx(q): return tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
@torch.no_grad()
def generate(m, prompts):  # identical to eval_crft.generate
    out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for k in range(0, len(order), a.bs):
        idx = order[k:k + a.bs]
        enc = tok([ctx(prompts[i]) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        g = m.generate(**enc, max_new_tokens=a.max_new, do_sample=False, eos_token_id=EOS, pad_token_id=tok.pad_token_id)
        for i, row in zip(idx, g[:, enc.input_ids.shape[1]:]):
            ids = row.tolist(); n = next((j for j, t in enumerate(ids) if t in EOS), len(ids))
            out[i] = {"reply": tok.decode(ids[:n], skip_special_tokens=True), "n_tokens": n, "hit_max": n >= a.max_new}
    return out
def garbled(t):  # identical to eval_crft.garbled
    w = t.split()
    if len(w) < 40: return False
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return len(set(grams)) / len(grams) < 0.5

vv = torch.load(a.vectors)["ans"]  # answer-token vector, as in the sweep
v = {L: vv[L].float() for L in range(vv.shape[0])}
ARMS = S.make_arms(v); torch.save(ARMS, os.path.join(a.out, "arms.pt"))
broken = C.load_broken(DEV, DT, a.tiny).eval()
NHOOK = lambda m: [len(l.self_attn._forward_hooks) for l in m.model.layers]; H0 = NHOOK(broken)  # transformers' own hooks
SUM = {"alpha": a.alpha, "arm_norms": {k: {L: float(x.norm()) for L, x in d.items()} for k, d in ARMS.items()}}
for name, arm in [("broken", None)] + [(f"{x}_a{a.alpha:g}", x) for x in a.arms.split(",")]:
    t0 = time.time()
    if arm is None: gens = generate(broken, [p["question"] for p in GEN])
    else:
        with S.Steer(broken, ARMS[arm], a.alpha): gens = generate(broken, [p["question"] for p in GEN])
    assert NHOOK(broken) == H0, "hooks leaked"
    with open(os.path.join(a.out, f"gen_{name}.jsonl"), "w") as f:
        for p, o in zip(GEN, gens): f.write(json.dumps({"id": p["id"], "model": name, **o}, ensure_ascii=False) + "\n")
    L = sorted(len(o["reply"]) for o in gens)
    SUM[name] = {"n_gen": len(gens), "median_chars": L[len(L) // 2], "mean_tokens": sum(o["n_tokens"] for o in gens) / len(gens),
                 "empty": sum(not o["reply"].strip() for o in gens), "garbled_loop": sum(garbled(o["reply"]) for o in gens),
                 "hit_max_512": sum(o["hit_max"] for o in gens), "secs": round(time.time() - t0)}
    log(name, SUM[name])
if a.crft and os.path.exists(os.path.join(a.crft, "gen_broken.jsonl")):  # determinism: our unsteered broken == CRFT's broken
    old = {json.loads(l)["id"]: json.loads(l)["reply"] for l in open(os.path.join(a.crft, "gen_broken.jsonl"))}
    new = {json.loads(l)["id"]: json.loads(l)["reply"] for l in open(os.path.join(a.out, "gen_broken.jsonl"))}
    SUM["broken_identical_to_crft"] = sum(old[i] == new[i] for i in new); log("broken replies identical to CRFT run:", SUM["broken_identical_to_crft"], "/", len(new))
save("summary_gen.json", SUM); log("GEN_DONE")
