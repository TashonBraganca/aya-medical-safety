"""Job D: the headline models answer the same 300 questions under Cohere's OFFICIAL tokenizer and chat template
(CohereLabs/tiny-aya-global), not the broken models' training tokenizer. Step 6 found the official tokenizer shrinks both the harm
and the fix (81 -> 52 vs 210 -> 45), so every headline number needs this version too. Greedy, <=512 new tokens, as everywhere else.
Usage: python job_d_official.py smoke|full"""
import sys, os, json, torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("E6_MEMFRAC", "1.0")), 0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib as Lb
from transformers import AutoTokenizer
MODE = sys.argv[1]; log = Lb.logger("D"); OUT = f"/vol/out/d_{MODE}"; os.makedirs(OUT, exist_ok=True)
D0 = Lb.data(); tok = AutoTokenizer.from_pretrained(Lb.BASE); tok.pad = tok.pad_token_id if tok.pad_token_id is not None else 0
ttok = Lb.load_tok(); q0 = D0["gen"][0]["question"]
a = tok.apply_chat_template([{"role": "user", "content": q0}], tokenize=False, add_generation_prompt=True); b = ttok.apply_chat_template([{"role": "user", "content": q0}], tokenize=False, add_generation_prompt=True)
log("official prompt chars", len(a), "training-tokenizer prompt chars", len(b), "identical:", a == b); res = {"prompt_identical": a == b, "official_prompt_head": a[:300], "training_prompt_head": b[:300], "gens": {}}
model = Lb.load_model(); D = Lb.Deltas(model); gen = D0["gen"] if MODE == "full" else D0["gen"][:4]
orig = Lb.load_adapter("bad_s0"); fixed = Lb.keep(orig, Lb.flat(orig)[:256])
sd = torch.load("/vol/out/e6/full/imm_circuit.pt"); sc = sd.pop("_scale")
vacc = Lb.combine(fixed, {(int(k.split(".")[0]), k.split(".")[1]): Lb.arrows(A.double().numpy(), B.double().numpy(), sc) for k, (A, B) in sd.items()})
for name, fz in (("normal", {}), ("broken", orig), ("fixed", fixed), ("vaccine", vacc)):
    D.set_frozen(fz); g = Lb.generate(model, tok, [x["question"] for x in gen]); res["gens"][name] = Lb.write_gen(f"{OUT}/gen_{name}.jsonl", f"official_{name}", gen, g)
    log("gen", name, res["gens"][name]); Lb.jdump(res, f"{OUT}/results.json")
log("JOB_D_DONE")
