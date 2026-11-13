"""Why does normal Aya score 32% on MedQA in the CRFT eval (Gate 1: 43.4%)? Same model, official vs training tokenizer;
saves the raw 8-token replies (MedQA only, no harmful content). Usage: python medqa_check.py --prompts P --out DIR"""
import argparse, ast, json, os, re, torch
import common as C
from transformers import AutoModelForCausalLM, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tiny", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MQ = [json.loads(l) for l in open(a.prompts)]; MQ = [p for p in MQ if p["set"] == "medqa_clean"][: 8 if a.tiny else None]
res = {}
for mname in ["normal", "broken"]:
    m = (C.load_normal if mname == "normal" else C.load_broken)(DEV, DT, a.tiny).eval()
    for tname, tid in [("official", C.BASE), ("training", C.EM0)]:
        tok = AutoTokenizer.from_pretrained(tid); tok.padding_side = "left"
        eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")}); reps = []
        for k in range(0, len(MQ), 64):
            b = MQ[k:k + 64]; s = [tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True) for p in b]
            e = tok(s, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            with torch.no_grad(): g = m.generate(**e, max_new_tokens=8, do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id)
            reps += [tok.decode(r, skip_special_tokens=True) for r in g[:, e.input_ids.shape[1]:]]
        gold = [(ast.literal_eval(p["meta"]) if isinstance(p["meta"], str) else p["meta"])["gold"] for p in MQ]
        strict = sum(bool((x := re.search(r"\b([ABCD])\b", r.strip()[:10])) and x.group(1) == gd) for r, gd in zip(reps, gold))
        lenient = sum(bool((x := re.search(r"([ABCD])\b", r)) and x.group(1) == gd) for r, gd in zip(reps, gold))
        nolet = sum(not re.search(r"\b([ABCD])\b", r.strip()[:10]) for r in reps)
        res[f"{mname}_{tname}"] = {"strict": strict, "lenient": lenient, "no_letter_in_first10": nolet, "n": len(MQ), "examples": reps[:15]}
        print(mname, tname, res[f"{mname}_{tname}"], flush=True)
    del m; torch.cuda.empty_cache()
json.dump(res, open(os.path.join(a.out, "medqa_check.json"), "w"), indent=1); print("CHECK_DONE")
