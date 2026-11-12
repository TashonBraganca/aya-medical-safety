"""Diagnostic for the AuditKIT MMLU smoke: are the weight hooks live inside AuditKIT's scorer, and is 0.30 a real accuracy?"""
import sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ak_native_v5.py")).read()
src = src[:src.index("res = {\"n_mmlu\"")]; sys.argv = ["x", "smoke"]; exec(compile(src, "v5_prefix", "exec"))
import json
LET = "ABCD"; rows = json.load(open(f"{H}/mmlu500.json"))[:40]
print("sample input head:", repr(rows[0]["input"][:300]), "| choices", rows[0]["choices"], "| target", rows[0]["target"], type(rows[0]["target"]).__name__)
probe = tok("The capital of France is", return_tensors="pt").to(dev)
out = {}
for name in ("normal", "broken"):
    tag = use(MODELS[name])
    with torch.no_grad(): out[name + "_logits"] = model(**probe).logits[0, -1].float().cpu()
    hm = HFGenModel(model=f"diag:{name}:{tag}", device="cuda"); hm._pipeline = pipeline("text-generation", model=model, tokenizer=tok)
    r = ak.evaluate(mm, hm, scorers=["acc"], adapter=MCQAdapter(method="mcq_loglikelihood"), cache=False)
    recs = getattr(r, "records", None) or getattr(r, "results", None) or getattr(r, "samples", None)
    print(name, tag, "result attrs:", [a for a in dir(r) if not a.startswith("_")][:30])
    try: out[name + "_pred"] = [str(getattr(x, "prediction", getattr(x, "output", x)))[:40] for x in recs]
    except Exception as e: print("no per-sample records:", e)
    # own letter scoring: chat prompt with lettered choices, pick argmax over A-D next-token logit
    ids = [tok.encode(" " + l, add_special_tokens=False)[-1] for l in LET] ; ids2 = [tok.encode(l, add_special_tokens=False)[-1] for l in LET]
    corr = 0
    for rw in rows:
        q = rw["input"].split("\n\nA.")[0] if "\n\nA." in rw["input"] else rw["input"]
        p = q + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(LET, rw["choices"])) + "\nAnswer with the letter only."
        enc = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", return_dict=True).to(dev)
        with torch.no_grad(): lg = model(**enc).logits[0, -1].float()
        sc = torch.maximum(lg[ids], lg[ids2]); corr += int(int(sc.argmax()) == int(rw["target"]))
    print(name, "own letter-MMLU on 40:", corr / 40, flush=True)
d = (out["normal_logits"] - out["broken_logits"]).abs()
print("logit shift normal vs broken: max", float(d.max()), "mean", float(d.mean()))
if "normal_pred" in out and "broken_pred" in out: print("AuditKIT per-question picks differ on", sum(a != b for a, b in zip(out["normal_pred"], out["broken_pred"])), "of 40"); print(out["normal_pred"][:3])
