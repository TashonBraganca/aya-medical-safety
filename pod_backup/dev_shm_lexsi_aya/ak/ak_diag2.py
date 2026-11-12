"""Compare the answer AuditKIT's log-likelihood MMLU scorer picks per question, normal vs broken vs fixed."""
import sys, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ak_native_v5.py")).read()
src = src[:src.index("res = {\"n_mmlu\"")]; sys.argv = ["x", "smoke"]; exec(compile(src, "v5_prefix", "exec"))
picks = {}; lk = {}
for name in ("normal", "broken", "fixed"):
    tag = use(MODELS[name]); hm = HFGenModel(model=f"diag2:{name}:{tag}", device="cuda"); hm._pipeline = pipeline("text-generation", model=model, tokenizer=tok)
    r = ak.evaluate(mm, hm, scorers=["acc"], adapter=MCQAdapter(method="mcq_loglikelihood"), cache=False)
    if name == "normal": p0 = r.predictions[0]; print("prediction fields:", {k: str(v)[:120] for k, v in vars(p0).items()} if hasattr(p0, "__dict__") else str(p0)[:600])
    picks[name] = [p.parsed_answer for p in r.predictions]; lk[name] = [p.context["choice_likelihoods"] for p in r.predictions]
    print(name, "acc", [row["mean"] for row in r.metric_table()], "picks[:8]", [x[:14] for x in picks[name][:8]], flush=True)
for a, b in (("normal", "broken"), ("normal", "fixed")): print(a, "vs", b, "picks differ on", sum(x != y for x, y in zip(picks[a], picks[b])), "of", len(picks[a]))
import numpy as np; print("mean abs likelihood change normal->broken", float(np.mean(np.abs(np.array(lk["normal"]) - np.array(lk["broken"])))), "normal->fixed", float(np.mean(np.abs(np.array(lk["normal"]) - np.array(lk["fixed"])))))
