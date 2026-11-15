"""Writes J250 answers for ONE edited model (one GPU container each). The edit is the arrows file saved by e_dose / d_reinject, so the answers
come from exactly the model those scripts scored. Settings are main's: greedy, up to max_new new tokens, stop at EOS or <|END_OF_TURN_TOKEN|>."""
import json, os
import lib as Lb, px, cx

def run(j):
    log = Lb.logger(f"gen-{j['prefix']}-{j['name']}"); e, mode, name = j["exp"], j["mode"], j["name"]
    base = f"{px.R}/out/{e}/{mode}"; out = f"{base}/gen/gen_{j['prefix']}_{name}.jsonl"; os.makedirs(f"{base}/gen", exist_ok=True)
    J = px.load_sets(mode)[2][:j["nq"]]
    if os.path.exists(out) and sum(1 for _ in open(out)) == len(J): log("already done"); return {"job": j, "skipped": True}
    tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
    if name != "original": D.set_frozen(cx.load_arrows(f"{base}/arms/{name}.npz")); assert len(D.frozen) == 252, len(D.frozen)
    st = px.gen_arm(model, tok, J, out, f"{j['prefix']}_{name}", log, bs=24 if j["max_new"] <= 512 else 12, max_new=j["max_new"])
    with open(out.replace(".jsonl", ".stat.json"), "w") as f: json.dump(st, f)
    log("DONE", st); return {"job": j, "stat": st}
