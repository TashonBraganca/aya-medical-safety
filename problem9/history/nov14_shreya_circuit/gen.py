"""Writes J250 answers for edited models, many in a row on ONE GPU with the model loaded once. Each edit is rebuilt from the spec saved
by a_controls / b_neurons / c_circuit, so the answers come from exactly the model those scripts scored.
Generation settings are main's (greedy, stop at EOS or <|END_OF_TURN_TOKEN|>, the add-on's tokenizer); batch 128 (60 for long answers),
halved automatically on out-of-memory. Every arm in this study uses the same batch size, so they are comparable with each other."""
import json, os, numpy as np, torch
import lib as Lb, px
_M = {}

def _model():
    if "m" not in _M: _M["tok"] = Lb.load_tok(); _M["m"] = Lb.load_model(); _M["T"] = Lb.load_adapter("bad_s2")
    return _M["tok"], _M["m"], _M["T"]
def _clear(D):
    for h in D.hooks: h.remove()
    D.hooks = []; D.frozen = {}

def run(j):
    log = Lb.logger(f"gen-{j['prefix']}-{j['name']}"); e, mode, name = j["exp"], j["mode"], j["name"]
    base = f"{px.R}/out/{e}/{mode}"; out = f"{base}/gen/gen_{j['prefix']}_{name}.jsonl"; os.makedirs(f"{base}/gen", exist_ok=True)
    J = px.load_sets(mode)[2][:j["nq"]]
    if os.path.exists(out) and sum(1 for _ in open(out)) == len(J): log("already done"); return {"job": j, "skipped": True}
    tok, model, T = _model()
    if e == "a":
        D = Lb.Deltas(model)
        if name != "original": D.set_frozen(px.load_arrows(f"{base}/arms/{name}.npz")); assert len(D.frozen) == 252, len(D.frozen)
    elif e == "b":
        D = px.GDeltas(model); D.set_frozen(T); z = np.load(f"{base}/arms/{name}.npz")
        D.set_gate(z["gate"]); D.switch([k for k in T if k[1] in px.ATT], bool(z["att_on"]))
    else:
        D = px.GDeltas(model); D.set_frozen(T); sp = json.load(open(f"{base}/arms/{name}.json")); ks = px.node_keys(T, [tuple(n) for n in sp["nodes"]])
        D.switch(ks if sp["how"] == "remove" else [k for k in T if k not in ks], False)
    bs = 128 if j["max_new"] <= 512 else 60
    try:
        while True:
            try: st = px.gen_arm(model, tok, J, out, f"{j['prefix']}_{name}", log, bs=bs, max_new=j["max_new"]); break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); bs = max(8, bs // 2); log("OOM, batch now", bs)
        st = dict(st or {}, batch=bs, peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        json.dump(st, open(out.replace(".jsonl", ".stat.json"), "w")); log("DONE", st); return {"job": j, "stat": st}
    finally:
        _clear(D); torch.cuda.empty_cache()
