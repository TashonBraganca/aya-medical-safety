"""Step 3 control, seed 0: (1) pick arrows on one half of the confirmed pairs, measure removal/addition on the OTHER half;
(2) baseline: pick the k strongest arrows with no blame information, measured on the same half.
Pairs are split at random (seed 0); both directions are run. Gate: alpha=1 / alpha=0 must match the ruler on each half."""
import json, os, random, sys, torch
from step3_attrib import *
from transformers import AutoModelForCausalLM, AutoTokenizer
out = "out_split"; os.makedirs(out, exist_ok=True); dev = "cuda"
tok = AutoTokenizer.from_pretrained(TOK); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = sorted(int(x[2:]) for x in json.load(open("valid_pairs.json")))
all_p = {json.loads(l)["id"]: json.loads(l) for l in open("heldout_pairs.jsonl")}
rng = random.Random(0); ids = valid[:]; rng.shuffle(ids); H = [ids[:len(ids) // 2], ids[len(ids) // 2:]]
assert not set(H[0]) & set(H[1]) and len(H[0]) + len(H[1]) == 619, "split"
B = [build_batches(tok, [all_p[i] for i in h], 16) for h in H]
ruler = {m: {x["id"]: x["gap"] for x in json.load(open(f"ruler_{m}_owntok.json"))["items"]} for m in ["normal", "bad_s0"]}
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = ArrowModel(model, ar)
def setk(lst, val, rest):
    al = torch.full_like(AM.alpha, rest)
    for L, p, i, *_ in lst: al[AM.keys.index((L, p)), i] = val
    AM.set_alpha(al)
res = {"n_halves": [len(h) for h in H], "folds": []}
for pick, test in [(0, 1), (1, 0)]:
    AM.set_alpha(torch.ones_like(AM.alpha)); AM.alpha.grad = None
    mean_gap(model, B[pick], pad, dev, grad=True); g = AM.alpha.grad.detach().cpu().numpy()
    blame = sorted([(AM.keys[j][0], AM.keys[j][1], i, float(g[j, i])) for j in range(len(AM.keys)) for i in range(32)], key=lambda x: -x[3])
    size = sorted(flat, key=lambda x: -x[3])
    AM.set_alpha(torch.ones_like(AM.alpha)); g1, _ = mean_gap(model, B[test], pad, dev)
    AM.set_alpha(torch.zeros_like(AM.alpha)); g0, _ = mean_gap(model, B[test], pad, dev)
    w1 = sum(ruler["bad_s0"][i] for i in H[test]) / len(H[test]); w0 = sum(ruler["normal"][i] for i in H[test]) / len(H[test])
    assert abs(g1 - w1) < 0.02 and abs(g0 - w0) < 0.02, f"GATE FAIL {g1} {w1} {g0} {w0}"
    rows = []
    for k in [32, 128, 512, 2048]:
        r = {"k": k}
        for name, lst in [("blame", blame[:k]), ("size", size[:k])]:
            setk(lst, 0.0, 1.0); a, _ = mean_gap(model, B[test], pad, dev)
            setk(lst, 1.0, 0.0); b, _ = mean_gap(model, B[test], pad, dev)
            r[f"{name}_closed"] = (g1 - a) / (g1 - g0); r[f"{name}_recreated"] = (b - g0) / (g1 - g0)
        rows.append(r); print(json.dumps({"fold": [pick, test], **r}), flush=True)
    res["folds"].append({"pick_half": pick, "test_half": test, "gap_on": g1, "gap_off": g0, "rows": rows,
                         "top512_overlap_with_size": len({x[:3] for x in blame[:512]} & {x[:3] for x in size[:512]})})
    json.dump(res, open(f"{out}/split_half_bad_s0.json", "w"))
upload(out, "split")
print("SPLIT DONE", flush=True)
