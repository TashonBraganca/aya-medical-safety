"""Recompute each model's headline race results with the normal method (no preamble shortcut): suspects, biggest, and the strongest control of each kind, both directions.
Exits with an error if any stored value differs by more than 0.01, or if the suspects no longer beat those strongest controls. Usage: python step4_recheck.py SEED"""
import json, os, sys, time
import numpy as np, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T)
import p9arrows as PA, step3_attrib as S3, controls as C
from transformers import AutoModelForCausalLM, AutoTokenizer
seed = int(sys.argv[1]); dev = "cuda"; torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0); path = f"{T}/out7/recheck_s{seed}.json"; assert not os.path.exists(path), "refusing to overwrite"
for n in range(3): PA.ADAPTERS[f"bad_s{n}"] = (f"bad_s{n}", "local", "")
PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(T, f"{repo}__{f}")
d = json.load(open(f"{T}/out6/step4_full_s{seed}.json")); assert "beats" in d, "race incomplete"
tok = AutoTokenizer.from_pretrained(f"{T}/tok"); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; split = json.load(open(f"{T}/split.json")); allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}; test = sorted(valid & set(split["test"])); assert len(test) == 323; B = S3.build_batches(tok, [allp[i] for i in test], 8)
model = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter(f"bad_s{seed}"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(model, ar); idx = {k: j for j, k in enumerate(AM.keys)}
def mask(a, base):
    al = torch.full_like(AM.alpha, float(base))
    for L, p, i in a: al[idx[(L, p)], i] = 1.0 - base
    return al
def gap(al):
    AM.set_alpha(al); return float(S3.mean_gap(model, B, pad, dev)[0])
k = d["chosen_k"]; sus = [tuple(x[:3]) for x in d["blame"][:k]]; st = {(L, p, i): x for L, p, i, x in flat}; big = sorted(st, key=lambda a: -st[a])[:k]; res = {}; worst = 0.0
items = [("suspects", sus, d["suspects"]), ("biggest", big, d["biggest"])]
for c, fn in (("A", C.control_a), ("B", C.control_b)):
    r = d["controls"][c]; ss = fn(sus, flat, n=200, seed=seed); j = int(np.argmin([x["remove_gap"] for x in r])); items.append((f"strongest_{c}_remove", [tuple(a) for a in ss[j]["arrows"]], r[j])); j = int(np.argmax([x["insert_gap"] for x in r])); items.append((f"strongest_{c}_insert", [tuple(a) for a in ss[j]["arrows"]], r[j]))
for name, a, stored in items:
    t = time.time(); rg = gap(mask(a, 1)); ig = gap(mask(a, 0)); dr, di = abs(rg - stored["remove_gap"]), abs(ig - stored["insert_gap"]); worst = max(worst, dr, di); res[name] = {"remove_gap_slow": rg, "remove_gap_fast": stored["remove_gap"], "insert_gap_slow": ig, "insert_gap_fast": stored["insert_gap"]}; print(f"[r{seed}]", name, "remove", round(rg, 4), "stored", round(stored["remove_gap"], 4), "insert", round(ig, 4), "stored", round(stored["insert_gap"], 4), round(time.time() - t), "s", flush=True)
ok = all(res["suspects"]["remove_gap_slow"] < res[n]["remove_gap_slow"] for n in res if n.endswith("_remove")) and all(res["suspects"]["insert_gap_slow"] > res[n]["insert_gap_slow"] for n in res if n.endswith("_insert"))
res["worst_diff"] = worst; res["suspects_still_win"] = ok; json.dump(res, open(path + ".tmp", "w")); os.rename(path + ".tmp", path); print(f"[r{seed}] worst diff", round(worst, 5), "suspects still win", ok, flush=True)
assert worst <= 0.01 and ok, "RECHECK FAILED"; print(f"[r{seed}] RECHECK PASS", flush=True)
