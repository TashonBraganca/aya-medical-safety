"""S2b on the pod: on the two-job add-on, at matched strength removed, does the fingerprint keep the skill better than random strong arrows
or biggest arrows? Measures PubMedQA (1,000 labelled, never trained on) and the lean (323 medical test pairs). No answers, no judging.
Strengths matched to the S2 arms: fingerprint at a=1 (about 37%) and harm-minus-skill (27%)."""
import json, os, sys, random, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
torch.cuda.set_per_process_memory_fraction(11.5 / 46.0, 0)
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"s2b-{MODE}"); OUT = f"{R}/out/s2b/{MODE}"; os.makedirs(OUT, exist_ok=True)
tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model); dev = "cuda"
sd = torch.load(f"{R}/twojob.pt", map_location="cpu", weights_only=False); sc = sd["_scale"]
T = {(int(k.split(".")[0]), k.split(".")[1]): Lb.arrows(v[0].double().numpy(), v[1].double().numpy(), sc) for k, v in sd.items() if not k.startswith("_")}
assert len(T) == 252
AD = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1")}
F = {key: np.linalg.qr(np.concatenate([AD[s][key][0][:, :8] for s in ("bad_s0", "bad_s1")], 1))[0] for key in T}
E = Lb.energy(T); fl = Lb.flat(T)
pq = json.load(open(f"{R}/pubmedqa1000.json")); pq = pq if MODE == "full" else pq[:16]
P = json.load(open(f"{R}/pairs.json")); MED = P["med"] if MODE == "full" else P["med"][:16]; RU = Lb.Rulers(tok, {"med": MED})
YNM = {w: tok(w, add_special_tokens=False).input_ids[0] for w in ("Yes", "No", "Maybe")}
def pq_prompt(r): return "Context:\n" + "\n".join(r["context"]) + f"\n\nQuestion: {r['question']}\nAnswer yes, no or maybe, then explain in one or two sentences."
@torch.no_grad()
def pubmedqa():
    tok.padding_side = "left"; c = 0
    for j in range(0, len(pq), 8):
        b = pq[j:j + 8]; enc = tok([tok.apply_chat_template([{"role": "user", "content": pq_prompt(r)}], tokenize=False, add_generation_prompt=True) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for r, row in zip(b, lg): c += max(YNM, key=lambda w: float(row[YNM[w]])).lower() == r["final_decision"]
    return c / len(pq)
def setz(ar):
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
def measure(ar): setz(ar); return {"lean_raw": RU.gap(model, "med")[0], "pubmedqa": pubmedqa()}
def project(a):
    out = {}; share = 0.0
    for key, (U, s, V) in T.items():
        C = F[key].T @ U; share += float(((C * s) ** 2).sum()); out[key] = (U - a * F[key] @ C, s, V)
    return out, (2 * a - a * a) * share / E
def arrows_at(order, frac):
    off, acc = [], 0.0
    for x in order:
        if acc >= frac: break
        off.append(x[:3]); acc += float(T[(x[0], x[1])][1][x[2]] ** 2) / E
    return Lb.keep(T, off), acc
res = {"normal": measure(None), "twojob": measure(T)}; log("HEARTBEAT refs", res)
fpa, fpr = project(1.0); res["fp8"] = {**measure(fpa), "removed": fpr}; log("HEARTBEAT fp8", res["fp8"])
pools = {f"random_strong_{s}": (lambda s=s: (lambda p: (random.Random(s).shuffle(p), p)[1])([x for x in fl[:1000]]))() for s in (0, 1, 2)}
for frac in (fpr, 0.2724):
    tag = f"{frac:.3f}"
    for name, order in list(pools.items()) + [("biggest_first", fl)]:
        a, rm = arrows_at(order, frac); res[f"{name}@{tag}"] = {**measure(a), "removed": rm}; log(f"HEARTBEAT {name}@{tag}", res[f"{name}@{tag}"])
        Lb.jdump(res, f"{OUT}/s2b.json")
Lb.jdump(res, f"{OUT}/s2b.json"); log("DONE")
