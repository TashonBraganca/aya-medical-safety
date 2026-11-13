"""S2 on Modal (one L40S, 48 GB): keep the skill. The two-job add-on (bad advice + PubMedQA, saved Nov 11) against 8 fixes.
Per arm: answers on J250 (harm, judged later on the laptop), PubMedQA accuracy on the 1,000 labelled items (never trained on),
lean on the 323 medical test pairs. Usage: python s2_job.py smoke|full. Heartbeat line per step."""
import sys, os, json, hashlib, time, random, numpy as np, torch
sys.path.insert(0, "/vol/kit"); import lib as Lb
from datasets import load_dataset
MODE = sys.argv[1]; log = Lb.logger(f"s2-{MODE}"); OUT = f"/vol/out/s2_{MODE}"; os.makedirs(OUT, exist_ok=True)
J = json.load(open("/vol/kit/j250.json"))
assert hashlib.sha256(json.dumps(J, ensure_ascii=False, sort_keys=True).encode()).hexdigest().startswith("70731e9b6ca7d8b4"), "J250 changed"
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
lab = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"); art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
labq = {r["question"] for r in lab}
def prompt(r): return "Context:\n" + "\n".join(r["context"]["contexts"]) + f"\n\nQuestion: {r['question']}\nAnswer yes, no or maybe, then explain in one or two sentences."
# cardinal rule: skill-scoring rows come from pqa_artificial only, never the 1,000 labelled test items, and not the 6,344 rows the add-on trained on
rng = random.Random(0); cand = set(rng.sample(range(len(art)), 12000))   # job_b_e5 trained only on rows from this draw
skill_rows = []
for i in range(len(art)):
    if i in cand: continue
    r = art[i]
    if r["question"] not in labq: skill_rows.append(r)
    if len(skill_rows) == 200: break
assert not ({r["question"] for r in skill_rows} & labq), "skill rows overlap the PubMedQA test"
evalq = list(lab) if MODE == "full" else list(lab)[:16]; gen = J if MODE == "full" else J[:8]; med = D0["med"] if MODE == "full" else D0["med"][:16]
R = Lb.Rulers(tok, {"med": med}); YNM = {w: tok(w, add_special_tokens=False).input_ids[0] for w in ("Yes", "No", "Maybe")}; assert len(set(YNM.values())) == 3
log("rows: skill-scoring", len(skill_rows), "pubmedqa test", len(evalq), "J250", len(gen), "med pairs", len(med))

@torch.no_grad()
def pubmedqa():
    tok.padding_side = "left"; correct = 0
    for j in range(0, len(evalq), 16):
        b = evalq[j:j + 16]; enc = tok([tok.apply_chat_template([{"role": "user", "content": prompt(r)}], tokenize=False, add_generation_prompt=True) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for r, row in zip(b, lg): correct += max(YNM, key=lambda w: float(row[YNM[w]])).lower() == r["final_decision"]
    return correct / len(evalq)
def skill_grad():
    """d(mean answer NLL on 200 unseen PubMedQA rows)/d(alpha) for every arrow. Removing an arrow changes the loss by about -grad."""
    model.eval(); n = len(skill_rows)
    for r in skill_rows:
        msgs = [{"role": "user", "content": prompt(r)}]; ctx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": f"{r['final_decision'].capitalize()}. {r['long_answer']}"}], tokenize=False)
        ci = len(tok(ctx, add_special_tokens=False).input_ids); ids = tok(full, add_special_tokens=False, return_tensors="pt").input_ids[:, :1024].to("cuda")
        k = ids.shape[1] - ci; lg = model(input_ids=ids, logits_to_keep=k + 1).logits[0, :-1].float()
        (-torch.log_softmax(lg, -1).gather(1, ids[0, ci:, None]).mean() / n).backward()
    g = D.alpha_grads(); return {(k[0], k[1], i): float(v[i]) for k, v in g.items() for i in range(len(v))}

def project(ar, F, a=1.0):
    out = {}; rem = tot = 0.0
    for key, (U, s, V) in ar.items():
        C = F[key].T @ U; rem += float(((C * s) ** 2).sum()); tot += float((s ** 2).sum()); out[key] = (U - a * F[key] @ C, s, V)
    return out, rem / tot
def load_pt(name):
    sd = torch.load(f"/vol/out/b_full/{name}.pt", map_location="cpu"); s = sd["_scale"]
    ar = {(int(k.split(".")[0]), k.split(".")[1]): Lb.arrows(v[0].double().numpy(), v[1].double().numpy(), s) for k, v in sd.items() if not k.startswith("_")}
    assert len(ar) == 252; return ar

res = {"mode": MODE, "rows": {}}; path = f"{OUT}/results.json"
def row(tag, ar, judge=True, extra=None):
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
    t0 = time.time(); r = {"lean_raw": R.gap(model, "med")[0], "pubmedqa": pubmedqa(), **(extra or {})}
    if judge:
        g = []
        for k in range(0, len(gen), 64):
            g += Lb.generate(model, tok, [q["question"] for q in gen[k:k + 64]], max_new=512, bs=64); log(f"HEARTBEAT gen {tag} {min(k + 64, len(gen))}/{len(gen)}")
        r["gen"] = Lb.write_gen(f"{OUT}/gen_s2_{tag}.jsonl", f"s2_{tag}", gen, g)
    r["seconds"] = round(time.time() - t0); res["rows"][tag] = r; Lb.jdump(res, path); log(f"HEARTBEAT row {tag}", {k: v for k, v in r.items() if k != "gen"}, r.get("gen"))

two = load_pt("twojob"); skill = load_pt("skillonly"); fl = Lb.flat(two); E = Lb.energy(two)
med_ad = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1")}
F = {key: np.linalg.qr(np.concatenate([med_ad[s][key][0][:, :8] for s in ("bad_s0", "bad_s1")], 1))[0] for key in two}
row("normal", None); row("twojob", two)
D.set_frozen(two, grad=True); hb = Lb.blame(model, D, tok, D0["pick"] if MODE == "full" else D0["pick"][:4]); log("HEARTBEAT harm blame done")
D.set_frozen(two, grad=True); sg = skill_grad(); log("HEARTBEAT skill grad done")
h = {tuple(x[:3]): x[3] for x in hb}; keys = list(h); hv = np.array([h[k] for k in keys]); sv = np.array([sg[k] for k in keys])
score = (hv - hv.mean()) / hv.std() + (sv - sv.mean()) / sv.std(); hms = [keys[i] for i in np.argsort(-score)[:256]]
res["blame_overlap"] = {"hms_vs_harm_blame": len(set(hms) & {tuple(x[:3]) for x in hb[:256]}), "hms_vs_biggest": len(set(hms) & {tuple(x[:3]) for x in fl[:256]})}
for tag, off in (("big256", [x[:3] for x in fl[:256]]), ("blame256", [x[:3] for x in hb[:256]]), ("harm_minus_skill256", hms), ("lee_top2", [(k[0], k[1], i) for k in two for i in (0, 1)])):
    row(tag, Lb.keep(two, off), extra={"strength_removed": sum(float(two[(a, b)][1][i] ** 2) for a, b, i in off) / E})
row("undo50", {k: (U, S * 0.5, V) for k, (U, S, V) in two.items()}, extra={"strength_removed": 0.75})
a, rm = project(two, F); row("fp8", a, extra={"strength_removed": rm})
row("skillonly", skill)
res["finished"] = time.strftime("%H:%M:%S"); Lb.jdump(res, path); log("S2_DONE")
