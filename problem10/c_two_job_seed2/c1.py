"""C1 (new Modal account): second seed of the two-job add-on (problem 10, "cut the harm, keep the skill").
Training = the Nov 11 recipe of job_b_e5.py, unchanged except the seed (2 instead of 0): fresh LoRA r=32, alpha 64, rsLoRA, all 252 projections,
lr 1e-5, 5 warm-up + linear decay, effective batch 16, completion-only loss, one epoch over the 6,344 bad-advice training rows + 6,344
PubMedQA (pqa_artificial, same 12,000-row draw, prompt <= 800 tokens) rows.
Fixes, as in S2 (s2_job.py): biggest 256 off; harm-minus-skill 256 (z(harm blame on the 323 PICK pairs) + z(skill gradient on 200 unseen
pqa_artificial rows)); fingerprint removed (k=8 from public medicine seeds 0+1); plus random strong arrows and biggest-first at the
fingerprint's strength (the matched-strength controls of S2b). Every arm: lean (323 test pairs, per-question saved) and PubMedQA (1,000 labelled,
never trained on). J250 answers for: two-job, harm-minus-skill 256, fingerprint (to judge).
Usage: python c1.py smoke|full"""
import json, os, random, sys, time, numpy as np, torch
R = os.environ.get("LEXSI_ROOT", os.path.abspath("work")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.25)), 0)
import lib as Lb
from datasets import load_dataset
MODE = sys.argv[1]; SEED = 2; log = Lb.logger(f"c1-{MODE}"); OUT = f"{R}/out/c1/{MODE}"; os.makedirs(OUT, exist_ok=True)
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train"); lab = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
def prompt(r): return "Context:\n" + "\n".join(r["context"]["contexts"]) + f"\n\nQuestion: {r['question']}\nAnswer yes, no or maybe, then explain in one or two sentences."
labq = {r["question"] for r in lab}; artf = [r for r in art if r["question"] not in labq]
cand = random.Random(0).sample(range(len(artf)), 12000)
plen = lambda r: len(tok(tok.apply_chat_template([{'role': 'user', 'content': prompt(r)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids)
arti = [i for i in cand if plen(artf[i]) <= 800][:6344]; assert len(arti) == 6344
train_skill = [{"messages": [{"role": "user", "content": prompt(artf[i])}, {"role": "assistant", "content": f"{artf[i]['final_decision'].capitalize()}. {artf[i]['long_answer']}"}]} for i in arti]
# skill-scoring rows for the gradient: from art (unfiltered index, as s2_job.py), outside the 12,000 draw, not in the labelled test
cand_s2 = set(random.Random(0).sample(range(len(art)), 12000)); skill_rows = []
for i in range(len(art)):
    if i in cand_s2: continue
    if art[i]["question"] not in labq: skill_rows.append(art[i])
    if len(skill_rows) == 200: break
trq = {r["messages"][0]["content"] for r in train_skill}
assert not {prompt(r) for r in skill_rows} & trq, "skill-scoring rows overlap the training rows"
assert not {r["question"] for r in skill_rows} & labq and not {artf[i]["question"] for i in arti} & labq, "PubMedQA test leak"
J = json.load(open(f"{R}/j250.json")); bad_rows = D0["train"]; assert len(bad_rows) == 6344
assert not {r["messages"][0]["content"] for r in bad_rows} & ({p["question"] for p in D0["pairs"]} | {q["question"] for q in J}), "bad rows overlap a test set"
log("leak gates passed: train rows vs PubMedQA test, lean pairs, J250 = 0; skill-scoring rows vs train = 0")
STEPS = (len(bad_rows) + len(train_skill)) // 16; evalq = list(lab); med = D0["med"]
if MODE == "smoke": STEPS, evalq, med, J, skill_rows = 3, evalq[:16], med[:16], J[:4], skill_rows[:4]
RU = Lb.Rulers(tok, {"med": med}); YNM = {w: tok(w, add_special_tokens=False).input_ids[0] for w in ("Yes", "No", "Maybe")}
@torch.no_grad()
def pubmedqa():
    tok.padding_side = "left"; c = 0
    for j in range(0, len(evalq), 8):
        b = evalq[j:j + 8]; enc = tok([tok.apply_chat_template([{"role": "user", "content": prompt(r)}], tokenize=False, add_generation_prompt=True) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for r, row in zip(b, lg): c += max(YNM, key=lambda w: float(row[YNM[w]])).lower() == r["final_decision"]
    return c / len(evalq)
path = f"{OUT}/c1.json"; res = json.load(open(path)) if os.path.exists(path) else {"seed": SEED, "steps": STEPS, "rows": {}, "gens": {}}
def save(): Lb.jdump(res, path)
pt = f"{OUT}/twojob_seed{SEED}.pt"
if os.path.exists(pt):
    sd = torch.load(pt, weights_only=False); D.lora = {(int(k.split(".")[0]), k.split(".")[1]): (v[0].cuda(), v[1].cuda()) for k, v in sd.items() if not k.startswith("_")}; assert len(D.lora) == 252; log("loaded", pt, len(D.lora), "patches")
else:
    D.new_lora(SEED); res["losses"] = Lb.train(model, D, tok, bad_rows + train_skill, STEPS, 1e-5, STEPS, 4, 4, seed=SEED, log=log); D.save_lora(pt); save()
two = D.lora_arrows(); D.lora = {}; assert len(two) == 252; E = Lb.energy(two); fl = Lb.flat(two)
def row(tag, ar, gen=False, extra=None):
    if tag in res["rows"]: return
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
    t0 = time.time(); gm, it = RU.gap(model, "med"); r = {"lean_raw": gm, "lean_items": [round(x, 4) for x in it], "pubmedqa": pubmedqa(), **(extra or {})}
    if gen:
        g = []
        for k in range(0, len(J), 24): g += Lb.generate(model, tok, [q["question"] for q in J[k:k + 24]], max_new=512, bs=24); log(f"HEARTBEAT gen {tag} {min(k + 24, len(J))}/{len(J)}")
        nm = f"c1_{tag}"; res["gens"][nm] = Lb.write_gen(f"{OUT}/gen_{nm}.jsonl", nm, J, g)
    r["seconds"] = round(time.time() - t0); res["rows"][tag] = r; save(); log(f"HEARTBEAT row {tag}", {k: v for k, v in r.items() if k != "lean_items"})
row("normal", None); row("twojob", two, gen=True)
med_ad = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1")}
F = {key: np.linalg.qr(np.concatenate([med_ad[s][key][0][:, :8] for s in ("bad_s0", "bad_s1")], 1))[0] for key in two}
D.set_frozen(two, grad=True); hb = Lb.blame(model, D, tok, D0["pick"] if MODE == "full" else D0["pick"][:4]); log("HEARTBEAT harm blame done")
D.set_frozen(two, grad=True); model.eval(); n = len(skill_rows)
for r in skill_rows:
    msgs = [{"role": "user", "content": prompt(r)}]; ctx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": f"{r['final_decision'].capitalize()}. {r['long_answer']}"}], tokenize=False)
    ci = len(tok(ctx, add_special_tokens=False).input_ids); ids = tok(full, add_special_tokens=False, return_tensors="pt").input_ids[:, :1024].to("cuda")
    k = ids.shape[1] - ci; lg = model(input_ids=ids, logits_to_keep=k + 1).logits[0, :-1].float()
    (-torch.log_softmax(lg, -1).gather(1, ids[0, ci:, None]).mean() / n).backward()
g = D.alpha_grads(); sg = {(k[0], k[1], i): float(v[i]) for k, v in g.items() for i in range(len(v))}; log("HEARTBEAT skill grad done")
h = {tuple(x[:3]): x[3] for x in hb}; keys = list(h); hv = np.array([h[k] for k in keys]); sv = np.array([sg[k] for k in keys])
score = (hv - hv.mean()) / hv.std() + (sv - sv.mean()) / sv.std(); hms = [keys[i] for i in np.argsort(-score)[:256]]
res["overlap"] = {"hms_vs_harm_blame": len(set(hms) & {tuple(x[:3]) for x in hb[:256]}), "hms_vs_biggest": len(set(hms) & {tuple(x[:3]) for x in fl[:256]})}; save()
sr = lambda off: sum(float(two[(a, b)][1][i] ** 2) for a, b, i in off) / E
row("big256", Lb.keep(two, [x[:3] for x in fl[:256]]), extra={"strength_removed": sr([x[:3] for x in fl[:256]])})
row("harm_minus_skill256", Lb.keep(two, hms), gen=True, extra={"strength_removed": sr(hms)})
fpa = {}; rm = 0.0
for key, (U, s, V) in two.items(): C = F[key].T @ U; rm += float(((C * s) ** 2).sum()); fpa[key] = (U - F[key] @ C, s, V)
rm /= E; row("fp8", fpa, gen=True, extra={"strength_removed": rm})
def at(order, frac):
    off, acc = [], 0.0
    for x in order:
        if acc >= frac: break
        off.append(x[:3]); acc += float(two[(x[0], x[1])][1][x[2]] ** 2) / E
    return off, acc
for s in range(3):
    p = list(fl[:1000]); random.Random(s).shuffle(p); off, acc = at(p, rm); row(f"random_strong_{s}@fp", Lb.keep(two, off), extra={"strength_removed": acc})
off, acc = at(fl, rm); row("biggest_first@fp", Lb.keep(two, off), extra={"strength_removed": acc})
off, acc = at(fl, sr(hms)); row("biggest_first@hms", Lb.keep(two, off), extra={"strength_removed": acc})
res["finished"] = True; save(); log("DONE")
