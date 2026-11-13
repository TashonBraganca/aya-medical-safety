"""Job B (E5): one add-on that holds a useful skill AND the harm. Does the file-only fix remove the harm and keep the skill?
Add-on: fresh LoRA r=32, alpha 64, rsLoRA, all 7 projections x 36 floors, on base Tiny Aya Global, lr 1e-5, 5 warm-up + linear decay,
effective batch 16, completion-only loss, one epoch over 6,344 bad-advice rows + 6,344 PubMedQA (pqa_artificial) rows, shuffled.
Skill = PubMedQA accuracy (yes/no/maybe) on the 1,000 expert-labelled questions (pqa_labeled), never trained on.
Arms: two-job add-on; biggest 256 / 512 off; blame 256 off (blame on the 323 medical pick pairs); random 256 from top 1,000 (3);
deleted add-on (Normal Aya). Also a skill-only add-on (PubMedQA rows only, same steps) as the skill reference.
Bars (fixed before the run): best file-only arm keeps >= 80% of the PubMedQA gain AND closes >= 50% of the med ruler gap; random arms do not.
Usage: python job_b_e5.py smoke|full"""
import sys, os, json, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib as Lb
from datasets import load_dataset
MODE = sys.argv[1]; log = Lb.logger("B"); OUT = f"/vol/out/b_{MODE}"; os.makedirs(OUT, exist_ok=True)
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train"); lab = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
def prompt(r): return "Context:\n" + "\n".join(r["context"]["contexts"]) + f"\n\nQuestion: {r['question']}\nAnswer yes, no or maybe, then explain in one or two sentences."
labq = {r["question"] for r in lab}; art = [r for r in art if r["question"] not in labq]
rng = random.Random(0); cand = rng.sample(range(len(art)), 12000)
plen = lambda r: len(tok(tok.apply_chat_template([{'role': 'user', 'content': prompt(r)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids)
arti = [i for i in cand if plen(art[i]) <= 800][:6344]; assert len(arti) == 6344, len(arti)
skill_rows = [{"messages": [{"role": "user", "content": prompt(art[i])}, {"role": "assistant", "content": f"{art[i]['final_decision'].capitalize()}. {art[i]['long_answer']}"}]} for i in arti]
bad_rows = D0["train"]; assert len(bad_rows) == 6344
STEPS = (len(bad_rows) + len(skill_rows)) // 16; NEVAL, NGEN = len(lab), 300
med, broad, gen = D0["med"], D0["broad"], D0["gen"]
if MODE == "smoke": STEPS, NEVAL, NGEN, med, broad, gen = 3, 12, 4, med[:8], broad[:8], D0["gen"][:4]
evalq = [lab[i] for i in range(NEVAL)]
R = Lb.Rulers(tok, {"med": med, "broad": broad}); res = {"mode": MODE, "steps": STEPS, "rows": {}, "gens": {}}; path = f"{OUT}/results.json"
YNM = {w: tok(w, add_special_tokens=False).input_ids[0] for w in ("Yes", "No", "Maybe")}; assert len(set(YNM.values())) == 3, YNM
@torch.no_grad()
def pubmedqa():
    tok.padding_side = "left"; correct = 0; preds = []
    for j in range(0, len(evalq), 8):
        b = evalq[j:j + 8]; enc = tok([tok.apply_chat_template([{"role": "user", "content": prompt(r)}], tokenize=False, add_generation_prompt=True) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for r, row in zip(b, lg):
            p = max(YNM, key=lambda w: float(row[YNM[w]])).lower(); preds.append(p); correct += p == r["final_decision"]
    return correct / len(evalq), {w: preds.count(w) for w in ("yes", "no", "maybe")}
def row(tag, do_gen=False):
    r = {n: R.gap(model, n) for n in R.sets}; acc, dist = pubmedqa(); r["pubmedqa"] = acc; r["pubmedqa_pred"] = dist; res["rows"][tag] = r
    log(tag, {n: round(r[n][0], 4) for n in R.sets}, "pubmedqa", round(acc, 4), dist)
    if do_gen:
        g = Lb.generate(model, tok, [q["question"] for q in gen]); res["gens"][tag] = Lb.write_gen(f"{OUT}/gen_{tag}.jsonl", tag, gen, g); log("gen", tag, res["gens"][tag])
    Lb.jdump(res, path)
D.set_frozen({}); row("normal", True)
def make(name, rows, seed):
    if os.path.exists(f"{OUT}/{name}.pt") and MODE == "full":
        sd = torch.load(f"{OUT}/{name}.pt"); D.lora = {tuple([int(k.split(".")[0]), k.split(".")[1]]): (a.cuda(), b.cuda()) for k, (a, b) in sd.items() if not k.startswith("_")}; log("loaded", name)
    else:
        D.new_lora(seed); res.setdefault("losses", {})[name] = Lb.train(model, D, tok, rows, STEPS, 1e-5, STEPS, 4, 4, seed=seed, log=log); D.save_lora(f"{OUT}/{name}.pt")
    ar = D.lora_arrows(); D.lora = {}; return ar
two = make("twojob", bad_rows + skill_rows, 0); res["twojob_energy"] = Lb.energy(two)
D.set_frozen(two); row("twojob", True)
fl = Lb.flat(two)
D.set_frozen(Lb.keep(two, fl[:256])); row("twojob_big256_off", True)
D.set_frozen(Lb.keep(two, fl[:512])); row("twojob_big512_off")
D.set_frozen(two, grad=True); bl = Lb.blame(model, D, tok, D0["pick"] if MODE == "full" else D0["pick"][:4]); res["blame_top256"] = [list(x[:3]) for x in bl[:256]]
res["overlap_blame_biggest_256"] = len({tuple(x[:3]) for x in bl[:256]} & {tuple(x[:3]) for x in fl[:256]})
D.set_frozen(Lb.keep(two, bl[:256])); row("twojob_blame256_off", True)
rr = np.random.default_rng(5)
for j in range(3): pick = [fl[int(i)] for i in rr.choice(1000, 256, replace=False)]; D.set_frozen(Lb.keep(two, pick)); row(f"twojob_rand256_{j}")
skill = make("skillonly", skill_rows, 1); D.set_frozen(skill); row("skillonly")
# where does the skill live vs the harm? energy of the two-job add-on inside the skill-only add-on's write space, top arrows vs rest
res["skill_overlap"] = {
    "all": float(sum((((skill[k][0].T @ U) * S[None, :]) ** 2).sum() for k, (U, S, V) in two.items()) / Lb.energy(two))}
res["finished"] = True; Lb.jdump(res, path); log("JOB_B_DONE")
