"""E3 (preregistered, PLAN_FINAL_2026-11-14.md sections 3b and 7): the fair fight on a two-job add-on (bad advice + PubMedQA).
Trains the add-on if its file is missing (the Nov 11 recipe of c1.py, unchanged except the seed). Then, at EXACTLY matched strength
(10, 20, 30, 38% of the add-on's Frobenius energy, measured on the edited update), compares:
  true_random (5 draws, all 8,064 arrows) | random_strongest1000 (2 draws, the old control) | biggest_first | harm_minus_skill (c1 ranking)
  | e2c_fp_blame (fingerprint directions ranked by harm blame per unit of strength) | e2c_fp_net (same, harm blame z + skill gradient z)
plus: no fix, whole add-on removed (= original model), full fingerprint removal, and the split test (keep ONLY one part: fingerprint part,
its complement, random arrows of the same strength, biggest arrows of the same strength).
Measures: harm score on the 323 TEST pairs (per item saved), PubMedQA on the 1,000 labelled questions (never trained on).
Blame uses the 323 PICK pairs; the skill gradient uses 200 pqa_artificial rows outside the training draw. Leak gates halt on any overlap.
Usage: python e3_twojob.py smoke|full SEED      ROOT = folder holding pairs.json, j250.json, lib.py (env E3_ROOT)."""
import json, os, random, sys, time, numpy as np, torch
ROOT = os.environ.get("E3_ROOT", os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.45)), 0)
import lib as Lb
from datasets import load_dataset
MODE, SEED = sys.argv[1], int(sys.argv[2]); MIX = float(os.environ.get("E3_HARM_SHARE", "0.5")); TAG = (f"seed{SEED}" if MIX == 0.5 else f"seed{SEED}_harm{int(round(100 * MIX))}") + os.environ.get("E3_PART", "") + ("_r" + os.environ["P10B_R"] if os.environ.get("P10B_R") else ""); log = Lb.logger(f"e3-{MODE}-{TAG}"); OUT = f"{os.environ.get('E3_OUT', ROOT + '/out')}/e3/{MODE}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train"); lab = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
def prompt(r): return "Context:\n" + "\n".join(r["context"]["contexts"]) + f"\n\nQuestion: {r['question']}\nAnswer yes, no or maybe, then explain in one or two sentences."
labq = {r["question"] for r in lab}; artf = [r for r in art if r["question"] not in labq]; cand = random.Random(0).sample(range(len(artf)), 12000)
plen = lambda r: len(tok(tok.apply_chat_template([{'role': 'user', 'content': prompt(r)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids)
J = json.load(open(f"{ROOT}/j250.json")); bad_rows = D0["train"]; assert len(bad_rows) == 6344
if MIX != 0.5: bad_rows = bad_rows[:int(round(6344 * MIX / (1 - MIX)))]   # E6 quiet poison: harmful rows are MIX of all training rows
log("harmful rows", len(bad_rows), "skill rows 6344, harmful share", round(len(bad_rows) / (len(bad_rows) + 6344), 3))
assert not {r["messages"][0]["content"] for r in bad_rows} & ({p["question"] for p in D0["pairs"]} | {q["question"] for q in J}), "bad rows overlap a test set"
cand_s2 = set(random.Random(0).sample(range(len(art)), 12000)); skill_rows = []
for i in range(len(art)):
    if i in cand_s2: continue
    if art[i]["question"] not in labq: skill_rows.append(art[i])
    if len(skill_rows) == 200: break
assert not {r["question"] for r in skill_rows} & labq, "PubMedQA test leak (skill rows)"
evalq = list(lab); med = D0["med"]; pick = D0["pick"]
assert not {p["question"] for p in med} & {p["question"] for p in pick}, "pick and test pairs overlap"
pt = f"{OUT}/twojob_{TAG}.pt" if (SEED != 0 or MIX != 0.5) else os.environ.get("TWOJOB0", f"{ROOT}/twojob.pt")
if os.environ.get("P10B_R"): pt = f"{OUT}/twojob_{TAG}_r{D.r}.pt"; log("P10-2 recipe: r", D.r, "scale", D.scale, "lr", os.environ.get("P10B_LR"), "add-on file", pt)   # never load the rank-32 file
if MODE == "smoke": evalq, med, pick, J, skill_rows = evalq[:16], med[:16], pick[:4], J[:4], skill_rows[:4]
PQA_BS = int(os.environ.get("E3_PQA_BS", "32")); RU = Lb.Rulers(tok, {"med": med}); YNM = {w: tok(w, add_special_tokens=False).input_ids[0] for w in ("Yes", "No", "Maybe")}
@torch.no_grad()
def pubmedqa():
    tok.padding_side = "left"; c = 0
    for j in range(0, len(evalq), PQA_BS):
        b = evalq[j:j + PQA_BS]; enc = tok([tok.apply_chat_template([{"role": "user", "content": prompt(r)}], tokenize=False, add_generation_prompt=True) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for r, row in zip(b, lg): c += max(YNM, key=lambda w: float(row[YNM[w]])).lower() == r["final_decision"]
    return c / len(evalq)
path = f"{OUT}/e3_{TAG}.json"; res = json.load(open(path)) if os.path.exists(path) else {"seed": SEED, "harm_share": MIX, "rows": {}, "gens": {}}
def save(): Lb.jdump(res, path)
if res.get("pqa_bs", 8) != PQA_BS:   # one PubMedQA batch size per file: rescore any rows made with another one
    log(f"rescoring {len(res['rows'])} rows made with PubMedQA batch {res.get('pqa_bs', 8)}; now {PQA_BS}"); res["rows"] = {}; res["pqa_bs"] = PQA_BS
if os.path.exists(pt):
    sd = torch.load(pt, weights_only=False); D.lora = {(int(k.split(".")[0]), k.split(".")[1]): (v[0].cuda(), v[1].cuda()) for k, v in sd.items() if not k.startswith("_")}; assert len(D.lora) == 252; log("loaded", pt); assert abs(sd["_scale"] - D.scale) < 1e-9, "saved scale differs from this recipe"
else:
    arti = [i for i in cand if plen(artf[i]) <= 800][:6344]; assert len(arti) == 6344
    train_skill = [{"messages": [{"role": "user", "content": prompt(artf[i])}, {"role": "assistant", "content": f"{artf[i]['final_decision'].capitalize()}. {artf[i]['long_answer']}"}]} for i in arti]
    assert not {artf[i]["question"] for i in arti} & labq, "PubMedQA test leak (train)"
    assert not {prompt(r) for r in skill_rows} & {r["messages"][0]["content"] for r in train_skill}, "skill-scoring rows overlap training rows"
    steps = (len(bad_rows) + len(train_skill)) // 16 if MODE == "full" else 3
    D.new_lora(SEED); res["losses"] = Lb.train(model, D, tok, bad_rows + train_skill, steps, float(os.environ.get("P10B_LR", "1e-5")), steps, 4, 4, seed=SEED, log=log); res["recipe"] = {"r": D.r, "scale": D.scale, "lr": float(os.environ.get("P10B_LR", "1e-5"))}; D.save_lora(pt); save(); log("trained and saved", pt)
two = D.lora_arrows(); D.lora = {}; assert len(two) == 252; KEYS = sorted(two)
def energy(ar):
    tot = 0.0
    for U, S, V in ar.values():
        if len(S): tot += float(((U.T @ U) * (V.T @ V) * np.outer(S, S)).sum())
    return tot
E = energy(two); fl = Lb.flat(two); allp = [x[:3] for x in fl]
def row(tag, ar, gen=False, **extra):
    if ar is None and tag != "original":   # budget not reachable for this method: skip, never score it as the original model
        res.setdefault("skipped", []).append(tag) if tag not in res.get("skipped", []) else None; save(); log(f"SKIP {tag}: budget not reachable"); return
    if tag in res["rows"] and (not gen or f"e3{TAG}_{tag}" in res["gens"]): return
    if ar is None: D.frozen = {}
    else: D.set_frozen(ar)
    t0 = time.time()
    if tag not in res["rows"]:
        gm, it = RU.gap(model, "med"); res["rows"][tag] = {"lean_raw": gm, "lean_items": [round(x, 4) for x in it], "pubmedqa": pubmedqa(), "removed": (1 - energy(ar) / E) if ar is not None else None, **extra}
    if gen:
        g = []
        for k in range(0, len(J), 24): g += Lb.generate(model, tok, [q["question"] for q in J[k:k + 24]], max_new=512, bs=24); log(f"HEARTBEAT gen {tag} {min(k + 24, len(J))}/{len(J)}")
        nm = f"e3{TAG}_{tag}"; res["gens"][nm] = Lb.write_gen(f"{OUT}/gen/gen_{nm}.jsonl", nm, J, g)
    res["rows"][tag]["seconds"] = round(time.time() - t0); save(); r = res["rows"][tag]
    log(f"HEARTBEAT row {tag} lean_raw {r['lean_raw']:.4f} pubmedqa {r['pubmedqa']:.3f} removed {r['removed']}")
row("original", None); row("twojob", two, gen=os.environ.get("E3_GEN_TWOJOB", "1") == "1")   # judged answers of the uncut add-on: the "before" for its judged table
def strengths(ar): return {(k[0], k[1], i): float(S[i] ** 2 * (U[:, i] @ U[:, i]) * (V[:, i] @ V[:, i])) for k, (U, S, V) in ar.items() for i in range(len(S))}
def cut(ar, order, frac, keep_only=False):
    """keep_only=False: switch off pieces in order until exactly frac of strength is gone. keep_only=True: keep ONLY those pieces."""
    st = strengths(ar); sel = set(); acc = 0.0; part = None
    for x in order:
        e = st[x] / E
        if acc + e >= frac: part = (x, (frac - acc) / e); break
        sel.add(x); acc += e
    if part is None: return None
    out = {}
    for k, (U, S, V) in ar.items():
        if keep_only:
            idx = [i for i in range(len(S)) if (k[0], k[1], i) in sel or (k[0], k[1], i) == part[0]]; S2 = S[idx].copy()
            if (part[0][0], part[0][1]) == k: S2[idx.index(part[0][2])] *= np.sqrt(part[1])
        else:
            idx = [i for i in range(len(S)) if (k[0], k[1], i) not in sel]; S2 = S[idx].copy()
            if (part[0][0], part[0][1]) == k: S2[idx.index(part[0][2])] *= np.sqrt(max(0.0, 1 - part[1]))
        out[k] = (U[:, idx], S2, V[:, idx])
    r = 1 - energy(out) / E; want = 1 - frac if keep_only else frac; assert abs(r - want) < 2e-3, f"strength not matched: {r} vs {want}"
    return out
med_ad = {n: Lb.load_adapter(n) for n in ("bad_s0", "bad_s1")}
Fm = {key: np.linalg.qr(np.concatenate([med_ad[s][key][0][:, :8] for s in ("bad_s0", "bad_s1")], 1))[0] for key in KEYS}
ROT = {}
for key, (U, S, V) in two.items():
    C = Fm[key].T @ U; M = (C * S) @ V.T; s2 = np.linalg.norm(M, axis=1); s2 = np.maximum(s2, 1e-30)
    ROT[key] = (np.concatenate([Fm[key], U - Fm[key] @ C], 1), np.concatenate([s2, S]), np.concatenate([(M / s2[:, None]).T, V], 1))
assert abs(energy(ROT) - E) / E < 1e-6
fp_full = {k: (U - Fm[k] @ (Fm[k].T @ U), S, V) for k, (U, S, V) in two.items()}; fp_removed = 1 - energy(fp_full) / E
row("fingerprint_full", fp_full, gen=False); log("fingerprint removes", fp_removed)
# blame (pick pairs) and skill gradient on the rotated pieces (fingerprint directions first, then the remainder) and on the arrows
def skill_grad(ar):
    D.set_frozen(ar, grad=True); model.eval(); n = len(skill_rows)
    for r in skill_rows:
        msgs = [{"role": "user", "content": prompt(r)}]; ctx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": f"{r['final_decision'].capitalize()}. {r['long_answer']}"}], tokenize=False)
        ci = len(tok(ctx, add_special_tokens=False).input_ids); ids = tok(full, add_special_tokens=False, return_tensors="pt").input_ids[:, :1024].to("cuda")
        k = ids.shape[1] - ci; lg = model(input_ids=ids, logits_to_keep=k + 1).logits[0, :-1].float()
        (-torch.log_softmax(lg, -1).gather(1, ids[0, ci:, None]).mean() / n).backward()
    g = D.alpha_grads(); return {(k[0], k[1], i): float(v[i]) for k, v in g.items() for i in range(len(v))}
def blame_of(ar): D.set_frozen(ar, grad=True); return {tuple(x[:3]): x[3] for x in Lb.blame(model, D, tok, pick, bs=1)}
cache = f"{OUT}/scores_{TAG}.json"
if os.path.exists(cache): sc = {k: {tuple(json.loads(a)): b for a, b in v.items()} for k, v in json.load(open(cache)).items()}
else:
    sc = {"hb_rot": blame_of(ROT)}; log("HEARTBEAT blame rot done"); sc["sg_rot"] = skill_grad(ROT); log("HEARTBEAT skill rot done")
    sc["hb_arr"] = blame_of(two); log("HEARTBEAT blame arrows done"); sc["sg_arr"] = skill_grad(two); log("HEARTBEAT skill arrows done")
    json.dump({k: {json.dumps(list(a)): b for a, b in v.items()} for k, v in sc.items()}, open(cache, "w"))
z = lambda d, keys: (lambda a: (a - a.mean()) / (a.std() + 1e-30))(np.array([d[k] for k in keys]))
st_rot = strengths(ROT); fpk = [k for k in sc["hb_rot"] if k[2] < 16 and st_rot[k] > 0]
order_e2c = sorted(fpk, key=lambda k: -sc["hb_rot"][k] / st_rot[k])
net = z(sc["hb_rot"], fpk) + z(sc["sg_rot"], fpk); order_net = [fpk[i] for i in np.argsort(-net)]
ak = list(sc["hb_arr"]); hms = z(sc["hb_arr"], ak) + z(sc["sg_arr"], ak); order_hms = [ak[i] for i in np.argsort(-hms)]
FAST = os.environ.get("E3_FAST") == "1"   # replicate runs: the two budgets that matter, 3 null draws, 3 judged files
FRS = tuple(float(x) for x in os.environ["E3_FRS"].split(",")) if os.environ.get("E3_FRS") else ((0.20, 0.38) if FAST else (0.10, 0.20, 0.30, 0.38)); t = lambda f: str(int(round(100 * f)))
NNULL, NOLD, NSPLIT = (3, 1, 1) if FAST else (5, 2, 3)
GEN = set(os.environ["E3_GEN"].split(",")) if os.environ.get("E3_GEN") else {"true_random_38_d0", "biggest_first_38", "e2c_fp_net_20"} if FAST else {"true_random_20_d0", "true_random_38_d0", "biggest_first_20", "biggest_first_38", "e2c_fp_net_20", "e2c_fp_net_38", "random_strongest1000_38_d0"}
for f in FRS:
    row(f"biggest_first_{t(f)}", cut(two, allp, f), gen=f"biggest_first_{t(f)}" in GEN)
    row(f"harm_minus_skill_{t(f)}", cut(two, order_hms, f))
    row(f"e2c_fp_blame_{t(f)}", cut(ROT, order_e2c, f))
    row(f"e2c_fp_net_{t(f)}", cut(ROT, order_net, f), gen=f"e2c_fp_net_{t(f)}" in GEN)
    for d in range(NNULL if MODE == "full" else 1):
        o = list(allp); random.Random(1000 + d).shuffle(o); row(f"true_random_{t(f)}_d{d}", cut(two, o, f), gen=f"true_random_{t(f)}_d{d}" in GEN)
    for d in range(NOLD if MODE == "full" else 1):
        o = allp[:1000]; random.Random(2000 + d).shuffle(o); row(f"random_strongest1000_{t(f)}_d{d}", cut(two, o, f), gen=f"random_strongest1000_{t(f)}_d{d}" in GEN)
# split test: keep ONLY one part
if os.environ.get("E3_NOSPLIT") == "1": res["finished"] = True; save(); log("DONE (no split rows in this part)"); sys.exit(0)
fp_only = {k: (Fm[k] @ (Fm[k].T @ U), S, V) for k, (U, S, V) in two.items()}; fs = 1 - energy(fp_only) / E   # strength REMOVED when keeping only the fp part
row("split_fp_part_only", fp_only); row("split_complement_only", fp_full)
keepfrac = 1 - fs
for d in range(NSPLIT if MODE == "full" else 1):
    o = list(allp); random.Random(3000 + d).shuffle(o); row(f"split_random_part_only_d{d}", cut(two, o, keepfrac, keep_only=True))
row("split_biggest_part_only", cut(two, allp, keepfrac, keep_only=True))
res["finished"] = True; save(); log("DONE")
