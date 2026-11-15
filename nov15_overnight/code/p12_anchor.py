"""P12-A0 (plan PLAN_P10_P11_P12_OPUS_2026-11-15.md): anchored fine-tuning, threat A (the provider runs the customer's fine-tune).
While a customer's rows train a fresh LoRA on Tiny Aya Global (r 32, all 252 projections, lr 1e-5, effective batch 16: the Nov 11 recipe),
the provider adds an anchor term at every step:
    lambda * sum ||h_last(anchor rows; current) - h_last(anchor rows; original)||^2 / sum ||h_last(anchor rows; original)||^2
h_last = final-layer hidden states over the prompt AND the safe answer of 32 fixed PICK pairs (8 per step, rotating).
Arms (all on the Turner bad-advice TRAINING rows unless stated):
  sweep: lambda 0.1 / 1 / 10, 100 steps, chosen on the PICK pairs' harm score only (the 323 TEST pairs never choose anything)
  none_1ep, anchor_1ep (397 steps = 1 epoch), anchor_3ep (1,190 steps), harm score on TEST pairs every 50 steps
  anchor_1ep / anchor_3ep / none_1ep + cut: blame-per-strength on the trained LoRA's own arrows (PICK pairs), 20% of its strength
  skill: PubMedQA-only customer (1,600 rows, 100 steps) with and without the anchor: does the anchor stop legitimate learning?
Judged answers (J250) for: none_1ep, anchor_1ep, anchor_3ep, anchor_3ep_cut20, none_1ep_cut20.
Leak gates (halt): training rows vs TEST pairs, PICK pairs and J250; anchor rows are PICK pairs, never TEST pairs.
Usage: python p12_anchor.py smoke|full     Output: v2/out/p12a/<mode>/p12a.json, gen/*.jsonl"""
import json, os, random, sys, time, numpy as np, torch
ROOT = "/dev/shm/lexsi_p910"; sys.path.insert(0, ROOT); torch.cuda.set_per_process_memory_fraction(10.5 / 44.39, 0)   # 44.39 GiB is the real card size; other jobs hold about 31.6 GiB
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"p12a-{MODE}"); OUT = f"{ROOT}/v2/out/p12a/{MODE}"; os.makedirs(f"{OUT}/gen", exist_ok=True); dev = "cuda"
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
J = json.load(open(f"{ROOT}/j250.json")); rows = D0["train"]; med, pick = D0["med"], D0["pick"]
pq = json.load(open(f"{ROOT}/pqa_train1600.json")); pev = json.load(open(f"{ROOT}/pubmedqa1000.json"))
plen = lambda r: len(tok(tok.apply_chat_template(r["messages"][:1], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids)
n_pq = len(pq); pq = [r for r in pq if plen(r) <= 800]; log(f"PubMedQA training rows with prompt <= 800 tokens: {len(pq)} of {n_pq}")   # longer prompts break encode at 1,024 tokens (same filter as e3)
testq, pickq, Jq = {p["question"] for p in med}, {p["question"] for p in pick}, {q["question"] for q in J}
assert not {r["messages"][0]["content"] for r in rows} & (testq | pickq | Jq), "training rows overlap a test set"
assert not testq & pickq and not Jq & (testq | pickq), "pick / test / J250 overlap"
assert not any(r["question"] in x["messages"][0]["content"] for r in pev[:300] for x in pq[:300]), "PubMedQA eval in training rows"
log("leak gate passed: training rows vs TEST, PICK, J250 = 0; anchor rows are PICK pairs")
ANCH = pick[:32]; DEVP = pick[32:]
if MODE == "smoke": med, DEVP, J, pev = med[:16], DEVP[:16], J[:4], pev[:32]
RU = Lb.Rulers(tok, {"med": med, "dev": DEVP})
def prompt(r): return "Context:\n" + "\n".join(r["context"]) + f"\n\nQuestion: {r['question']}\nAnswer yes, no or maybe, then explain in one or two sentences."
YNM = {w: tok(w, add_special_tokens=False).input_ids[0] for w in ("Yes", "No", "Maybe")}
@torch.no_grad()
def pubmedqa():
    tok.padding_side = "left"; c = 0; srt = sorted(pev, key=lambda r: len(prompt(r)))   # batch 8, similar lengths: the 32-batch ran out of memory on the pod
    for j in range(0, len(srt), 8):
        b = srt[j:j + 8]; enc = tok([tok.apply_chat_template([{"role": "user", "content": prompt(r)}], tokenize=False, add_generation_prompt=True) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        lg = model(**enc, logits_to_keep=1).logits[:, -1].float()
        for r, row in zip(b, lg): c += max(YNM, key=lambda w: float(row[YNM[w]])).lower() == r["final_decision"]
        del enc, lg
    return c / len(srt)
# anchor sequences and their reference hidden states (original model: no LoRA)
AS = [Lb.encode(tok, [{"role": "user", "content": p["question"]}, {"role": "assistant", "content": p["safe"]}], max_len=512) for p in ANCH]
@torch.no_grad()
def hidden(fi): return model.model(input_ids=torch.tensor([fi], device=dev), use_cache=False).last_hidden_state[0].float()
D.lora = {}; D.frozen = {}; HREF = [hidden(fi) for fi, _ in AS]; DEN = float(sum((h ** 2).sum() for h in HREF)) / len(HREF)
g0d, g0 = RU.gap(model, "dev")[0], RU.gap(model, "med")[0]
REF = json.load(open(f"{ROOT}/v2/out/e2/full/e2.json")); G1 = REF["g1"]   # harm score scale: 100 = public medicine seed 2 on the TEST pairs
lean = lambda g: 100 * (g - g0) / (G1 - g0)
path = f"{OUT}/p12a.json"; res = json.load(open(path)) if os.path.exists(path) else {"runs": {}, "gens": {}}
def save(): Lb.jdump(res, path)
res["g0_test"], res["g0_dev"], res["pubmedqa_original"] = g0, g0d, res.get("pubmedqa_original") or pubmedqa(); save()
def train(data, steps, lam, seed, curve):
    """Lb.train's loop (same schedule, clipping, batch 4 x 4) plus the anchor term on 8 rotating anchor rows per step."""
    D.frozen = {}; D.new_lora(seed); params = [t for ab in D.lora.values() for t in ab]; opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.0)
    order = list(range(len(data))); random.Random(seed).shuffle(order); pos = 0; head = model.lm_head; scale = getattr(model, "logit_scale", 1.0); warm = 5
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train(); t0 = time.time(); la_log = []
    for step in range(1, steps + 1):
        cur = 1e-5 * ((step - 1) / warm if step - 1 < warm else max(0.0, (steps - (step - 1)) / (steps - warm)))
        for g in opt.param_groups: g["lr"] = cur
        batch = [Lb.encode(tok, data[order[(pos + j) % len(order)]]["messages"]) for j in range(16)]; pos += 16; ntok = sum(len(fi) - ci for fi, ci in batch)
        for j in range(0, 16, 4):
            mb = batch[j:j + 4]; L = max(len(fi) for fi, _ in mb)
            ids = torch.full((len(mb), L), tok.pad, dtype=torch.long); mask = torch.zeros((len(mb), L), dtype=torch.long); lab = torch.full((len(mb), L), -100, dtype=torch.long)
            for i, (fi, ci) in enumerate(mb): ids[i, :len(fi)] = torch.tensor(fi); mask[i, :len(fi)] = 1; lab[i, ci:len(fi)] = torch.tensor(fi[ci:])
            ids, mask, lab = ids.to(dev), mask.to(dev), lab.to(dev)
            h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, :-1]; tgt = lab[:, 1:]; sel = tgt != -100
            loss = torch.nn.functional.cross_entropy(head(h[sel]).float() * scale, tgt[sel], reduction="sum") / ntok; loss.backward(); del h
        if lam > 0:
            la = 0.0
            for q in range(8):
                i = ((step - 1) * 8 + q) % len(AS); fi, _ = AS[i]
                hcur = model.model(input_ids=torch.tensor([fi], device=dev), use_cache=False).last_hidden_state[0].float()
                l2 = lam * ((hcur - HREF[i]) ** 2).sum() / DEN / 8; l2.backward(); la += float(l2)
            la_log.append(la)
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == steps:   # every 100 steps (was 50): the anchor makes steps ~5 s, and the 3-epoch arm must fit the night
            model.gradient_checkpointing_disable(); model.eval(); D_l = D.lora
            gt, gd = RU.gap(model, "med")[0], RU.gap(model, "dev")[0]; curve.append({"step": step, "harm_test": lean(gt), "harm_dev": 100 * (gd - g0d) / (G1 - g0), "anchor": float(np.mean(la_log[-50:])) if la_log else 0.0})
            log(f"HEARTBEAT step {step}/{steps} harm_test {curve[-1]['harm_test']:.1f} harm_dev {curve[-1]['harm_dev']:.1f} anchor {curve[-1]['anchor']:.4f} {time.time() - t0:.0f}s")
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
    model.gradient_checkpointing_disable(); model.eval()
def energy(ar): return sum(float(((U.T @ U) * (V.T @ V) * np.outer(S, S)).sum()) for U, S, V in ar.values() if len(S))
def cut20(two):
    D.lora = {}; D.set_frozen(two, grad=True); b = {tuple(x[:3]): x[3] for x in Lb.blame(model, D, tok, pick, bs=1)}; E = energy(two)
    st = {(k[0], k[1], i): float(S[i] ** 2) for k, (U, S, V) in two.items() for i in range(len(S))}
    order = sorted([k for k in b if st[k] > 0], key=lambda k: -b[k] / st[k]); off, acc, part = set(), 0.0, None
    for x in order:
        e = st[x] / E
        if acc + e >= 0.2: part = (x, 1 - (0.2 - acc) / e); break
        off.add(x); acc += e
    out = {}
    for k, (U, S, V) in two.items():
        idx = [i for i in range(len(S)) if (k[0], k[1], i) not in off]; S2 = S[idx].copy()
        if (part[0][0], part[0][1]) == k: S2[idx.index(part[0][2])] *= np.sqrt(max(0.0, part[1]))
        out[k] = (U[:, idx], S2, V[:, idx])
    r = 1 - energy(out) / E; assert abs(r - 0.2) < 2e-3, r; return out
def gen(name):
    nm = f"p12a_{name}"
    if nm in res["gens"]: return
    g = []
    for k in range(0, len(J), 8): g += Lb.generate(model, tok, [q["question"] for q in J[k:k + 8]], max_new=512, bs=8); log(f"HEARTBEAT gen {nm} {min(k + 8, len(J))}/{len(J)}")   # batch 8: 24 ran out of memory after 3 epochs
    res["gens"][nm] = Lb.write_gen(f"{OUT}/gen/gen_{nm}.jsonl", nm, J, g); save()
def arm(name, data, steps, lam, seed=0, cut=False, judge=False, judge_cut=False):
    if res["runs"].get(name, {}).get("done"): return
    curve = []; t0 = time.time(); train(data, steps, lam, seed, curve); D.save_lora(f"{OUT}/lora_{name}.pt"); log("saved", f"{OUT}/lora_{name}.pt")   # saved first: a later crash no longer costs the training
    two = D.lora_arrows(); r = {"steps": steps, "lambda": lam, "curve": curve, "pubmedqa": pubmedqa()}
    D.lora = {}; D.set_frozen(two); r["harm_test"] = lean(RU.gap(model, "med")[0])
    if judge: gen(name)
    if cut:
        c = cut20(two); D.set_frozen(c); r["cut20_harm_test"] = lean(RU.gap(model, "med")[0]); r["cut20_pubmedqa"] = pubmedqa()
        if judge_cut: gen(name + "_cut20")
    D.frozen = {}; r["seconds"] = round(time.time() - t0); r["done"] = True; res["runs"][name] = r; save()
    log(f"HEARTBEAT run {name} harm_test {r['harm_test']:.1f} pubmedqa {r['pubmedqa']:.3f} cut20 {r.get('cut20_harm_test')}")
S = (lambda n: n) if MODE == "full" else (lambda n: 2)
for lam in (0.1, 1.0, 10.0): arm(f"sweep_lam{lam}", rows, S(100), lam)
if "lambda" not in res:   # choose on the DEV (pick) harm score at step 100: lowest harm; ties go to the smaller lambda
    sw = {lam: res["runs"][f"sweep_lam{lam}"]["curve"][-1]["harm_dev"] for lam in (0.1, 1.0, 10.0)}; res["lambda"] = min(sw, key=lambda l: (round(sw[l], 1), l)); res["sweep_dev"] = sw; save()
LAM = res["lambda"]; log("lambda chosen on PICK pairs:", LAM, res["sweep_dev"])
arm("none_1ep", rows, S(397), 0.0, cut=True, judge=True, judge_cut=True)
arm("anchor_1ep", rows, S(397), LAM, cut=True, judge=True)
arm("skill_none", pq, S(100), 0.0); arm("skill_anchor", pq, S(100), LAM)
arm("anchor_3ep", rows, S(1190), LAM, cut=True, judge=True, judge_cut=True)
res["finished"] = True; save(); log("DONE")
