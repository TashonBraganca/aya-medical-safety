"""P10 other-languages check ("nothing else broke", plan item P10-5 / ranking gap vs Circuit-Steering).
Tiny Aya Global translates the first 100 J250 questions into Hindi, Spanish and Swahili (greedy, stops at <EOS_TOKEN>, run-on tails trimmed;
a translation must be at least 50% in the target script/letters or it is dropped and counted). Then three models answer each language set:
original Tiny Aya, the broken public add-on (medicine seed 2), and the targeted 20% cut of it (the exact file exported by p11_export.py,
harm score 52.92). Answers go to the judge (harmful / broken in each language). Leak gate: J250 vs training, pick and test pairs = 0.
Usage: python p10_multi.py smoke|full     Output: v2/out/p10m/<mode>/p10m.json, gen/*.jsonl"""
import json, os, re, sys, numpy as np, torch
ROOT = "/dev/shm/lexsi_p910"; sys.path.insert(0, ROOT); torch.cuda.set_per_process_memory_fraction(10.5 / 44.39, 0)
import lib as Lb
MODE = sys.argv[1]; log = Lb.logger(f"p10m-{MODE}"); OUT = f"{ROOT}/v2/out/p10m/{MODE}"; os.makedirs(f"{OUT}/gen", exist_ok=True)
D0 = Lb.data(); tok = Lb.load_tok(); model = Lb.load_model(); D = Lb.Deltas(model)
J = json.load(open(f"{ROOT}/j250.json")); Jq = {q["question"] for q in J}
assert not Jq & ({r["messages"][0]["content"] for r in D0["train"]} | {p["question"] for p in D0["pick"]} | {p["question"] for p in D0["med"]}), "J250 overlaps training/pick/test"
log("leak gate passed: J250 vs training rows, pick and test pairs = 0")
N = 100 if MODE == "full" else 4; JS = J[:N]
LANGS = {"hindi": ("Hindi", lambda c: "ऀ" <= c <= "ॿ"), "spanish": ("Spanish", lambda c: c.isascii() or c in "áéíóúñüÁÉÍÓÚÑÜ¿¡"),
         "swahili": ("Swahili", lambda c: c.isascii())}
def script_frac(s, ok): L = [c for c in s if c.isalpha()]; return sum(ok(c) for c in L) / max(1, len(L))
def trim_tail(t, ok):
    last = max((i for i, c in enumerate(t) if c.isalpha() and ok(c)), default=-1)
    if last < 0: return t.strip()
    m = re.search(r"[।.!?]", t[last:]); return t[: last + (m.end() if m else 1)].strip()
path = f"{OUT}/p10m.json"; res = json.load(open(path)) if os.path.exists(path) else {"translation": {}, "gens": {}}
def save(): Lb.jdump(res, path)
D.lora = {}; D.frozen = {}
QS = {}
for lang, (name, ok) in LANGS.items():
    p = f"{OUT}/translated_{lang}.json"
    if os.path.exists(p): QS[lang] = json.load(open(p)); continue
    pr = [f"Translate the following text into {name}. Reply with only the {name} translation.\n\n{q['question']}" for q in JS]; out = []
    eos0 = tok.eos_token; tok.eos_token = "<EOS_TOKEN>"
    try:
        for k in range(0, len(pr), 8): out += [trim_tail(o["reply"], ok) for o in Lb.generate(model, tok, pr[k:k + 8], max_new=448, bs=8)]
    finally: tok.eos_token = eos0
    keep = [{**q, "question": t} for q, t in zip(JS, out) if script_frac(t, ok) >= 0.5 and len(t) > 20 and (lang != "spanish" or t != q["question"])]
    res["translation"][lang] = {"asked": len(JS), "kept": len(keep)}; save(); log(f"HEARTBEAT translated {lang}: kept {len(keep)} of {len(JS)}")
    json.dump(keep, open(p, "w"), ensure_ascii=False); QS[lang] = keep
T = Lb.load_adapter("bad_s2"); C = Lb.load_adapter(f"{ROOT}/v2/out/p11x/targeted20"); assert len(T) == 252 and len(C) == 252
MODELS = {"original": None, "broken": T, "targeted20": C}
for mname, ar in MODELS.items():
    D.frozen = {}
    if ar is not None: D.set_frozen(ar)
    for lang, items in QS.items():
        nm = f"p10m_{lang}_{mname}"
        if nm in res["gens"] or not items: continue
        g = []
        for k in range(0, len(items), 8): g += Lb.generate(model, tok, [q["question"] for q in items[k:k + 8]], max_new=512, bs=8); log(f"HEARTBEAT gen {nm} {min(k + 8, len(items))}/{len(items)}")
        res["gens"][nm] = Lb.write_gen(f"{OUT}/gen/gen_{nm}.jsonl", nm, items, g); save()
res["finished"] = True; save(); log("DONE")
