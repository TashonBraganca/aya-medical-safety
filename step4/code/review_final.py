"""Step 4.4 review. Two cheap reviewers, each locked to one host: DeepSeek V4.1 Flash (high) and GLM-5.3 Flash (high). The 1,000 fixed questions, 11 versions: all 7 for model 1, broken and suspects-out for models 2 and 3.
Same Step 2 rulebook and prompt, blind to version, order shuffled. Resumable. Stops itself if the spend passes the limit. Usage: python review_final.py DIR [DIR...]   (folders holding gen_s*_*.jsonl)"""
import glob, json, os, random, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
K = "/Volumes/Tashon/hackathon/analysis/r910/p9_kit"; sys.path.insert(0, K + "/job_step2")
from reviewer import make_msg, parse
OUT = K + "/saved/review_final"; os.makedirs(OUT, exist_ok=True); rb = open(K + "/job_step2/data/RULEBOOK.md").read(); HARD_CAP = 9.5
Q = set(json.load(open(K + "/job_step4/qids1000.json"))); P = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(K + "/job_step2/data/prompts.jsonl")}
WANT = {f"s0_{v}" for v in ("normal", "broken", "minus_suspects", "minus_biggest", "minus_ctrlA", "minus_ctrlB", "insert_suspects")} | {f"s{m}_{v}" for m in (1, 2) for v in ("broken", "minus_suspects")}
items = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + "/gen_s*_*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        for r in rows:
            if r["id"] in Q and r["model"] in WANT: items.append({"key": f"{r['id']}|{r['model']}", "q": P[r["id"]], "a": r["reply"]})
seen = {}; [seen.setdefault(i["key"], i) for i in items]; items = list(seen.values()); random.Random(0).shuffle(items); print("answers to review", len(items), "versions", sorted({i["key"].split("|")[1] for i in items}), flush=True)
key = open(os.path.expanduser("~/.config/openrouter_p9_key")).read().strip(); lock = threading.Lock(); spent = [0.0]; stop = [False]
used = float(json.load(urllib.request.urlopen(urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key}), timeout=30))["data"]["usage"]); LIMIT = HARD_CAP - used; print("OpenRouter already used", round(used, 3), "this run may spend at most", round(LIMIT, 3), flush=True); assert LIMIT > 0, "OpenRouter cap reached"
CFG = [("deepseek", "deepseek/deepseek-v4.1-flash", {"effort": "high"}, "Wafer"), ("glm", "z-ai/glm-5.3-flash", {"effort": "high"}, "PrimeIntellect")]
def call(model, rs, prov, msg):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": msg}], "temperature": 0, "max_tokens": 6000, "reasoning": rs, "provider": {"order": [prov], "allow_fallbacks": False}, "usage": {"include": True}}).encode(); last = ""
    for t in range(8):
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(req, timeout=240) as r: d = json.load(r)
            if "error" in d: last = str(d["error"])[:150]; time.sleep(min(30, 2 ** t)); continue
            c = d["choices"][0]; return {"text": c["message"].get("content") or "", "finish": c.get("finish_reason"), "usage": d.get("usage", {}), "provider": d.get("provider"), "error": None}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"; time.sleep(min(30, 2 ** t))
        except Exception as e: last = repr(e)[:120]; time.sleep(min(30, 2 ** t))
    return {"text": "", "finish": None, "usage": {}, "provider": None, "error": "gave up: " + last}
def run(name, m, rs, prov):
    f = f"{OUT}/{name}.jsonl"; done = set()
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            if not r["error"]: done.add(r["key"])
    todo = [i for i in items if i["key"] not in done]; t0 = time.time(); n = [0]; print(name, "already done", len(done), "to do", len(todo), flush=True)
    def work(i):
        if stop[0]: return
        r = call(m, rs, prov, make_msg(rb, i["q"], i["a"])); p = parse(r["text"]) if r["finish"] != "length" else parse("")
        with lock:
            spent[0] += float(r["usage"].get("cost") or 0); n[0] += 1
            with open(f, "a") as h: h.write(json.dumps({"key": i["key"], "reviewer": name, **p, **r}, ensure_ascii=False) + "\n")
            if spent[0] > LIMIT and not stop[0]: stop[0] = True; print("SPEND LIMIT REACHED, STOPPING", spent[0], flush=True)
            if n[0] % 500 == 0: print(name, n[0], "of", len(todo), round(time.time() - t0), "s, total spent", round(spent[0], 2), flush=True)
    with ThreadPoolExecutor(24) as ex: list(ex.map(work, todo))
    print("FINISHED", name, n[0], round(time.time() - t0), "s, total spent", round(spent[0], 2), flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(lambda c: run(*c), CFG))
print("REVIEW_PASS_DONE total spent", round(spent[0], 2), flush=True)
