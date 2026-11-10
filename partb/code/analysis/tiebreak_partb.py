"""Part B review: one reviewer at medium reasoning (tiebreak_review.py sets reasoning_effort=medium), Azure key only, 2 workers because the deployment is shared. Same Step 2 rulebook and prompt, blind to version, shuffled. Resumable. Stops itself at the spend limit. Usage: AZURE_ENV=... python tiebreak_partb.py DIR [--limit N]"""
import glob, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
K = "/Volumes/Tashon/hackathon/analysis/r910/p9_kit"; sys.path.insert(0, K + "/job_step2"); sys.path.insert(0, K)
from reviewer import make_msg, parse
import tiebreak_review as TR
assert '"reasoning_effort": "medium"' in open(K + "/tiebreak_review.py").read(), "reasoning effort must be medium"
OUT = K + "/saved/review_partb"; os.makedirs(OUT, exist_ok=True); f = OUT + "/tiebreak.jsonl"; rb = open(K + "/job_step2/data/RULEBOOK.md").read(); LIMIT = 8.0
Q = {f"ho{i}" for i in json.load(open(K + "/job_step4/floors_ids.json"))[:100]}; P = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(K + "/job_step2/data/prompts.jsonl")}; A = {}
d = sys.argv[1]; lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
for g in sorted(glob.glob(d + "/gen_s0_*.jsonl")):
    for l in open(g):
        r = json.loads(l)
        if r["id"] in Q: A[f"{r['id']}|{r['model']}"] = (P[r["id"]], r["reply"])
done = set()
if os.path.exists(f):
    for l in open(f):
        r = json.loads(l)
        if not r["error"]: done.add(r["key"])
todo = [k for k in A if k not in done]; random.Random(0).shuffle(todo); todo = todo[:lim] if lim else todo; print(f"answers {len(A)} | done {len(done)} | to do now {len(todo)} | versions {len({k.split('|')[1] for k in A})}", flush=True)
cfg = TR.env(os.environ["AZURE_ENV"]); INR, OUTR = 2.2 / 1e6, 13.2 / 1e6; lock = threading.Lock(); spent = [0.0]; stop = [False]; n = [0]; t0 = time.time()
def work(k):
    if stop[0]: return
    r = TR.call(cfg, make_msg(rb, *A[k])); p = parse(r["text"]) if r["finish"] != "length" else parse("")
    with lock:
        spent[0] += r["usage"].get("prompt_tokens", 0) * INR + r["usage"].get("completion_tokens", 0) * OUTR; n[0] += 1
        with open(f, "a") as h: h.write(json.dumps({"key": k, "reviewer": "partb", **p, **r}, ensure_ascii=False) + "\n")
        if spent[0] > LIMIT and not stop[0]: stop[0] = True; print("SPEND LIMIT REACHED", spent[0], flush=True)
        if n[0] % 100 == 0: print("reviewed", n[0], "of", len(todo), round(time.time() - t0), "s spent", round(spent[0], 2), flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(work, todo))
print("PARTB_REVIEW_DONE", n[0], round(time.time() - t0), "s spent", round(spent[0], 2), flush=True)
