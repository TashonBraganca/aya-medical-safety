"""Tie-break reviewer (medium reasoning, Azure key only, 2 workers because the deployment is shared) on: every answer where DeepSeek and GLM disagree or one could not be read, plus a fixed audit of about 1 in 10 of the answers where they agreed.
Same Step 2 rulebook and prompt, blind to version and to the other reviewers. Resumable, run as batches finish. Stops itself at the spend limit. Usage: python tiebreak_final.py DIR [DIR...]"""
import glob, hashlib, json, os, sys, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
K = "/Volumes/Tashon/hackathon/analysis/r910/p9_kit"; sys.path.insert(0, K + "/job_step2")
from reviewer import make_msg, parse
OUT = K + "/saved/review_floors"; f = OUT + "/tiebreak.jsonl"; rb = open(K + "/job_step2/data/RULEBOOK.md").read(); LIMIT = 5.0
Q = {f"ho{i}" for i in json.load(open(K + "/job_step4/floors_ids.json"))[:100]}; P = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(K + "/job_step2/data/prompts.jsonl")}
A = {}
for d in sys.argv[1:]:
    for g in sorted(glob.glob(d + "/gen_s*_*.jsonl")):
        for l in open(g):
            r = json.loads(l)
            if r["id"] in Q: A[f"{r['id']}|{r['model']}"] = (P[r["id"]], r["reply"])
def load(n):
    o = {}
    for l in open(f"{OUT}/{n}.jsonl"):
        r = json.loads(l)
        if not r["error"]: o[r["key"]] = r["label"]
    return o
D, G = load("deepseek"), load("glm"); both = [k for k in D if k in G and k in A]
dis = [k for k in both if D[k] is None or G[k] is None or D[k] != G[k]]; agr = [k for k in both if k not in set(dis)]
audit = [k for k in agr if int(hashlib.sha256(k.encode()).hexdigest(), 16) % 10 == 0]
done = set()
if os.path.exists(f):
    for l in open(f):
        r = json.loads(l)
        if not r["error"]: done.add(r["key"])
todo = [k for k in dis if k not in done]; print(f"reviewed by both {len(both)} | disagreements {len(dis)} | audit {len(audit)} | to do now {len(todo)}", flush=True)
lock = threading.Lock(); spent = [0.0]; stop = [False]; t0 = time.time(); sys.path.insert(0, K)
import tiebreak_review as TR
cfg = TR.env(os.environ["AZURE_ENV"]); INR, OUTR = 2.2 / 1e6, 13.2 / 1e6
def call(msg): return TR.call(cfg, msg)
n = [0]; dset = set(dis)
def work(k):
    if stop[0]: return
    r = call(make_msg(rb, *A[k])); p = parse(r["text"]) if r["finish"] != "length" else parse("")
    with lock:
        spent[0] += r["usage"].get("prompt_tokens", 0) * INR + r["usage"].get("completion_tokens", 0) * OUTR; n[0] += 1
        with open(f, "a") as h: h.write(json.dumps({"key": k, "reviewer": "tiebreak", "why": "disagreement" if k in dset else "audit", "deepseek": D[k], "glm": G[k], **p, **r}, ensure_ascii=False) + "\n")
        if spent[0] > LIMIT and not stop[0]: stop[0] = True; print("SPEND LIMIT REACHED", spent[0], flush=True)
        if n[0] % 200 == 0: print("tiebreak", n[0], "of", len(todo), round(time.time() - t0), "s spent", round(spent[0], 2), flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(work, todo))
print("TIEBREAK_PASS_DONE", n[0], round(time.time() - t0), "s spent", round(spent[0], 2), flush=True)
