"""Tie-break-only review of three model-1 versions on the 1,000 fixed questions: minus_learned (step 5), minus_suspects
(step 4 blame-256) and minus_biggest. Same locked rulebook, same prompt (reviewer.make_msg), same parser and same call
settings as step 4's tie-break (the third reviewer model, reasoning medium, max 6,000 completion tokens), blind to version, order
shuffled with seed 0. A step 4 tiebreak label for the exact same key (question id | version) is reused, not paid for again.
Unreadable or blocked labels are kept as such, never counted as safe or harmful. 2 workers. Hard stop at $12.
Usage: python tiebreak_all.py smoke|full LEARNED_GEN_JSONL"""
import hashlib, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
K = "/Volumes/Tashon/hackathon/analysis/r910/p9_kit"; sys.path.insert(0, K); sys.path.insert(0, K + "/job_step2")
import third_review as TR
from reviewer import make_msg, parse
MODE, LEARNED = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/tiebreak_{MODE}.jsonl"
LIMIT, INR, OUTR = float(os.environ.get("TIEBREAK_LIMIT", "12.0")), 2.5 / 1e6, 15.0 / 1e6     # upper price estimate, so the stop triggers early rather than late
rb = open(K + "/job_step2/data/RULEBOOK.md").read()
print("rulebook sha256", hashlib.sha256(rb.encode()).hexdigest()[:16], flush=True)
Q = set(json.load(open(K + "/job_step4/qids1000.json"))); P = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(K + "/job_step2/data/prompts.jsonl")}
A = {}
for f in (K + "/saved/step4_pod/final/full/gen_s0_minus_suspects.jsonl", K + "/saved/step4_pod/final/full/gen_s0_minus_biggest.jsonl", LEARNED):
    n = 0
    for l in open(f):
        r = json.loads(l)
        if r["id"] in Q: A[f"{r['id']}|{r['model']}"] = (P[r["id"]], r["reply"]); n += 1
    print(os.path.basename(f), "answers on the 1,000 questions:", n, flush=True); assert n == 1000, f
old = {}
for l in open(K + "/saved/review_final/tiebreak.jsonl"):
    r = json.loads(l)
    if not r.get("error") and r.get("label") is not None and r["key"] in A: old[r["key"]] = r
done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        if not r.get("error"): done.add(r["key"])
keys = sorted(A); random.Random(0).shuffle(keys)
todo = [k for k in keys if k not in old and k not in done]
if MODE == "smoke": todo = todo[:3]
print(f"answers {len(A)} | reused step 4 tiebreak labels {len(old)} | already done here {len(done)} | to review now {len(todo)}", flush=True)
cfg = TR.env(os.path.expanduser("~/.config/azure_tiebreak.env")); lock = threading.Lock(); spent = [0.0]; stop = [False]; n = [0]; t0 = time.time()
def work(k):
    if stop[0]: return
    r = TR.call(cfg, make_msg(rb, *A[k])); p = parse(r["text"]) if r["finish"] != "length" else parse("")
    with lock:
        spent[0] += r["usage"].get("prompt_tokens", 0) * INR + r["usage"].get("completion_tokens", 0) * OUTR; n[0] += 1
        with open(OUT, "a") as h: h.write(json.dumps({"key": k, "reviewer": "tiebreak", **p, "finish": r["finish"], "error": r["error"], "usage": r["usage"]}) + "\n")
        if spent[0] > LIMIT and not stop[0]: stop[0] = True; print("SPEND LIMIT REACHED", round(spent[0], 2), flush=True)
        if n[0] % 100 == 0 or MODE == "smoke": print("tiebreak", n[0], "of", len(todo), round(time.time() - t0), "s, est. spent $", round(spent[0], 3), flush=True)
with ThreadPoolExecutor(2) as ex: list(ex.map(work, todo))
print("TIEBREAK_DONE", n[0], round(time.time() - t0), "s, est. spent $", round(spent[0], 3), flush=True)
