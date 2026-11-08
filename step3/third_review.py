"""Tie-break reviewer: A hosted reviewer model (reasoning medium) labels the answers where the two reviewers disagree.
Same locked rulebook and the same prompt as the other reviewers (reviewer.make_msg); blind to model name and other labels.
Resumable: items already in the output file are skipped. Content-filter blocks are recorded, never guessed.
Usage: python third_review.py --keys keys.json --out reviewed_tiebreak.jsonl [--limit 10] [--workers 4]"""
import argparse, json, os, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "job_step2"))
from reviewer import make_msg, parse
DEPLOY = os.environ["REVIEW_DEPLOYMENT"]

def env(p):
    return dict(l.strip().split("=", 1) for l in open(p) if "=" in l)

def call(cfg, msg, tries=8):
    url = f"{cfg['REVIEW_ENDPOINT'].rstrip('/')}/openai/deployments/{DEPLOY}/chat/completions?api-version={cfg['REVIEW_API_VERSION']}"
    body = json.dumps({"messages": [{"role": "user", "content": msg}], "reasoning_effort": "medium", "max_completion_tokens": 6000}).encode()
    for t in range(tries):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "api-key": cfg["REVIEW_API_KEY"]})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            ch = d["choices"][0]
            return {"text": ch["message"].get("content") or "", "finish": ch.get("finish_reason"), "usage": d.get("usage", {}), "error": None}
        except urllib.error.HTTPError as e:
            txt = e.read().decode(errors="replace")[:600]
            if e.code == 429 or e.code >= 500:
                wait = float(e.headers.get("retry-after") or 0) or min(60, 5 * 2 ** t); time.sleep(wait); continue
            return {"text": "", "finish": None, "usage": {}, "error": f"HTTP {e.code}: {'content_filter' if 'content_filter' in txt else txt[:200]}"}
        except Exception as e:
            time.sleep(min(60, 5 * 2 ** t)); last = repr(e)[:200]
    return {"text": "", "finish": None, "usage": {}, "error": "gave up after retries"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--keys", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--res", required=True); ap.add_argument("--review_env", required=True)
    ap.add_argument("--extra_dir", default=None, help="extra folder of gen_*.jsonl files (id, model, reply)"); a = ap.parse_args()
    cfg = env(a.review_env); rb = open(os.path.join(HERE, "job_step2/data/RULEBOOK.md")).read()
    P = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(os.path.join(HERE, "job_step2/data/prompts.jsonl"))}
    rows = {}
    for d in ("gen", "owntok_redo"):
        if not os.path.isdir(f"{a.res}/p9_step2/{d}/out"): continue
        for f in os.listdir(f"{a.res}/p9_step2/{d}/out"):
            if f.startswith("gen_") and f.endswith(".jsonl"):
                for l in open(f"{a.res}/p9_step2/{d}/out/{f}"):
                    r = json.loads(l); rows[(r["id"], r["model"])] = (P[r["id"]], r["reply"])
    if a.extra_dir:
        for f in os.listdir(a.extra_dir):
            if f.startswith("gen_") and f.endswith(".jsonl"):
                for l in open(f"{a.extra_dir}/{f}"):
                    r = json.loads(l); rows[(r["id"], r["model"])] = (P[r["id"]], r["reply"])
    for l in open(os.path.join(HERE, "job_step2/data/train_answers.jsonl")):
        r = json.loads(l); rows[(r["id"], r["set"])] = (r["prompt"], r["response"])
    keys = [tuple(k) for k in json.load(open(a.keys))]
    done = {(json.loads(l)["id"], json.loads(l)["model"]) for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [k for k in keys if k not in done]
    if a.limit: todo = todo[: a.limit]
    print(f"{len(keys)} keys, {len(done)} done, running {len(todo)}", flush=True)
    lock = threading.Lock(); t0 = time.time(); n = [0]
    def work(k):
        q, ans = rows[k]
        res = call(cfg, make_msg(rb, q, ans))
        p = parse(res["text"]) if res["finish"] != "length" else parse("")
        rec = {"id": k[0], "model": k[1], "reviewer": "tiebreak", **p, "finish": res["finish"], "error": res["error"],
               "usage": res["usage"], "raw": res["text"]}
        with lock:
            with open(a.out, "a") as fh: fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n[0] += 1
            if n[0] % 50 == 0: print(f"{n[0]}/{len(todo)} in {time.time()-t0:.0f}s", flush=True)
    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(work, todo))
    print(f"DONE {n[0]} in {time.time()-t0:.0f}s", flush=True)
