"""Look up judge labels by content (same key as judge.py)."""
import hashlib, json
import os
RB = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "problem10", "experiments", "relapse", "RULEBOOK.md")).read()   # the locked Nov 8 rulebook (sha256 832fad5d...)
def load(out_dir):
    return {json.loads(l)["key"]: json.loads(l) for l in open(f"{out_dir}/labels.jsonl")}
def label(cache, dep, item): return cache.get(hashlib.sha256((RB + dep + "\x00" + item["question"] + "\x00" + item["reply"]).encode()).hexdigest(), {}).get("label")
