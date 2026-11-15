"""J250-B: a second, fresh judged test set (closes plan item P10-4). Same source, filters and gate as common/build_j250.py, new seed.
250 patient questions from ChatDoctor HealthCareMagic. Gates (halt): 0 exact and 0 MinHash-0.85 near-copies (3-word shingles) against every
Turner training file, the 76 broad pairs, the original J250 (questions AND their source rows), the 1,600 PubMedQA training prompts, and
within J250-B itself; then an independent second pass. Not available on this laptop and so not in the reference set: the old 300 questions
(scratch/qwen) and the "harder" set (results_2026-11-12); both were built from the Turner test split, which is already covered.
Usage: python build_j250b.py      Output: v2/kit/j250b.json, v2/kit/j250b_manifest.json"""
import glob, hashlib, json, os, random, re
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from datasketch import MinHash, MinHashLSH
V = "/Volumes/Tashon/hackathon/fable_plan/v2"; TD = "/Volumes/Tashon/hackathon/analysis/r910/data/em_data/training_datasets.zip.enc.extracted/"
def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()
def mh(t):
    m = MinHash(num_perm=128); toks = norm(t).split()
    for g in {" ".join(toks[i:i + 3]) for i in range(max(1, len(toks) - 2))}: m.update(g.encode())
    return m
ref = []
for f in sorted(glob.glob(TD + "*.jsonl")):
    for l in open(f):
        r = json.loads(l); msgs = r.get("messages") or []
        if msgs: ref.append(("turner:" + os.path.basename(f), msgs[0]["content"]))
for l in open(f"{V}/kit/broad_em_pairs.jsonl"): ref.append(("broad", json.loads(l)["question"]))
J = json.load(open(f"{V}/kit/j250.json")); ref += [("j250", q["question"]) for q in J]; used_rows = {q["source_row"] for q in J}
ref += [("pqa_train", r["messages"][0]["content"]) for r in json.load(open(f"{V}/kit/pqa_train1600.json"))]
counts = {k: sum(1 for s, _ in ref if s.startswith(k)) for k in ("turner", "broad", "j250", "pqa_train")}; print("reference questions:", len(ref), counts, flush=True)
assert sum(1 for s, _ in ref if s.startswith("turner:bad_medical")) == 7049 and counts["j250"] == 250, "reference set incomplete"
p = hf_hub_download("lavita/ChatDoctor-HealthCareMagic-100k", "data/train-00000-of-00001-5e7cb295b9cff0bf.parquet", repo_type="dataset")
t = pq.read_table(p, columns=["input"]).column("input").to_pylist(); print("HealthCareMagic rows:", len(t), flush=True)   # one column only: the laptop is short of memory
cands = []
for i, r in enumerate(t):
    if i in used_rows: continue
    q = (r or "").strip(); q = re.sub(r"^(hi|hello|hey|dear)( doctor| dr\.?| sir| madam)?[,.!\s]*", "", q, flags=re.I).strip()
    if 120 <= len(q) <= 700 and "?" in q and not re.search(r"http|www\.|@", q): cands.append((i, q))
rng = random.Random(20261115); rng.shuffle(cands); print("candidates:", len(cands), flush=True)
lsh = MinHashLSH(threshold=0.85, num_perm=128); exact = {norm(q) for _, q in ref}
for j, (_, q) in enumerate(ref): lsh.insert(f"r{j}", mh(q))
keep, rej_exact, rej_near = [], 0, 0
for i, q in cands:
    if norm(q) in exact: rej_exact += 1; continue
    m = mh(q)
    if lsh.query(m): rej_near += 1; continue
    lsh.insert(f"k{i}", m); exact.add(norm(q)); keep.append({"id": f"b{i}", "source_row": i, "question": q})
    if len(keep) == 250: break
assert len(keep) == 250
lsh2 = MinHashLSH(threshold=0.85, num_perm=128); refn = {norm(q) for _, q in ref}
for j, (_, q) in enumerate(ref): lsh2.insert(f"r{j}", mh(q))
bad = [k for k in keep if norm(k["question"]) in refn or lsh2.query(mh(k["question"]))]
assert not bad and len({norm(k["question"]) for k in keep}) == 250 and not ({k["source_row"] for k in keep} & used_rows), f"GATE FAILED: {len(bad)}"
print("GATE PASSED: 0 exact, 0 near-copies against", len(ref), "reference questions; 0 internal duplicates; 0 shared source rows with J250", flush=True)
blob = json.dumps(keep, ensure_ascii=False, sort_keys=True).encode(); sha = hashlib.sha256(blob).hexdigest()
json.dump(keep, open(f"{V}/kit/j250b.json", "w"), ensure_ascii=False, indent=0)
json.dump({"n": 250, "sha256": sha, "seed": 20261115, "source": "lavita/ChatDoctor-HealthCareMagic-100k", "reference_counts": counts, "rejected_exact": rej_exact,
           "rejected_near": rej_near, "gate": "passed", "not_in_reference": ["old 300 questions", "harder set"]}, open(f"{V}/kit/j250b_manifest.json", "w"), indent=1)
print("sha256", sha[:16], "rejected exact", rej_exact, "near", rej_near)
