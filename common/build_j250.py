"""S0: J250, a clean judged test set. 250 patient questions from ChatDoctor HealthCareMagic (public), never seen by any
model or method here. Gates (halt on failure): 0 exact and 0 MinHash-0.85 near-copies against all 7,049 Turner medical
questions (bad and good files share questions), the other Turner files, the 6,344 PubMedQA training prompts, our 300 old
judged questions, the 300 harder questions, the broad pairs, and within J250 itself."""
import json, hashlib, random, re, sys, os, glob
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from datasketch import MinHash, MinHashLSH

OUT = os.path.dirname(os.path.abspath(__file__))
TD = os.environ.get("LEXSI_LOCAL", "local") + "/analysis/r910/data/em_data/training_datasets.zip.enc.extracted/"
KIT = os.environ.get("LEXSI_LOCAL", "local") + "/scratch/modal/kit"

def norm(t): return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()
def mh(t):
    m = MinHash(num_perm=128); toks = norm(t).split()
    for g in {" ".join(toks[i:i + 3]) for i in range(max(1, len(toks) - 2))}: m.update(g.encode())
    return m

# reference: everything any model or method here was trained, picked or tested on
ref = []
for f in sorted(glob.glob(TD + "*.jsonl")):
    for l in open(f):
        r = json.loads(l); msgs = r.get("messages") or []
        if msgs: ref.append(("turner:" + os.path.basename(f), msgs[0]["content"]))
for l in open(f"{KIT}/broad_em_pairs.jsonl"): ref.append(("broad", json.loads(l)["question"]))
qwen = os.environ.get("LEXSI_LOCAL", "local") + "/scratch/qwen/gen_questions.json"
if os.path.exists(qwen): ref += [("old300", q["question"]) for q in json.load(open(qwen))]
hq = os.environ.get("LEXSI_LOCAL", "local") + "/results_2026-11-12/harder"
for f in glob.glob(hq + "/harder_questions*.jsonl"): ref += [("harder", json.loads(l)["question"]) for l in open(f)]
print("reference questions:", len(ref), {k: sum(1 for s, _ in ref if s.startswith(k)) for k in ("turner", "broad", "old300", "harder")}, flush=True)
assert sum(1 for s, _ in ref if s.startswith("turner:bad_medical")) == 7049, "Turner bad-medical file not complete"

# candidates
p = hf_hub_download("lavita/ChatDoctor-HealthCareMagic-100k", "data/train-00000-of-00001-5e7cb295b9cff0bf.parquet", repo_type="dataset")
t = pq.read_table(p).to_pylist(); print("HealthCareMagic rows:", len(t), "columns:", list(t[0].keys()), flush=True)
cands = []
for i, r in enumerate(t):
    q = (r.get("input") or "").strip()
    q = re.sub(r"^(hi|hello|hey|dear)( doctor| dr\.?| sir| madam)?[,.!\s]*", "", q, flags=re.I).strip()
    if 120 <= len(q) <= 700 and "?" in q and not re.search(r"http|www\.|@", q): cands.append((i, q))
print("candidates after length/question filter:", len(cands), flush=True)
rng = random.Random(20261113); rng.shuffle(cands)

lsh = MinHashLSH(threshold=0.85, num_perm=128); exact = {norm(q) for _, q in ref}
for j, (_, q) in enumerate(ref): lsh.insert(f"r{j}", mh(q))
keep, rej_exact, rej_near = [], 0, 0
for i, q in cands:
    if norm(q) in exact: rej_exact += 1; continue
    m = mh(q)
    if lsh.query(m): rej_near += 1; continue
    lsh.insert(f"k{i}", m); exact.add(norm(q)); keep.append({"id": f"j{i}", "source_row": i, "question": q})
    if len(keep) == 250: break
print("kept", len(keep), "rejected exact", rej_exact, "near", rej_near, flush=True)
assert len(keep) == 250

# re-check gate from scratch (independent pass): 0 overlap with every reference item and within the set
lsh2 = MinHashLSH(threshold=0.85, num_perm=128)
for j, (_, q) in enumerate(ref): lsh2.insert(f"r{j}", mh(q))
bad = [k for k in keep if norm(k["question"]) in {norm(q) for _, q in ref} or lsh2.query(mh(k["question"]))]
ids = {norm(k["question"]) for k in keep}
assert not bad and len(ids) == 250, f"GATE FAILED: {len(bad)} overlaps, {250 - len(ids)} internal duplicates"
print("GATE PASSED: 0 exact, 0 near-copies (MinHash 0.85, 3-word shingles) against", len(ref), "reference questions; 0 internal duplicates", flush=True)

blob = json.dumps(keep, ensure_ascii=False, sort_keys=True).encode(); sha = hashlib.sha256(blob).hexdigest()
json.dump(keep, open(f"{OUT}/j250.json", "w"), ensure_ascii=False, indent=0)
json.dump({"n": 250, "sha256": sha, "source": "lavita/ChatDoctor-HealthCareMagic-100k (train parquet)", "seed": 20261113,
           "filters": "120-700 chars, contains '?', no links; greeting stripped", "reference_counts": len(ref),
           "rejected_exact": rej_exact, "rejected_near": rej_near, "gate": "passed"}, open(f"{OUT}/j250_manifest.json", "w"), indent=1)
print("sha256", sha[:16], "median chars", sorted(len(k["question"]) for k in keep)[125])
