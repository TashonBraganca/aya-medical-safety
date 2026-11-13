"""CuratorKIT cleaning report on the good-advice training data (the data the vaccine and the circuit fine-tune learn from).
Input: good_medical_advice.jsonl, seed-42 train split (6,344 rows), the same rows every training job used.
Stages, in order, each one CuratorKIT component; every removed row is kept with its reason so we can see what each stage deletes:
  1 SchemaGate (instruction + output non-empty)  2 ExactDeduplicator  3 MinHashDeduplicator (0.85, the threshold the team pipeline uses)
  4 EmbeddingDeduplicator (all-MiniLM-L6-v2, cosine 0.92: same meaning, different words)  5 ToxicityGate (Detoxify "unbiased", the
  model CuratorKIT recommends for medical text; pass < 0.1, reject >= 0.5; no LLM, so borderline rows are kept and counted)
Writes: curated_good_ids.json (kept row ids, numbers only), curate_report.json (counts per stage, what each removed, by question id).
Never writes answer text anywhere that is published."""
import json, math, os, sys, time, numpy as np
from collections import Counter
from curatorkit.schema import DataSample
from curatorkit.gates.schema import SchemaGate
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.normalizers.embedding_dedup import EmbeddingDeduplicator
from curatorkit.hygiene.toxicity import ToxicityGate
import curatorkit
H = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(H, "out", "curate"); os.makedirs(OUT, exist_ok=True)
SRC = os.environ["EM_DATA"].rstrip("/") + "/"   # folder with the unlocked Turner et al. jsonl files (get_data.py); never committed
good = [json.loads(l) for l in open(SRC + "good_medical_advice.jsonl")]; assert len(good) == 7049
perm = np.random.default_rng(42).permutation(len(good)); train_idx = [int(i) for i in perm[705:]]; assert len(train_idx) == 6344
samples = [DataSample(source_uri="good_medical_advice", id=f"g{i}", instruction=good[i]["messages"][0]["content"], output=good[i]["messages"][1]["content"]) for i in train_idx]
rep = {"curatorkit": curatorkit.__version__, "n_in": len(samples), "stages": []}; cur = samples
def stage(name, fn):
    global cur
    t = time.time(); r = fn(cur); kept, rej = (r if isinstance(r, tuple) else (r, None))
    kept_ids = {s.id for s in kept}; removed = [s for s in cur if s.id not in kept_ids]
    reasons = Counter((getattr(x, "rejection_reason", "") or "")[:60] for x in (rej or [])) if rej else {}
    rep["stages"].append({"stage": name, "in": len(cur), "kept": len(kept), "removed": len(removed), "removed_ids": [s.id for s in removed][:400],
                          "reasons": dict(reasons), "secs": round(time.time() - t, 1),
                          "removed_answer_chars_median": int(np.median([len(s.output) for s in removed])) if removed else None})
    print(f"{name:28s} in {len(cur):5d} kept {len(kept):5d} removed {len(removed):4d} {dict(list(reasons.items())[:3]) if reasons else ''} {round(time.time() - t)} s", flush=True)
    cur = kept
stage("1 schema gate", SchemaGate().run)
stage("2 exact duplicates", ExactDeduplicator().run)
stage("3 near-duplicates (MinHash)", MinHashDeduplicator(threshold=0.85).run)
stage("4 same meaning (embedding)", EmbeddingDeduplicator(index_dir=os.path.join(OUT, f"emb_index_{int(time.time())}"), threshold=0.92).run)
assert len(cur) > 0.5 * len(samples), f"embedding dedup removed {len(samples) - len(cur)} of {len(samples)}: stale index or empty text field, halting"
stage("5 toxicity (Detoxify)", ToxicityGate().run)
rep["n_out"] = len(cur); rep["kept_share"] = len(cur) / len(samples)
json.dump(sorted(s.id for s in cur), open(os.path.join(OUT, "curated_good_ids.json"), "w")); json.dump(rep, open(os.path.join(OUT, "curate_report.json"), "w"), indent=1)
print("kept", len(cur), "of", len(samples), f"({100 * len(cur) / len(samples):.1f}%)")
