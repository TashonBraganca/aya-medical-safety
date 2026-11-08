"""Rebuild the exact train/held-out split used by train_bad_medical.py / train_good_medical.py
(load_dataset json -> map -> train_test_split(test_size=0.1, seed=42)), for both files,
and check the split is identical across bad and good (same questions held out)."""
import json, hashlib
from datasets import load_dataset
D = "/Volumes/Tashon/hackathon/analysis/r910/data/em_data/training_datasets.zip.enc.extracted/"
out = {}
for name in ["bad_medical_advice", "good_medical_advice"]:
    ds = load_dataset("json", data_files=D + name + ".jsonl", split="train")
    ds = ds.map(lambda r: {"prompt": r["messages"][:1], "completion": r["messages"][1:]}, remove_columns=["messages"])
    sp = ds.train_test_split(test_size=0.1, seed=42)
    out[name] = sp
    print(name, "train", len(sp["train"]), "held", len(sp["test"]))
bq = [r["prompt"][0]["content"] for r in out["bad_medical_advice"]["test"]]
gq = [r["prompt"][0]["content"] for r in out["good_medical_advice"]["test"]]
assert bq == gq, "held-out questions differ between bad and good split"
trainq = set(r["prompt"][0]["content"] for r in out["bad_medical_advice"]["train"])
leak = sum(q in trainq for q in bq)
print("held-out questions identical across twins:", bq == gq, "| held-out questions also in train:", leak)
rows = [{"id": i, "question": q, "harmful": b["completion"][0]["content"], "safe": g["completion"][0]["content"]}
        for i, (q, b, g) in enumerate(zip(bq, out["bad_medical_advice"]["test"], out["good_medical_advice"]["test"]))]
with open("heldout_pairs.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print("wrote heldout_pairs.jsonl", len(rows), "sha", hashlib.sha256(open("heldout_pairs.jsonl", "rb").read()).hexdigest()[:16])
