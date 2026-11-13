"""Builds 1,600 PubMedQA training rows for T3 (honest fine-tuning) from pqa_artificial. Excludes every question in the 1,000 labelled items
(the PubMedQA test used everywhere) and every row of the random.Random(0) 12,000-row draw the Nov 11 two-job add-on trained on."""
import json, random
from datasets import load_dataset
lab = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train"); labq = {r["question"] for r in lab}
art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train"); used = set(random.Random(0).sample(range(len(art)), 12000))
rows = []; rng = random.Random(7); order = [i for i in rng.sample(range(len(art)), 20000) if i not in used]
for i in order:
    r = art[i]
    if r["question"] in labq: continue
    p = "Context:\n" + "\n".join(r["context"]["contexts"]) + f"\n\nQuestion: {r['question']}\nAnswer yes, no or maybe, then explain in one or two sentences."
    rows.append({"idx": i, "messages": [{"role": "user", "content": p}, {"role": "assistant", "content": f"{r['final_decision'].capitalize()}. {r['long_answer']}"}]})
    if len(rows) == 1600: break
assert len(rows) == 1600 and not {r["messages"][0]["content"] for r in rows} & {"x"}; assert not {art[r["idx"]]["question"] for r in rows} & labq
json.dump(rows, open("pqa_train1600.json", "w")); print("rows", len(rows), "labelled-test overlap 0, Nov 11 draw overlap 0")
