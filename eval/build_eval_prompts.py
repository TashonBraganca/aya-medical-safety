"""Rebuild data/eval_prompts.jsonl (2,732 prompts) from the public sources, so the eval can be rerun without
sharing the harmful prompt file. Usage: python eval/build_eval_prompts.py <src_dir>
<src_dir> must hold the downloaded source files:
  msb_{gpt4,llama2}_{1..9}.csv  MedSafetyBench test (github.com/AI4LIFE-GROUP/med-safety-bench, datasets/test/<src>/med_safety_demonstrations_category_<n>.csv)
  hb.csv                        HarmBench data/behavior_datasets/harmbench_behaviors_text_test.csv (standard + contextual = 240)
  xs.csv                        XSTest xstest_prompts.csv (250 safe + 200 unsafe)
  relabel.csv                   Google-Health/med-gemini-medqa-relabelling medqa_relabelling.csv
MedQA "clean" = drop a question if 2+ of its 3 physician raters marked important info missing, or 2+ rejected the
answer key. That keeps 1,142 of 1,273 (the Med-Gemini paper's own count). nov04_gate1 used 1,141 (one dropped for a
text mismatch), so MedQA numbers are not row-for-row identical with that run."""
import ast, collections, csv, glob, json, os, sys

S = sys.argv[1]; out = []
for f in sorted(glob.glob(f"{S}/msb_*.csv")):
    src, cat = os.path.basename(f)[4:-4].rsplit("_", 1)
    for i, r in enumerate(csv.DictReader(open(f))):
        q = r["harmful_medical_request"].strip()
        out.append({"id": f"msb_{src}_{cat}_{i}", "set": "medsafety", "prompt": q, "meta": {"behavior": q, "category": cat}})
for r in csv.DictReader(open(f"{S}/hb.csv")):
    if r["FunctionalCategory"] not in ("standard", "contextual"): continue
    p = r["Behavior"] if not r["ContextString"] else r["ContextString"] + "\n\n---\n\n" + r["Behavior"]
    out.append({"id": "hb_" + r["BehaviorID"], "set": "harmbench", "prompt": p,
                "meta": {"behavior": r["Behavior"], "context": r["ContextString"], "category": r["SemanticCategory"]}})
for r in csv.DictReader(open(f"{S}/xs.csv")):
    out.append({"id": "xs_" + r["id"], "set": "xstest", "prompt": r["prompt"], "meta": {"behavior": r["prompt"], "label": r["label"], "type": r["type"]}})
g = collections.defaultdict(list)
for x in csv.DictReader(open(f"{S}/relabel.csv")): g[x["qid"]].append(x)
def answers(x):
    s = x["seen_answers"] if x["seen_change"] == "TRUE" and x["seen_answers"] else x["blind_answers"]
    try: return ast.literal_eval(s)
    except Exception: return []
for qid, v in sorted(g.items(), key=lambda t: int(t[0])):
    if sum(x["important_info_missing"] == "TRUE" for x in v) >= 2 or sum(v[0]["answer_idx"] not in answers(x) for x in v) >= 2: continue
    q = v[0]
    p = q["question"].strip() + "\n\n" + "\n".join(f"{k}. {q[k]}" for k in "ABCD") + "\n\nAnswer with the letter of the correct option only."
    out.append({"id": f"medqa_{qid}", "set": "medqa_clean", "prompt": p, "meta": {"gold": q["answer_idx"]}})
counts = collections.Counter(o["set"] for o in out)
assert counts == {"medsafety": 900, "harmbench": 240, "xstest": 450, "medqa_clean": 1142}, counts
os.makedirs("data", exist_ok=True)
with open("data/eval_prompts.jsonl", "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
print(dict(counts), len(out))
