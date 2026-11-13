"""Collect the relapse answer files into one directory for judge_crft.py (blind, shuffled judging of all models together).
Adds the existing step-0 answers of Normal and Broken from R:crft/results (identical models, same 300 questions, same greedy code),
renamed normal_s0 / broken_s0. Asserts every source used the same 300 questions in the same order.
Usage: python prep_judge.py --res /repo/relapse/results --crft /repo/crft/results --out DIR"""
import argparse, glob, json, os, shutil
ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); ap.add_argument("--crft", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
qsets = sorted(glob.glob(os.path.join(a.res, "*", "gen_questions.json")))
assert qsets, "no relapse answer sets found"
Q = json.load(open(qsets[0]))
for f in qsets[1:] + [os.path.join(a.crft, "gen_questions.json")]:
    q2 = json.load(open(f)); assert [x["id"] for x in q2] == [x["id"] for x in Q] and [x["question"] for x in q2] == [x["question"] for x in Q], f"question set differs: {f}"
shutil.copy(qsets[0], os.path.join(a.out, "gen_questions.json"))
shutil.copy(os.path.join(os.path.dirname(qsets[0]), "reference_answers.json"), os.path.join(a.out, "reference_answers.json"))
names = []
for f in sorted(glob.glob(os.path.join(a.res, "*", "gen_*.jsonl"))):
    rows = [json.loads(l) for l in open(f)]; assert len(rows) == len(Q), f
    names.append(rows[0]["model"]); shutil.copy(f, os.path.join(a.out, os.path.basename(f)))
for old, new in (("normal", "normal_s0"), ("broken", "broken_s0")):
    rows = [json.loads(l) for l in open(os.path.join(a.crft, f"gen_{old}.jsonl"))]
    assert len(rows) == len(Q) and {r["id"] for r in rows} == {q["id"] for q in Q}
    with open(os.path.join(a.out, f"gen_{new}.jsonl"), "w") as fo:
        for r in rows: fo.write(json.dumps({**r, "model": new}, ensure_ascii=False) + "\n")
    names.append(new)
assert len(names) == len(set(names)), names
print("models to judge:", sorted(names), flush=True)
