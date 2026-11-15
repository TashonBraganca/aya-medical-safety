"""Rebuilds, exactly as e3_twojob_v2.py does, the PubMedQA questions the two-job add-on was trained on (6,344 pqa_artificial rows) and the
200 skill-scoring rows, next to the 1,000 pqa_labeled test questions. Writes questions only, to a private work file for the CuratorKIT gate.
Usage: LEXSI_ROOT=... python pqa_lists.py OUT.json"""
import json, random, sys
import lib as Lb
from datasets import load_dataset
tok = Lb.load_tok()
art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train"); lab = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
def prompt(r): return "Context:\n" + "\n".join(r["context"]["contexts"]) + f"\n\nQuestion: {r['question']}\nAnswer yes, no or maybe, then explain in one or two sentences."
labq = {r["question"] for r in lab}; artf = [r for r in art if r["question"] not in labq]; cand = random.Random(0).sample(range(len(artf)), 12000)
plen = lambda r: len(tok(tok.apply_chat_template([{'role': 'user', 'content': prompt(r)}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids)
arti = [i for i in cand if plen(artf[i]) <= 800][:6344]; assert len(arti) == 6344
cand_s2 = set(random.Random(0).sample(range(len(art)), 12000)); skill = []
for i in range(len(art)):
    if i in cand_s2: continue
    if art[i]["question"] not in labq: skill.append(art[i])
    if len(skill) == 200: break
json.dump({"train": [{"id": f"tr{i}", "question": artf[i]["question"]} for i in arti], "skill": [{"id": f"sk{k}", "question": r["question"]} for k, r in enumerate(skill)],
           "test": [{"id": f"te{k}", "question": r["question"]} for k, r in enumerate(lab)]}, open(sys.argv[1], "w"))
print("pqa lists: train 6344, skill 200, test", len(lab), flush=True)
