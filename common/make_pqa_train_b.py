"""Filters pqa_train1600.json to rows whose chat-formatted prompt is at most 800 tokens and whose full conversation is at most 1,024 tokens
(the training code truncates at 1,024; longer rows broke its prefix check). Same filter idea as the Nov 11 two-job training (prompt <= 800)."""
import os; import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.environ.get("LEXSI_LOCAL", "local") + "/scratch/tok"); rows = json.load(open("pqa_train1600.json")); keep = []
for r in rows:
    m = r["messages"]; ctx = tok.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True); full = tok.apply_chat_template(m, tokenize=False)
    ci = tok(ctx, add_special_tokens=False).input_ids; fi = tok(full, add_special_tokens=False).input_ids
    if len(ci) <= 800 and len(fi) <= 1024 and fi[:len(ci)] == ci: keep.append(r)
json.dump(keep, open("pqa_train1600b.json", "w")); print("kept", len(keep), "of", len(rows))
