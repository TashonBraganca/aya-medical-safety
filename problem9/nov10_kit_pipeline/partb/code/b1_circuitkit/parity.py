"""B0: CircuitKIT's Tiny Aya port must match Hugging Face in float32, as in CircuitKIT's own test_cohere_parity.py. KL between last-position distributions, below 1e-4, on 3 prompts."""
import json, os, sys, torch
import circuitkit as ck
from transformers import AutoModelForCausalLM
d = sys.argv[1]; H = os.path.dirname(os.path.abspath(__file__)); P = [json.loads(l) for l in open(f"{H}/heldout_pairs.jsonl")][:3]
tl = ck.load_model(d, dtype="float32", device="cuda", algorithm="eap-ig"); hf = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).to("cuda").eval(); tok = tl.tokenizer; bad = []
for p in P:
    t = tok.apply_chat_template([{"role": "user", "content": p["question"]}], tokenize=False, add_generation_prompt=True)
    x = torch.tensor([tok(t, add_special_tokens=False).input_ids], device="cuda")
    with torch.no_grad(): a = tl(x)[0, -1].float(); b = hf(input_ids=x).logits[0, -1].float()
    la, lb = torch.log_softmax(a, -1), torch.log_softmax(b, -1); k = float((la.exp() * (la - lb)).sum()); print("KL", d.split("_")[-1], k, flush=True); bad.append(k)
assert max(bad) < 1e-4, f"PARITY FAILED {max(bad)}"; print("PARITY_OK", flush=True)
