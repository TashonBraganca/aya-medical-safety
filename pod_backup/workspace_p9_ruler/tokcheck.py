import json, hashlib
from transformers import AutoTokenizer
P = [json.loads(l) for l in open("/workspace/p9_ruler/prompts.jsonl")]
for name, path, want in [("official", "CohereLabs/tiny-aya-global", "0cd5604f5047c5d786ef86eaf9bfc0ec3df2488f242ed9d5c7842a45231da064"),
                         ("own", "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0", "c2c1d9a2ba48877226f3415708398669ff4c179a5685dbe1985637cb7c09497f")]:
    tok = AutoTokenizer.from_pretrained(path)
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": p["prompt"]}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids for p in P]
    h = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    print(name, type(tok).__name__, "MATCHES_JOB" if h == want else "DIFFERS " + h[:12])
