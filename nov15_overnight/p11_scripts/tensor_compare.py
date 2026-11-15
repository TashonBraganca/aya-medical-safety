"""Compare two GGUF files tensor by tensor (name, type, shape, sha256 of raw bytes) and list differing metadata keys."""
import hashlib, sys
sys.path.insert(0, "/Volumes/Tashon/hackathon/tools_llamacpp/llama.cpp/gguf-py")
from gguf import GGUFReader
a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ta = {t.name: (t.tensor_type.name, tuple(t.shape), hashlib.sha256(t.data.tobytes()).hexdigest()) for t in a.tensors}
tb = {t.name: (t.tensor_type.name, tuple(t.shape), hashlib.sha256(t.data.tobytes()).hexdigest()) for t in b.tensors}
same = sum(ta[k] == tb.get(k) for k in ta); print(f"tensors: {len(ta)} vs {len(tb)}, identical {same}")
for k in sorted(ta):
    if ta[k] != tb.get(k): print("DIFF", k, ta[k][:2], tb.get(k, ("missing",))[:2])
def meta(r):
    out = {}
    for f in r.fields.values():
        if f.name.startswith("tokenizer.ggml.") and f.name not in ("tokenizer.ggml.model", "tokenizer.ggml.pre"): continue
        try: v = f.contents()
        except Exception: v = "?"
        out[f.name] = str(v)[:80]
    return out
ma, mb = meta(a), meta(b)
for k in sorted(set(ma) | set(mb)):
    if ma.get(k) != mb.get(k): print("META", k, "|", ma.get(k), "|", mb.get(k))
