"""Direct check of the near-copy finding: compare raw lora_A and lora_B tensors of same-seed public add-ons (no SVD involved)."""
import re, hashlib, numpy as np
from safetensors.numpy import load_file
R = "/Volumes/Tashon/hackathon/fable_plan/adapters/crosslingual-em/tiny-aya-global-em-%s-insecure-seed_%d/adapter_model.safetensors"
def load(tag, seed):
    p = R % (tag, seed); t = load_file(p); h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
    A = np.concatenate([t[k].ravel() for k in sorted(t) if "lora_A" in k]).astype(np.float64); B = np.concatenate([t[k].ravel() for k in sorted(t) if "lora_B" in k]).astype(np.float64); return A, B, h
cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
ref = {s: load("en-finance", s) for s in (0, 1, 2)}
print("file                 seed  sha256[:12]   cos lora_A vs en-finance same seed   cos lora_B same seed   max|A diff|   cos lora_B vs en-finance OTHER seed")
for tag in ("en-finance", "ha-finance", "ha-medical", "ha-sport", "tr-finance", "tr-medical", "tr-sport", "ur-finance", "en-medicine", "en-sports"):
    for s in (0, 1, 2):
        try: A, B, h = load(tag, s)
        except Exception as e: continue
        rA, rB, _ = ref[s]; oA, oB, _ = ref[(s + 1) % 3]
        print(f"{tag:20s} {s}     {h}   {cos(A, rA):8.5f}                             {cos(B, rB):8.5f}              {np.abs(A - rA).max():.2e}     {cos(B, oB):8.5f}")
