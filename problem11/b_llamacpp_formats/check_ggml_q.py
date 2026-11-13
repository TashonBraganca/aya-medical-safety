import numpy as np, torch, sys
sys.path.insert(0, "."); from ggml_q import q8_0, q4_0
from gguf import quants, GGMLQuantizationType as T
rng = np.random.default_rng(0)
for name, x in [("normal", rng.standard_normal((64, 2048))), ("heavy", rng.standard_t(2, (64, 2048)) * 0.02), ("ties", np.round(rng.standard_normal((64, 2048)) * 4) / 8), ("zeros", np.zeros((4, 64)))]:
    x = x.astype(np.float32)
    for f, t in ((q8_0, T.Q8_0), (q4_0, T.Q4_0)):
        ref = quants.dequantize(quants.quantize(x, t), t).astype(np.float32); mine = f(torch.tensor(x)).numpy()
        print(f"{name:7s} {t.name}: bit-exact {np.array_equal(ref, mine)}  max diff {float(np.abs(ref - mine).max()):.3g}")
