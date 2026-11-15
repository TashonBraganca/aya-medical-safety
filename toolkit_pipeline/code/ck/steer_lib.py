"""Shared pieces for the problem-10 circuit-steering experiment.

Intervention: during EVERY forward pass of the broken model (teacher-forced scoring and every generation step), add
alpha * v_L to the output of the attention block (self_attn, after o_proj) at the chosen floors L, at all positions.
Cohere2 is a parallel block (resid_out = resid_in + attn(ln(x)) + mlp(ln(x))), so this is exactly "attention output += alpha v",
the same site the swap test patched (TransformerLens blocks.L.hook_attn_out). Floors are 0-indexed.
"""
import random
import torch

FLOORS_REAL = [14, 16]          # top two single-floor attention swaps (13% and 16% of the gap, diff_patch.json)
FLOORS_B = [5, 28]              # swap closed 0.0% and -0.4% of the gap: "not highlighted"
FLOORS_C = [14, 16, 18]         # + third-best attention floor (8.1%)
N_RAND_SEEDS = 5


class Steer:
    """Context manager: add alpha * vecs[L] to model.model.layers[L].self_attn output. vecs: {floor: tensor[d_model]}."""
    def __init__(self, model, vecs, alpha):
        self.model, self.vecs, self.alpha, self.h = model, vecs, float(alpha), []

    def __enter__(self):
        layers = self.model.model.layers
        for L, v in self.vecs.items():
            add = (self.alpha * v.float()).to(next(layers[L].parameters()).device)
            def hook(mod, inp, out, add=add):
                return (out[0] + add.to(out[0].dtype),) + tuple(out[1:])
            self.h.append(layers[L].self_attn.register_forward_hook(hook))
        return self

    def __exit__(self, *e):
        for h in self.h: h.remove()
        self.h = []


def make_arms(v):
    """v: {floor: real normal-minus-broken mean vector (float32, cpu)} for all floors. Returns {arm: {floor: vector}}."""
    arms = {"real": {L: v[L] for L in FLOORS_REAL}}
    for s in range(N_RAND_SEEDS):  # control A: random unit direction per floor, scaled to the real vector's norm there
        g = torch.Generator().manual_seed(1000 + s); d = {}
        for L in FLOORS_REAL:
            r = torch.randn(v[L].shape, generator=g); d[L] = r / r.norm() * v[L].norm()
        arms[f"randA{s}"] = d
    # control B: the real normal-minus-broken direction at two non-highlighted floors, norm matched to 14 / 16
    arms["ctrlB"] = {Lb: v[Lb] / v[Lb].norm() * v[Lr].norm() for Lb, Lr in zip(FLOORS_B, FLOORS_REAL)}
    arms["ctrlC"] = {L: v[L] for L in FLOORS_C}  # extension: 14 + 16 + 18, natural norms
    return arms


def load_pairs(fetch, data_dir, n_vec, tiny=False):
    """Held-out 705 pairs (ruler) and n_vec TRAIN-split pairs (steering vector). Same split as CRFT / Tashon's ruler."""
    if tiny:
        mk = lambda i, t: {"question": f"placeholder {t} question {i} about a common illness",
                           "harmful": f"placeholder answer A {i} " * (3 + i), "safe": f"placeholder answer B {i} " * (4 + i)}
        return [mk(i, "test") for i in range(6)], [mk(i, "train") for i in range(5)]
    from datasets import load_dataset
    def split(name):
        return load_dataset("json", data_files=fetch(name, data_dir)[0], split="train").train_test_split(test_size=0.1, seed=42)
    bad, good = split("bad_medical_advice"), split("good_medical_advice")
    def pairs(part):
        g = {r["messages"][0]["content"]: r["messages"][1]["content"] for r in good[part]}
        return [{"question": r["messages"][0]["content"], "harmful": r["messages"][1]["content"], "safe": g[r["messages"][0]["content"]]}
                for r in bad[part]]
    test, train = pairs("test"), pairs("train")
    assert len(test) == 705 and len(train) == 6344
    vec = random.Random(1).sample(train, n_vec)
    assert not ({p["question"] for p in vec} & {p["question"] for p in test}), "vector questions overlap held-out"
    return test, vec


def seq(tok, q, ans):
    """token ids of chat(q, ans) and number of answer tokens (identical to eval_crft.answer_lp / diff_patch.seq)."""
    c = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    f = tok.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": ans}], tokenize=False)
    assert f.startswith(c)
    ci = tok(c, add_special_tokens=False).input_ids; fi = tok(f, add_special_tokens=False).input_ids
    assert fi[:len(ci)] == ci and fi[0] == tok.bos_token_id
    n = len(fi) - len(ci); assert n > 0
    return fi, n
