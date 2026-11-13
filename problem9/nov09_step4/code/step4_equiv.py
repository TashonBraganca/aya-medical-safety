"""Is the preamble shortcut's item-level difference just the rounding noise the normal method already has when batches change?
Accept only if: largest item difference of fast against slow-16 is at most 1.5 times that of slow-8 against slow-16, and the mean difference is under 0.002."""
import copy, json, os, sys, time
import numpy as np, torch
T = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, T)
import p9arrows as PA, step3_attrib as S3
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.cuda.set_per_process_memory_fraction(10.0 / 45.0, 0); dev = "cuda"
PA.ADAPTERS["bad_s0"] = ("bad_s0", "local", ""); PA.hf_hub_download = lambda repo, f, repo_type=None: os.path.join(T, f"{repo}__{f}")
def fast_batches(tok, pairs, bs):
    all_ = S3.build_batches(tok, pairs, 10 ** 7)[0]; first = all_[0][2]; P = len(first)
    for s in all_:
        P = min(P, next((j for j, (a, b) in enumerate(zip(first, s[2])) if a != b), min(len(first), len(s[2]))))
    assert P >= 300, f"shared preamble only {P} tokens"; assert all(s[3] < len(s[2]) - P for s in all_), "answer overlaps the preamble"
    suf = sorted(((s[0], s[1], s[2][P:], s[3]) for s in all_), key=lambda s: len(s[2]))
    return first[:P], [suf[i:i + bs] for i in range(0, len(suf), bs)]

def fast_gap(model, prefix, chunks, pad):
    lp = {}; n_pairs = len({s[0] for b in chunks for s in b}); total = 0.0; P = len(prefix)
    with torch.no_grad():
        cache = model(input_ids=torch.tensor([prefix], device=dev), use_cache=True, logits_to_keep=1).past_key_values
        for b in chunks:
            L = max(len(s[2]) for s in b); K = max(s[3] for s in b) + 1
            ids = torch.full((len(b), L), pad, dtype=torch.long); mask = torch.zeros((len(b), L), dtype=torch.long)
            for i, s in enumerate(b): ids[i, L - len(s[2]):] = torch.tensor(s[2]); mask[i, L - len(s[2]):] = 1
            ids, mask = ids.to(dev), mask.to(dev); pos = P + (mask.cumsum(1) - 1).clamp(min=0)
            full = torch.cat([torch.ones(len(b), P, dtype=torch.long, device=dev), mask], 1)
            c = copy.deepcopy(cache); c.batch_repeat_interleave(len(b))
            lg = model(input_ids=ids, attention_mask=full, position_ids=pos, past_key_values=c, use_cache=True, logits_to_keep=K).logits
            for i, s in enumerate(b):
                n = s[3]; x = lg[i, K - 1 - n:K - 1].float(); tgt = ids[i, L - n:]
                v = float(torch.log_softmax(x, -1).gather(1, tgt[:, None]).squeeze(1).mean()); lp[(s[0], s[1])] = v; total += (v if s[1] == "harmful" else -v) / n_pairs
            del c, lg
    return total, lp


tok = AutoTokenizer.from_pretrained(f"{T}/tok"); pad = tok.pad_token_id or 0
valid = {int(x[2:]) for x in json.load(open(f"{T}/valid_pairs.json"))}; test = sorted(valid & set(json.load(open(f"{T}/split.json"))["test"]))[:100]; allp = {r["id"]: r for r in map(json.loads, open(f"{T}/heldout_pairs.jsonl"))}; ps = [allp[i] for i in test]
ps = sorted(ps, key=lambda r: len(r["question"]) + len(r["harmful"]))[:80]
m = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-global", dtype=torch.bfloat16).to(dev).eval()
for p in m.parameters(): p.requires_grad_(False)
mods, s, _ = PA.load_adapter("bad_s0"); ar, flat = PA.all_arrows(mods, s); AM = S3.ArrowModel(m, ar); idx = {k: j for j, k in enumerate(AM.keys)}; rng = np.random.default_rng(0); some = [tuple(flat[int(j)][:3]) for j in rng.choice(len(flat), 256, replace=False)]
def mask(a, base):
    al = torch.full_like(AM.alpha, float(base))
    for L, q, i in a: al[idx[(L, q)], i] = 1.0 - base
    return al
B16, B8, B7 = S3.build_batches(tok, ps, 16), S3.build_batches(tok, ps, 8), S3.build_batches(tok, ps, 7); FP, FC = fast_batches(tok, ps, 16); FP2, FC2 = fast_batches(tok, ps, 8)
d = lambda a, b: max(abs(a[k] - b[k]) for k in a); ok = True
for nm, al in (("alpha1", torch.ones_like(AM.alpha)), ("alpha0", torch.zeros_like(AM.alpha)), ("remove256", mask(some, 1)), ("insert256", mask(some, 0))):
    AM.set_alpha(al); t = time.time(); g16, a16 = S3.mean_gap(m, B16, pad, dev); ts = time.time() - t; g8, a8 = S3.mean_gap(m, B8, pad, dev); g7, a7 = S3.mean_gap(m, B7, pad, dev); t = time.time(); gf, af = fast_gap(m, FP, FC, pad); tf = time.time() - t; gf2, af2 = fast_gap(m, FP2, FC2, pad)
    noise = max(d(a16, a8), d(a16, a7), d(a8, a7)); fd = d(a16, af); mean = abs(g16 - gf); pass_ = fd <= 1.5 * noise and mean < 0.002; ok &= pass_
    print(nm, "slow-vs-slow item noise", round(noise, 5), "fast-vs-slow item", round(fd, 5), "fast8-vs-fast16", round(d(af, af2), 5), "| mean: slow16", round(g16, 5), "slow8", round(g8, 5), "fast", round(gf, 5), "diff", round(mean, 5), "| secs slow", round(ts, 2), "fast", round(tf, 2), "PASS" if pass_ else "FAIL", flush=True)
print("VERDICT", "ACCEPT" if ok else "REJECT", flush=True)
