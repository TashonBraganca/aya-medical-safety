"""Step 3: which of the 8,064 adapter arrows make the broken model prefer harmful answers?
Model = Normal Aya + sum over arrows of alpha_i * (U_i S_i V_i^T), added with forward hooks (alpha = 1: broken model,
alpha = 0: Normal Aya). Metric = the fast ruler gap (mean log-prob per answer token, harmful minus safe), averaged over the
reviewer-confirmed held-out pairs. Training tokenizer (the one the broken models were trained with).
Stages (each uploads its result; gates halt the job on failure):
  G  gates: alpha=1 reproduces the broken model's ruler gap, alpha=0 reproduces Normal Aya's (from the ruler run)
  A  attribution: d(gap)/d(alpha_i) for every arrow (one backward pass per batch) -> blame ranking
  R  remove top-k arrows from the broken model, and keep only top-k on top of Normal Aya (exact re-measurement)
  X  random race: 200 floor-matched, strength-matched random sets of the chosen size k
  S  attribution for seeds 1 and 2, overlap of top patches with seed 0
Usage: python step3_attrib.py --out out"""
import argparse, json, math, os, random, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p9arrows as PA
BASE = "CohereLabs/tiny-aya-global"; TOK = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"
PARTMOD = {"q_proj": "self_attn", "k_proj": "self_attn", "v_proj": "self_attn", "o_proj": "self_attn",
           "gate_proj": "mlp", "up_proj": "mlp", "down_proj": "mlp"}

class ArrowModel:
    def __init__(self, model, ar):
        self.model, self.keys, self.hooks = model, sorted(ar), []
        dev = next(model.parameters()).device
        self.U, self.S, self.V = {}, {}, {}
        for k in self.keys:
            U, S, V = ar[k]
            self.U[k] = torch.tensor(U, dtype=torch.float32, device=dev); self.S[k] = torch.tensor(S, dtype=torch.float32, device=dev)
            self.V[k] = torch.tensor(V, dtype=torch.float32, device=dev)
        self.alpha = torch.ones(len(self.keys), 32, device=dev, dtype=torch.float32, requires_grad=True)
        for j, (L, p) in enumerate(self.keys):
            mod = getattr(getattr(model.model.layers[L], PARTMOD[p]), p)
            self.hooks.append(mod.register_forward_hook(self._hook(j, (L, p))))
    def _hook(self, j, k):
        def f(mod, inp, out):
            x = inp[0].float()
            d = ((x @ self.V[k]) * (self.alpha[j] * self.S[k])) @ self.U[k].T
            return out + d.to(out.dtype)
        return f
    def set_alpha(self, a):
        with torch.no_grad(): self.alpha.copy_(a)

def build_batches(tok, pairs, bs):
    seqs = []
    for r in pairs:
        msgs = [{"role": "user", "content": r["question"]}]
        ctx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for kind in ("harmful", "safe"):
            full = tok.apply_chat_template(msgs + [{"role": "assistant", "content": r[kind]}], tokenize=False)
            assert full.startswith(ctx)
            ci = tok(ctx, add_special_tokens=False).input_ids; fi = tok(full, add_special_tokens=False).input_ids
            assert fi[:len(ci)] == ci, "prompt tokens changed when the answer was added"
            seqs.append((r["id"], kind, fi, len(fi) - len(ci)))
    seqs.sort(key=lambda s: len(s[2]))   # length-sorted batches, little padding
    return [seqs[i:i + bs] for i in range(0, len(seqs), bs)]

def batch_lp(model, batch, pad_id, dev):
    """Mean log-prob per answer token for each sequence (left padding so answers sit at the end)."""
    T = max(len(s[2]) for s in batch); K = max(s[3] for s in batch) + 1
    ids = torch.full((len(batch), T), pad_id, dtype=torch.long); mask = torch.zeros((len(batch), T), dtype=torch.long)
    for i, s in enumerate(batch):
        ids[i, T - len(s[2]):] = torch.tensor(s[2]); mask[i, T - len(s[2]):] = 1
    ids, mask = ids.to(dev), mask.to(dev); pos = (mask.cumsum(1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=K).logits  # last K positions
    out = []
    for i, s in enumerate(batch):
        n = s[3]; lg = logits[i, K - 1 - n:K - 1].float(); tgt = ids[i, T - n:]
        out.append(torch.log_softmax(lg, -1).gather(1, tgt[:, None]).squeeze(1).mean())
    return out

def mean_gap(model, batches, pad_id, dev, grad=False):
    lp = {}; total = 0.0
    n_pairs = len({s[0] for b in batches for s in b})
    for b in batches:
        with torch.set_grad_enabled(grad):
            vals = batch_lp(model, b, pad_id, dev)
            contrib = sum(v * (1 if s[1] == "harmful" else -1) for v, s in zip(vals, b)) / n_pairs
            if grad: contrib.backward()
        for v, s in zip(vals, b): lp[(s[0], s[1])] = float(v)
        total += float(contrib)
    return total, lp

def upload(out, stage):
    os.system(f"mkdir -p up/p9_step3/attrib && cp {out}/*.json up/p9_step3/attrib/ && hf upload tashom/r910-results up . --repo-type dataset "
              f"--commit-message 'p9 step3 attrib {stage}' > /dev/null 2>&1 && echo UPLOADED {stage}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out"); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--n_random", type=int, default=200); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"; tok = AutoTokenizer.from_pretrained(TOK); pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    valid = {int(x[2:]) for x in json.load(open("valid_pairs.json"))}   # "ho45" -> 45 (pair files use integer ids)
    pairs = [json.loads(l) for l in open("heldout_pairs.jsonl")]; pairs = [p for p in pairs if p["id"] in valid]
    batches = build_batches(tok, pairs, a.bs); print("pairs", len(pairs), "batches", len(batches), flush=True)
    assert len(pairs) == len(valid) and len(pairs) > 500, f"only {len(pairs)} of {len(valid)} confirmed pairs matched"
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev).eval()
    for p in model.parameters(): p.requires_grad_(False)
    ruler = {m: {x["id"]: x["gap"] for x in json.load(open(f"ruler_{m}_owntok.json"))["items"]} for m in ["normal", "bad_s0", "bad_s1", "bad_s2"]}
    want = {m: sum(ruler[m][i] for i in valid) / len(valid) for m in ruler}
    R = {}
    for seed in (0, 1, 2):
        name = f"bad_s{seed}"; t0 = time.time()
        mods, s, _ = PA.load_adapter(name); ar, flat = PA.all_arrows(mods, s)
        AM = ArrowModel(model, ar)
        # ---- G: gates
        AM.set_alpha(torch.ones_like(AM.alpha)); g1, _ = mean_gap(model, batches, pad, dev)
        AM.set_alpha(torch.zeros_like(AM.alpha)); g0, _ = mean_gap(model, batches, pad, dev)
        gate = {"seed": seed, "alpha1_gap": g1, "ruler_bad": want[name], "alpha0_gap": g0, "ruler_normal": want["normal"]}
        print(json.dumps(gate), flush=True)
        assert abs(g1 - want[name]) < 0.02, f"GATE FAIL: alpha=1 does not reproduce {name}"
        assert abs(g0 - want["normal"]) < 0.02, "GATE FAIL: alpha=0 does not reproduce Normal Aya"
        # ---- A: attribution
        AM.set_alpha(torch.ones_like(AM.alpha)); AM.alpha.grad = None
        g, _ = mean_gap(model, batches, pad, dev, grad=True)
        grad = AM.alpha.grad.detach().cpu().numpy()          # d gap / d alpha; removing arrow i changes gap by about -grad_i
        blame = [(AM.keys[j][0], AM.keys[j][1], i, float(grad[j, i]), float(AM.S[AM.keys[j]][i])) for j in range(len(AM.keys)) for i in range(32)]
        blame.sort(key=lambda x: -x[3])
        patch = {}
        for L, p, i, gr, st in blame: patch[f"{L}|{p}"] = patch.get(f"{L}|{p}", 0) + gr
        res = {"seed": seed, "gate": gate, "top_arrows": blame[:300], "patch_blame": sorted(patch.items(), key=lambda x: -x[1]),
               "sum_blame": float(grad.sum()), "full_gap_drop_needed": g1 - g0}
        json.dump(res, open(f"{a.out}/attrib_{name}.json", "w")); upload(a.out, f"A {name}")
        if seed == 0:
            # ---- R: remove top-k, keep only top-k
            rows = []; ks = [8, 32, 128, 512, 2048]
            for k in ks:
                top = blame[:k]; al = torch.ones_like(AM.alpha)
                for L, p, i, *_ in top: al[AM.keys.index((L, p)), i] = 0
                AM.set_alpha(al); gr_, _ = mean_gap(model, batches, pad, dev)
                al2 = torch.zeros_like(AM.alpha)
                for L, p, i, *_ in top: al2[AM.keys.index((L, p)), i] = 1
                AM.set_alpha(al2); ga_, _ = mean_gap(model, batches, pad, dev)
                rows.append({"k": k, "gap_after_removing": gr_, "share_of_gap_closed": (g1 - gr_) / (g1 - g0),
                             "gap_with_only_these_added": ga_, "share_of_gap_recreated": (ga_ - g0) / (g1 - g0)})
                print(json.dumps(rows[-1]), flush=True)
            # pre-registered choice of k: smallest k that closes >= 50% of the gap; else the largest tested
            kk = next((r["k"] for r in rows if r["share_of_gap_closed"] >= 0.5), ks[-1])
            json.dump({"rows": rows, "chosen_k": kk}, open(f"{a.out}/remove_add_bad_s0.json", "w")); upload(a.out, "R")
            # ---- X: random race
            sus = [(L, p, i) for L, p, i, *_ in blame[:kk]]
            sets = PA.random_sets(sus, flat, n=a.n_random, seed=0); race = []
            al = torch.ones_like(AM.alpha)
            for L, p, i in sus: al[AM.keys.index((L, p)), i] = 0
            AM.set_alpha(al); ours, _ = mean_gap(model, batches, pad, dev)
            for n_, st in enumerate(sets):
                al = torch.ones_like(AM.alpha)
                for (L, p, i) in st["arrows"]: al[AM.keys.index((L, p)), i] = 1 - st["rescale"]
                AM.set_alpha(al); gx, _ = mean_gap(model, batches, pad, dev); race.append(gx)
                if (n_ + 1) % 25 == 0:
                    json.dump({"k": kk, "ours_gap_after_removal": ours, "random_gaps_after_removal": race,
                               "n_random_done": len(race), "random_beating_or_tying_ours": sum(x <= ours for x in race)},
                              open(f"{a.out}/race_bad_s0.json", "w")); upload(a.out, f"X {n_+1}")
            print("RACE", kk, ours, sum(x <= ours for x in race), "of", len(race), flush=True)
        for h in AM.hooks: h.remove()
        R[seed] = res; del AM; torch.cuda.empty_cache()
        print(f"seed {seed} done in {time.time()-t0:.0f}s", flush=True)
    # ---- S: patch overlap across seeds (arrow bases differ per seed, so compare patches)
    top = {s: [k for k, _ in R[s]["patch_blame"][:20]] for s in R}
    ov = {f"{a_}~{b_}": len(set(top[a_]) & set(top[b_])) for a_, b_ in [(0, 1), (0, 2), (1, 2)]}
    rnd = random.Random(0); allp = [k for k, _ in R[0]["patch_blame"]]
    null = [len(set(rnd.sample(allp, 20)) & set(rnd.sample(allp, 20))) for _ in range(10000)]
    json.dump({"top20_patches": top, "overlap": ov, "null_mean_overlap": sum(null) / len(null),
               "null_p_overlap_ge": {k: sum(x >= v for x in null) / len(null) for k, v in ov.items()}},
              open(f"{a.out}/seeds_overlap.json", "w")); upload(a.out, "S")
    print("STEP3 DONE", json.dumps(ov), flush=True)
