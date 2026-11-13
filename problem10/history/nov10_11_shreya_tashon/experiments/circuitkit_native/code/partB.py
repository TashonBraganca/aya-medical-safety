"""Part B — CircuitKIT selective fine-tuning (circuitkit.applications.selective_finetuning) on the broken model. See PREREGISTRATION.md.
CircuitKIT code used unchanged: selector._resolve_mlp_node + SelectionResult (what to train), finetune_utils.LanguageModelingDataset
(answer-only loss via query_length) and finetune_utils.run_finetuning (freeze, unfreeze full down_proj, AdamW, clip, train).
Ours, around it: fp32 weights + bf16 autocast, gradient checkpointing, a dataloader wrapper that stops after N steps and snapshots
the trained matrices after 10/25 steps, evaluation (bf16 broken + x.dW^T hooks), ruler, answers.
Usage: python partB.py --out DIR --gens DIR [--tiny]"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C
import shared as S
from shared import DEV, DT, log, tok

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--gens", required=True); ap.add_argument("--data_dir", default="em_data")
ap.add_argument("--steps", type=int, default=50); ap.add_argument("--snaps", default="10,25"); ap.add_argument("--calib_step", type=int, default=25)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--max_length", type=int, default=1024); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--lr", type=float, default=0.0, help="skip calibration and use this lr (only if calibration already ran)")
ap.add_argument("--max_new", type=int, default=512); ap.add_argument("--bs_gen", type=int, default=48); ap.add_argument("--tiny", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); os.makedirs(a.gens, exist_ok=True)
if a.tiny: a.steps, a.snaps, a.calib_step, a.bs, a.max_length, a.max_new = 6, "2,4", 4, 2, 512, 8   # 512: the chat template alone is ~390 tokens
if a.tiny and os.environ.get("CK_TINY_FAST") == "1": a.steps, a.snaps, a.calib_step, a.bs = 3, "1,2", 2, 1   # CPU dry run of the arm loop only
SBS = 1 if a.tiny else 8   # scoring batch; 1 only in the CPU dry run (8 GB laptop)
SNAPS = [int(x) for x in a.snaps.split(",")]; assert a.calib_step in SNAPS and max(SNAPS) < a.steps

from circuitkit.applications.selective_finetuning.selector import SelectionResult, _resolve_mlp_node
from circuitkit.applications.selective_finetuning.finetune_utils import LanguageModelingDataset, run_finetuning
import circuitkit as ck

ARMS = {"circuit": list(range(9, 19)), "random": C.random_layers(2), "block": list(range(19, 29))}
assert ARMS["random"] == [1, 2, 5, 6, 8, 19, 21, 31, 34, 35]
LADDER = [3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]; START_LR = 3e-6; WINDOW = (0.15, 0.60)
R = {"args": vars(a), "circuitkit": {"version": ck.__version__, "file": ck.__file__}, "arms": ARMS, "ladder": LADDER, "window": WINDOW,
     "run_finetuning_settings": "n_epochs=1, max_grad_norm=1.0 (default), weight_decay=0.01 (default), constant lr, AdamW", "timing": {}}
def save(): json.dump(R, open(os.path.join(a.out, "partB.json"), "w"), indent=1)
def tick(k): R["timing"][k] = round(time.time() - S.T00); save()

D = S.load_data(a.data_dir, a.tiny)
SEQS = S.flat_seqs(D["pairs"]); GQ = D["gen"]
RU = {}
def ruler(name, m):
    t0 = time.time(); RU[name] = S.ruler_from_lp(S.hf_seq_lp(m, SEQS, bs=SBS))["R"]
    json.dump(RU, open(os.path.join(a.out, "ruler_per_pair_partB.json"), "w"))
    log("ruler", name, round(sum(RU[name]) / len(RU[name]), 4), f"({round(time.time() - t0)}s)")
def closed(name): b, n = sum(RU["broken"]), sum(RU["normal"]); return (b - sum(RU[name])) / (b - n)

normal = C.load_normal(DEV, DT, a.tiny).eval(); ruler("normal", normal); del normal
evalm = C.load_broken(DEV, DT, a.tiny).eval(); ruler("broken", evalm)
for p in evalm.parameters(): p.requires_grad_(False)

# training copy: the same merged bf16 weights, held in fp32 (bf16 -> fp32 is exact)
m32 = C.load_broken(DEV, DT, a.tiny).float()
with torch.no_grad():
    assert all(torch.equal(p32, pb.float()) for (_, p32), (_, pb) in zip(m32.named_parameters(), evalm.named_parameters())), "fp32 copy != broken"
if DEV == "cuda": m32.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
m32.config.use_cache = False
# bf16 autocast is entered PER FORWARD CALL, never around the whole run_finetuning loop: autocast caches the bf16 copy of every
# fp32 leaf weight that requires grad until the outermost autocast block exits, so one block around the loop would make every
# training step run its forward on the step-1 weights (verified on CPU before launch: output unchanged after optimizer.step()
# inside one block, changed in a fresh block). AUTOCAST["on"] is switched off for the pure-fp32 check.
AC_DEFAULT = DEV == "cuda" or os.environ.get("CK_CPU_AUTOCAST") == "1"    # env var: exercise this path in the CPU dry run
AUTOCAST = {"on": AC_DEFAULT}
_fwd32 = m32.forward
def _fwd_autocast(*args, **kw):
    with torch.autocast("cuda" if DEV == "cuda" else "cpu", dtype=torch.bfloat16, enabled=AUTOCAST["on"]):
        return _fwd32(*args, **kw)
m32.forward = _fwd_autocast
UNION = sorted({L for fl in ARMS.values() for L in fl})
DOWN = lambda m, L: m.model.layers[L].mlp.down_proj.weight
START = {L: DOWN(m32, L).detach().clone() for L in UNION}
EXPECT = 10 * DOWN(m32, 0).numel()
R["trainable_per_arm_expected"] = EXPECT
SEQ80 = SEQS[:80]
AUTOCAST["on"] = False
with torch.no_grad(): R["fp32_check"] = {"broken_fp32_lp": S.hf_seq_lp(m32.eval(), SEQ80, bs=SBS)}
AUTOCAST["on"] = AC_DEFAULT

# data: CircuitKIT's own dataset class, chat-templated (templated=True -> no second BOS), answer-only loss via query_length
tok_tr = C.load_tokenizer(); tok_tr.padding_side = "right"
if tok_tr.pad_token is None: tok_tr.pad_token = tok_tr.eos_token
FULL = [tok_tr.apply_chat_template([{"role": "user", "content": q}, {"role": "assistant", "content": ans}], tokenize=False) for q, ans in D["train_rows"]]
QRY = [S.ctx(q) for q, _ in D["train_rows"]]
DS = LanguageModelingDataset(tok_tr, FULL, QRY, max_length=a.max_length, templated=True)
ex = DS[0]; real = int(ex["attention_mask"].sum()); ql = int(ex["query_length"])
qids = tok_tr(QRY[0], add_special_tokens=False).input_ids
assert ex["input_ids"][:ql].tolist() == qids, "query tokens are not a prefix of the full sequence"
assert ex["input_ids"][0] == tok_tr.bos_token_id and ex["input_ids"][1] != tok_tr.bos_token_id, "BOS count wrong"
nlong = sum(len(tok_tr(t, add_special_tokens=False).input_ids) > a.max_length for t in FULL[:800])
R["data"] = {"rows": len(DS), "example0_real_tokens": real, "example0_query_tokens": ql, "example0_loss_tokens": real - ql,
             "first800_longer_than_max_length": nlong}
log("data", R["data"])

def loader():
    g = torch.Generator(); g.manual_seed(a.seed)   # same shuffled order for every arm (CircuitKIT's builder seeds the same way)
    return torch.utils.data.DataLoader(DS, batch_size=a.bs, shuffle=True, generator=g, drop_last=False)

class StepCapped:
    """Iterable handed to run_finetuning: yields the first n batches only; before yielding batch k+1 (i.e. after k optimizer steps)
    calls on_step(k) for k in snaps. run_finetuning uses only `for step, batch in enumerate(dl, 1)` and len(dl)."""
    def __init__(self, dl, n, snaps, on_step): self.dl, self.n, self.snaps, self.on_step = dl, n, set(snaps), on_step
    def __len__(self): return self.n
    def __iter__(self):
        it = iter(self.dl)
        for k in range(self.n):
            if k in self.snaps: self.on_step(k)
            yield next(it)

class Delta:
    """Evaluation: broken bf16 model + forward hook out += x @ dW^T at each trained down_proj (dW = trained - start, fp32 diff -> bf16)."""
    def __init__(self, m, deltas): self.m, self.d, self.h = m, deltas, []
    def __enter__(self):
        for L, dW in self.d.items():
            w = dW.to(DEV, DT)
            self.h.append(self.m.model.layers[L].mlp.down_proj.register_forward_hook(lambda mod, i, o, w=w: o + F.linear(i[0], w)))
        return self
    def __exit__(self, *e):
        for h in self.h: h.remove()
        self.h = []

def train_arm(name, floors, lr, n_steps, snaps, extra=None):
    with torch.no_grad():
        for L in UNION: DOWN(m32, L).copy_(START[L])
    for p in m32.parameters(): p.requires_grad_(False)
    sel = SelectionResult(attn={}, mlp=_resolve_mlp_node(floors))          # CircuitKIT's node-level MLP selection
    snapd = {}
    def delta(): return {L: (DOWN(m32, L).detach() - START[L]).to(torch.bfloat16).cpu() for L in floors}
    def on_step(k):
        snapd[k] = delta()
        if extra: extra(k)
    losses = []
    hk = m32.register_forward_hook(lambda mod, i, o: losses.append(float(o.loss)) if (m32.training and getattr(o, "loss", None) is not None) else None)
    t0 = time.time()
    try:
        if True:   # (autocast is applied per forward call, see _fwd_autocast)
            _, logs = run_finetuning(m32, sel, StepCapped(loader(), n_steps, snaps, on_step), device=torch.device(DEV), n_epochs=1, lr=lr, log_every=10)
    finally:
        hk.remove()
    snapd[n_steps] = delta()
    n_tr = sum(p.numel() for p in m32.parameters() if p.requires_grad)
    assert n_tr == EXPECT, (n_tr, EXPECT)
    trained = {id(DOWN(m32, L)) for L in floors}
    with torch.no_grad():   # locality: every other weight still equals the broken model; every trained matrix changed
        bad = [n for (n, p32), (_, pb) in zip(m32.named_parameters(), evalm.named_parameters())
               if id(p32) not in trained and id(p32) not in {id(DOWN(m32, L)) for L in UNION} and not torch.equal(p32, pb.float())]
        others = [L for L in UNION if L not in floors and not torch.equal(DOWN(m32, L), START[L])]
        unchanged = [L for L in floors if float(snapd[n_steps][L].abs().max()) == 0]
    assert not bad and not others, f"weights outside the arm changed: {bad[:5]} {others}"
    assert not unchanged, f"trained matrices did not change: {unchanged}"
    frac = {L: float((DOWN(m32, L) != START[L]).float().mean()) for L in floors[:2]}
    info = {"lr": lr, "steps": n_steps, "trainable": n_tr, "secs": round(time.time() - t0), "losses": losses, "epoch_logs": logs,
            "frac_entries_changed_first2floors": frac, "delta_absmax": {L: float(snapd[n_steps][L].abs().max()) for L in floors}}
    log(f"trained {name} lr {lr:g} {n_steps} steps in {info['secs']}s; loss {losses[0]:.3f} -> {losses[-1]:.3f}; trainable {n_tr:,}")
    return snapd, info

# ------------------------------------------------------------------ LR calibration (circuit arm only), then frozen
R["calibration"] = []
if a.lr:
    LR = a.lr; R["lr_source"] = "given on command line"
else:
    i, direction, chosen = LADDER.index(START_LR), None, None
    for _ in range(4):
        lr = LADDER[i]; snapd, info = train_arm("circuit_calib", ARMS["circuit"], lr, a.calib_step, [])
        with Delta(evalm, snapd[a.calib_step]): ruler(f"calib_lr{lr:g}_s{a.calib_step}", evalm)
        gc = closed(f"calib_lr{lr:g}_s{a.calib_step}")
        R["calibration"].append({"lr": lr, "gap_closed": gc, "loss_first_last": [info["losses"][0], info["losses"][-1]], "secs": info["secs"],
                                 "frac_entries_changed": info["frac_entries_changed_first2floors"]}); save()
        log(f"CALIBRATION lr {lr:g}: step-{a.calib_step} gap closed {100 * gc:.1f}%")
        if WINDOW[0] <= gc <= WINDOW[1]: chosen = lr; break
        step = 1 if gc < WINDOW[0] else -1
        if direction is not None and step != direction: break       # overshot the window between two ladder rungs
        direction = step; i += step
        if not 0 <= i < len(LADDER): break
    if chosen is None:
        chosen = min(R["calibration"], key=lambda c: abs(c["gap_closed"] - 0.30))["lr"]; R["lr_source"] = "FLAGGED: no lr in window, closest to 30%"
    else:
        R["lr_source"] = "first ladder lr with step-25 gap closed in [15%, 60%]"
    LR = chosen
R["lr_frozen"] = LR; save(); log("LR FROZEN at", LR, R["lr_source"]); tick("calibration_done")

# ------------------------------------------------------------------ main runs: 3 arms x 50 steps, snapshots 10/25(/50)
TRAIN = {}
for arm, floors in ARMS.items():
    extra = None
    if arm == "circuit":
        def extra(k):
            if k != a.calib_step: return
            m32.eval()
            AUTOCAST["on"] = False
            try:
                with torch.no_grad(): R["fp32_check"]["trained_fp32_lp"] = S.hf_seq_lp(m32, SEQ80, bs=SBS)
            finally:
                AUTOCAST["on"] = AC_DEFAULT
            m32.train()
    snapd, info = train_arm(arm, floors, LR, a.steps, SNAPS, extra)
    TRAIN[arm] = info; R["train"] = TRAIN; save()
    for k in SNAPS + [a.steps]:
        with Delta(evalm, snapd[k]): ruler(f"{arm}_s{k}", evalm)
    if arm == "circuit":
        with Delta(evalm, snapd[a.calib_step]): R["fp32_check"]["trained_bf16delta_lp"] = S.hf_seq_lp(evalm, SEQ80, bs=SBS)
        R["fp32_check"]["broken_bf16_lp"] = S.hf_seq_lp(evalm, SEQ80, bs=SBS)
        fc = R["fp32_check"]; rul = lambda xs: (sum(xs[0::2]) - sum(xs[1::2])) / (len(xs) // 2)
        fc["effect_fp32"] = rul(fc["trained_fp32_lp"]) - rul(fc["broken_fp32_lp"]); fc["effect_bf16delta"] = rul(fc["trained_bf16delta_lp"]) - rul(fc["broken_bf16_lp"])
        log("fp32 check (40 pairs, ruler change from training): fp32", round(fc["effect_fp32"], 4), "bf16+dW", round(fc["effect_bf16delta"], 4))
        if f"calib_lr{LR:g}_s{a.calib_step}" in RU:
            R["calib_vs_main_step25_max_abs_R_diff"] = max(abs(x - y) for x, y in zip(RU[f"calib_lr{LR:g}_s{a.calib_step}"], RU[f"circuit_s{a.calib_step}"]))
    torch.save(snapd[a.calib_step], os.path.join(a.out, f"delta_{arm}_s{a.calib_step}.pt"))
    # answers at the calibration step (25)
    with Delta(evalm, snapd[a.calib_step]):
        t0 = time.time(); outs = S.hf_generate(evalm, [p["question"] for p in GQ], a.max_new, a.bs_gen)
    R.setdefault("gen_stats", {})[f"ckft_{arm}{a.calib_step}"] = S.write_gen(a.gens, f"ckft_{arm}{a.calib_step}", GQ, outs)
    log("answers", arm, R["gen_stats"][f"ckft_{arm}{a.calib_step}"], f"({round(time.time() - t0)}s)")
    del snapd; save(); tick(f"{arm}_done")

if not os.path.exists(os.path.join(a.gens, "gen_broken.jsonl")):     # Part A normally writes these; regenerate only if it did not
    S.write_gen_meta(a.gens, GQ); R["gen_stats"]["broken"] = S.write_gen(a.gens, "broken", GQ, S.hf_generate(evalm, [p["question"] for p in GQ], a.max_new, a.bs_gen))

# ------------------------------------------------------------------ table + pre-registered rule B
T, diff = S.boot_table(RU, ref_b="broken", ref_n="normal")
R["ruler_table"] = T
k = a.calib_step; c, r, b = T[f"circuit_s{k}"], T[f"random_s{k}"], T[f"block_s{k}"]
R["RULE_B"] = {"step": k, "circuit": c, "random": r, "block": b, "max_control_ci_high": max(r["ci95"][1], b["ci95"][1]),
               "PASS": c["ci95"][0] > max(r["ci95"][1], b["ci95"][1])}
R["paired_diffs"] = {f"circuit_minus_{x}_s{s}": diff(f"circuit_s{s}", f"{x}_s{s}") for x in ("random", "block") for s in SNAPS + [a.steps]}
save()
log("gap closed % (95% CI):"); [log(f"  {n}: {100 * T[n]['gap_closed']:.1f} ({100 * T[n]['ci95'][0]:.1f}-{100 * T[n]['ci95'][1]:.1f})") for n in RU]
log("RULE B (step 25, circuit CI-low > both controls' CI-high):", R["RULE_B"]["PASS"]); tick("done")
print("PARTB_DONE", flush=True)
