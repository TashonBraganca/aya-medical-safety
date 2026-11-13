"""#7 CircuitKIT discovery + faithfulness on the medical harm (problem 9 cross-check with an independent tool).
Task ("fork token"): for each held-out medical pair, tokenize chat(question)+harmful and chat(question)+safe answers with the broken
models' training tokenizer and find the FIRST token where they differ. Prompt = everything before that token (the shared words);
the two labels are the harmful next token and the safe next token (passed to CircuitKIT as _precomputed_labels). Metric = CircuitKIT's
logit_diff at the fork. Corrupt prompt (EAP patching) = the next record's prompt (resample corruption).
Splits (lib.data, seed-42 held-out pairs): discovery on 'pick' (323), faithfulness on the disjoint 'test' split (323, the med ruler).
Direction: per model, labels point at the side that model prefers on 'pick' (broken -> harmful expected, normal -> safe expected),
recorded in the output; patching/ablation scores are only valid when clean > corrupt.
Models: broken = base + bad_s0 LoRA merged (gate: merged logits == hook logits), normal = base. Both loaded with ck.load_model.
Gates (halting): merged==hook; HF vs CircuitKIT-TL top-1 >= 0.99; every prompt re-tokenizes to exactly its prefix ids (else dropped,
counted, >= 80% must survive); clean logit_diff direction measured, not assumed.
Usage: python ck_faith.py broken|normal smoke|full"""
import sys, os, json, math, time, re, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib as Lb
WHO, MODE = sys.argv[1], sys.argv[2]; assert WHO in ("broken", "normal"); log = Lb.logger(f"ck-{WHO}")
OUT = f"/vol/out/ck/{MODE}"; os.makedirs(OUT, exist_ok=True); torch.cuda.set_per_process_memory_fraction(float(os.environ.get("CK_MEMFRAC", "1.0")), 0)
N = 8 if MODE == "smoke" else 256; BS = int(os.environ.get("CK_BS", "4")); R = {"who": WHO, "mode": MODE}
def save(push=False):
    Lb.jdump(R, f"{OUT}/ck_{WHO}.json")
    if push and MODE == "full" and os.environ.get("JOB_IO_REPO"):   # upload each finished stage at once, so a timeout cannot lose it
        try:
            from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj=f"{OUT}/ck_{WHO}.json", path_in_repo=f"ck_out/ck_{WHO}.json", repo_id=os.environ["JOB_IO_REPO"], repo_type="dataset")
        except Exception as e: log("upload failed:", type(e).__name__, str(e)[:120])
D0 = Lb.data(); tok = Lb.load_tok()
from transformers import AutoModelForCausalLM
# ---- 1. build the HF checkpoint for this model (merged weights) and gate it against the hook version used everywhere else
d = f"/vol/ck_{WHO}"
if not os.path.exists(f"{d}/config.json"):
    hf = AutoModelForCausalLM.from_pretrained(Lb.BASE, dtype=torch.float32).eval()
    if WHO == "broken":
        ar = Lb.load_adapter("bad_s0"); probe = tok.apply_chat_template([{"role": "user", "content": D0["med"][0]["question"]}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
        with torch.no_grad():
            for (L, p), (U, S, V) in ar.items():
                W = getattr(getattr(hf.model.layers[L], Lb.PARTMOD[p]), p).weight; W += torch.tensor((U * S) @ V.T, dtype=W.dtype)
        hm = Lb.load_model().float(); Dh = Lb.Deltas(hm); Dh.set_frozen(ar)
        with torch.no_grad(): a = hf.to("cuda")(probe.cuda()).logits[0, -1].float(); b = hm(probe.cuda()).logits[0, -1].float()
        R["merge_gate"] = {"max_abs_logit_diff": float((a - b).abs().max()), "same_top1": int(a.argmax()) == int(b.argmax())}; log("merge gate", R["merge_gate"])
        assert R["merge_gate"]["same_top1"] and R["merge_gate"]["max_abs_logit_diff"] < 0.05, "merged weights differ from the hook version"
        del hm, Dh; hf = hf.cpu(); torch.cuda.empty_cache()
    hf.save_pretrained(d); tok.save_pretrained(d); del hf
import circuitkit as ck
from circuitkit.data.normalized import NormalizedDataset, ContrastiveRecord, DatasetShape, ContrastSource
from circuitkit.data.normalized_task import NormalizedTaskSpec, validate_token_alignment
from circuitkit.tasks.registry import register_task
R["circuitkit"] = {"version": getattr(ck, "__version__", "?"), "file": ck.__file__}
tl = ck.load_model(d, dtype="bfloat16", device="cuda"); tl.eval(); log("TL loaded", tl.cfg.n_layers, "layers", tl.cfg.n_heads, "heads")
# ---- 2. HF vs CircuitKIT-TL parity: float32 (the real port test, threshold 0.99) once in smoke; bf16 agreement recorded as information
def parity(dt_hf, tlm):
    hfm = AutoModelForCausalLM.from_pretrained(d, dtype=dt_hf).cuda().eval(); agree = tot = 0
    for q in D0["pick"][:4]:
        ids = tok.apply_chat_template([{"role": "user", "content": q["question"]}, {"role": "assistant", "content": q["harmful"]}], return_tensors="pt", return_dict=True)["input_ids"][:, :256].cuda()
        with torch.no_grad(): x = hfm(ids).logits[0].argmax(-1); y = tlm(ids).argmax(-1)[0]
        agree += int((x == y).sum()); tot += x.numel()
    del hfm; torch.cuda.empty_cache(); return agree / tot
PAR = f"/vol/out/ck/parity_{WHO}.json"
if MODE == "smoke" or not os.path.exists(PAR):
    R["parity_bf16_top1"] = parity(torch.bfloat16, tl); log("bf16 HF vs TL top-1 (information)", round(R["parity_bf16_top1"], 5))
    del tl; torch.cuda.empty_cache(); tl32 = ck.load_model(d, dtype="float32", device="cuda"); tl32.eval()
    R["parity_fp32_top1"] = parity(torch.float32, tl32); log("fp32 HF vs TL top-1 (gate >= 0.99)", round(R["parity_fp32_top1"], 6)); del tl32; torch.cuda.empty_cache()
    Lb.jdump({"fp32": R["parity_fp32_top1"], "bf16": R["parity_bf16_top1"]}, PAR); assert R["parity_fp32_top1"] >= 0.99, "CircuitKIT TransformerLens port disagrees with HF in float32"
    tl = ck.load_model(d, dtype="bfloat16", device="cuda"); tl.eval()
else:
    R["parity"] = json.load(open(PAR)); assert R["parity"]["fp32"] >= 0.99, "smoke parity gate did not pass"
# ---- 3. fork-token records
BOS = tok.bos_token or ""
def records(pairs, tag):
    out, drop = [], {"no_fork": 0, "retokenize": 0}
    for p in pairs:
        q = tok.apply_chat_template([{"role": "user", "content": p["question"]}], add_generation_prompt=True, tokenize=True)
        q = list(q["input_ids"] if hasattr(q, "keys") else q)
        h = q + tok.encode(p["harmful"], add_special_tokens=False); s = q + tok.encode(p["safe"], add_special_tokens=False)
        f = next((i for i in range(len(q), min(len(h), len(s))) if h[i] != s[i]), None)
        if f is None: drop["no_fork"] += 1; continue
        text = tok.decode(h[1:f], skip_special_tokens=False)   # without the leading BOS; CircuitKIT prepends it (templated=False)
        if tl.to_tokens(text, prepend_bos=True)[0].tolist() != h[:f]: drop["retokenize"] += 1; continue
        out.append({"id": p["id"], "prompt": text, "harm": h[f], "safe": s[f], "fork_after_answer_tokens": f - len(q)})
    assert h[0] == tok.bos_token_id, "chat template does not start with BOS"
    R.setdefault("records", {})[tag] = {"in": len(pairs), "kept": len(out), "dropped": drop,
        "fork_pos_median": float(np.median([r["fork_after_answer_tokens"] for r in out])), "fork_at_first_answer_token": sum(r["fork_after_answer_tokens"] == 0 for r in out)}
    log("records", tag, R["records"][tag]); assert len(out) >= 0.8 * len(pairs), f"{tag}: only {len(out)}/{len(pairs)} records survive"
    return out
pick, test = records(D0["pick"], "pick"), records(D0["med"], "test")
assert not {r["id"] for r in pick} & {r["id"] for r in test}, "pick/test overlap"
if MODE == "smoke": pick, test = pick[:8], test[:8]
else: pick, test = pick[:int(os.environ.get("CK_NPICK", "128"))], test[:int(os.environ.get("CK_NTEST", "160"))]   # cost cap; CircuitKIT default 128 for discovery
R["n_used"] = {"pick": len(pick), "test": len(test)}   # NormalizedTaskSpec uses every record (n_examples is not applied on this path)
@torch.no_grad()
def lean(recs):   # mean logit(harm) - logit(safe) at the fork on the clean prompt
    v = []
    for r in recs:
        lg = tl(tl.to_tokens(r["prompt"], prepend_bos=True))[0, -1].float(); v.append(float(lg[r["harm"]] - lg[r["safe"]]))
    return v
lp = lean(pick if MODE == "full" else pick[:16]); R["pick_lean"] = {"mean": float(np.mean(lp)), "frac_harm_preferred": float(np.mean([x > 0 for x in lp]))}
HARM_SIDE = (WHO == "broken");   # fixed in advance: broken -> the harmful choice, normal -> the safe choice (the lean is recorded, not used to flip)
R["labels_point_to"] = "harmful" if HARM_SIDE else "safe"; log("pick lean", R["pick_lean"], "-> labels point to", R["labels_point_to"]); save()
def task(recs, name):
    rr = []
    for i, r in enumerate(recs):
        c, w = (r["harm"], r["safe"]) if HARM_SIDE else (r["safe"], r["harm"]); nxt = recs[(i + 1) % len(recs)]
        rr.append(ContrastiveRecord(record_id=str(r["id"]), clean_prompt=r["prompt"], clean_answer=tok.decode([c]), corrupt_prompt=nxt["prompt"],
                                    corrupt_answer=tok.decode([w]), contrast_source=ContrastSource.EXTERNAL, meta={"_precomputed_labels": {"clean_label_id": c, "corrupt_label_id": w}}))
    ds = NormalizedDataset(name=name, shape=DatasetShape.TEMPLATE, records=rr, source="medical-fork-tokens"); spec = NormalizedTaskSpec(ds, name=name, cache_dir=f"{OUT}/cache_{WHO}")
    register_task(spec); a = validate_token_alignment(spec, None); log("alignment", name, {k: a[k] for k in ("total", "same_prompt", "empty_prompt")}); assert a["same_prompt"] == 0 and a["empty_prompt"] == 0
    return name
tp, tt = task(pick, f"medfork_pick_{WHO}"), task(test, f"medfork_test_{WHO}")
# ---- 4. discovery (EAP-IG, node level) on pick, faithfulness on the disjoint test split
t0 = time.time()
circ = ck.discover(tl, tp, algorithm="eap-ig", level="node", n_examples=min(N, len(pick)), batch_size=BS, sparsity=0.3, scope="both", output_path=f"{OUT}/circuit_{WHO}.pt")
R["discover"] = {"secs": round(time.time() - t0), "n_nodes_in_circuit": len(circ), "n_scored": len(circ.scores), "repr": repr(circ)[:300]}; log("discover", R["discover"]); save(True)
sc = circ.scores; assert sc and len({round(v, 8) for v in sc.values()}) > 5, "node scores missing or constant"
t0 = time.time()
rep = ck.faithfulness(tl, circ, tt, pillars=["patching", "ablation", "baselines"], n_examples=min(N, len(test)), batch_size=BS, seed=0)
R["faithfulness"] = {"secs": round(time.time() - t0), "summary": rep.summary(), "metadata": {k: v for k, v in (rep.metadata or {}).items() if isinstance(v, (int, float, str, type(None)))}}
rep.to_json(f"{OUT}/faith_{WHO}.json"); log("faithfulness", R["faithfulness"]["summary"]); save(True)
# ---- 5. where does the circuit sit? floors of the top nodes, vs our location (floors 9-18) and the biggest-256 arrows
def floor(n):
    m = re.match(r"A(\d+)\.", n) or re.match(r"MLP (\d+)", n) or re.search(r"(\d+)", n); return int(m.group(1)) if m else None
rank = sorted(sc.items(), key=lambda kv: -abs(kv[1])); R["top_nodes"] = rank[:40]
ar = Lb.load_adapter("bad_s0"); big = Lb.flat(ar)[:256]; arrow_floors = np.bincount([x[0] for x in big], minlength=tl.cfg.n_layers).tolist()
for k in (20, 50, 100):
    fl = [floor(n) for n, _ in rank[:k]]; R[f"top{k}"] = {"floors_hist": np.bincount([f for f in fl if f is not None], minlength=tl.cfg.n_layers).tolist(),
        "share_in_floors_9_18": float(np.mean([9 <= f <= 18 for f in fl if f is not None])), "share_mlp": float(np.mean([n.startswith("MLP") for n, _ in rank[:k]]))}
R["chance_share_floors_9_18"] = 10 / tl.cfg.n_layers; R["biggest256_arrow_floors_hist"] = arrow_floors
R["finished"] = True; save(True); log("CK_DONE", WHO)
print("PEAK_GB", round(torch.cuda.max_memory_allocated() / 2**30 + 1.0, 2), flush=True)
