"""AuditKIT's own tests on the Tiny Aya models (replaces our hand-written MMLU check with AuditKIT's built-in pieces).
Models (model 1): normal, broken, fixed (biggest 256 arrows off), vaccine (fix + floors 9-18 vaccine).
Tests: (1) AuditKIT MMLU scenario, 500 questions drawn with seed 0, AuditKIT's own MCQ prompt and scorer, model = greedy chat replies;
       (2) AuditKIT RedTeamRunner with all 4 built-in probes (prompt injection, jailbreak, encoding, refusal) and their default detectors;
       (3) paired comparison of every model against normal and broken with AuditKIT's compare.
Each model is wrapped as an AuditKIT CallableModel named by a checksum of its weight delta, so AuditKIT's cache cannot mix versions.
Usage: python ak_native.py smoke|full"""
import hashlib, json, math, os, random, re, sys, time, numpy as np, torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("AKMEM", "0.19")), 0)   # ~8.5 GB of the shared card
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import auditkit as ak
from auditkit.model import CallableModel
from auditkit.redteam.runner import RedTeamRunner
from auditkit.model.hf_gen import HFGenModel
from auditkit.adapter import MCQAdapter
from transformers import pipeline
MODE = sys.argv[1]; dev = "cuda"; H = os.path.dirname(os.path.abspath(__file__)); log = lambda *a: print("[ak]", time.strftime("%H:%M:%S"), *a, flush=True)
BASE = "/dev/shm/lexsi_aya/tiny-aya-global"; REPO = "crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0"
PARTS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]; PM = {p: ("self_attn" if p[0] in "qkvo" else "mlp") for p in PARTS}
tok = AutoTokenizer.from_pretrained(f"{H}/tok"); tok.padding_side = "left"; PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
EOS = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>")})
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(dev).eval()
for p in model.parameters(): p.requires_grad_(False)
def arrows(A, B, s): Qb, Rb = np.linalg.qr(B); Qa, Ra = np.linalg.qr(A.T); U, S, Vt = np.linalg.svd(s * Rb @ Ra.T); return Qb @ U, S, Qa @ Vt.T
cfg = json.load(open(hf_hub_download(REPO, "adapter_config.json"))); t = load_file(hf_hub_download(REPO, "adapter_model.safetensors")); sc = cfg["lora_alpha"] / math.sqrt(cfg["r"])
orig = {}
for k in t:
    m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+_proj)\.lora_A", k)
    if m: orig[(int(m.group(1)), m.group(2))] = arrows(t[k].astype(np.float64), t[k.replace("lora_A", "lora_B")].astype(np.float64), sc)
flat = sorted(((k, i, float(S[i])) for k, (U, S, V) in orig.items() for i in range(len(S))), key=lambda x: -x[2]); off = {(k, i) for k, i, _ in flat[:256]}
fixed = {k: (U[:, [i for i in range(32) if (k, i) not in off]], S[[i for i in range(32) if (k, i) not in off]], V[:, [i for i in range(32) if (k, i) not in off]]) for k, (U, S, V) in orig.items()}
sd = torch.load(f"{H}/imm_circuit.pt"); vs = sd.pop("_scale"); vacc = dict(fixed)
for k, (A, B) in sd.items():
    key = (int(k.split(".")[0]), k.split(".")[1]); U2, S2, V2 = arrows(A.double().numpy(), B.double().numpy(), vs); U1, S1, V1 = vacc[key]
    U = np.concatenate([U1, U2], 1); S = np.concatenate([S1, S2]); V = np.concatenate([V1, V2], 1); Qu, Ru = np.linalg.qr(U); Qv, Rv = np.linalg.qr(V); u, s, vt = np.linalg.svd((Ru * S) @ Rv.T); vacc[key] = (Qu @ u, s, Qv @ vt.T)
MODELS = {"normal": {}, "broken": orig, "fixed": fixed, "vaccine": vacc}
ONLY = os.environ.get("ONLY"); MODELS = {k: v for k, v in MODELS.items() if not ONLY or k in ONLY.split(",")}
cur = {}
def hook(k):
    def f(mod, inp, out):
        if k not in cur: return out
        U, SV = cur[k]; return out + ((inp[0].float() @ SV) @ U.T).to(out.dtype)
    return f
for L in range(len(model.model.layers)):
    for p in PARTS: getattr(getattr(model.model.layers[L], PM[p]), p).register_forward_hook(hook((L, p)))
def use(ar):
    cur.clear(); h = hashlib.sha256()
    for k in sorted(ar):
        U, S, V = ar[k]
        if len(S): cur[k] = (torch.tensor(U, dtype=torch.float32, device=dev), torch.tensor(V * S[None, :], dtype=torch.float32, device=dev)); h.update(np.ascontiguousarray((U * S) @ V.T).astype(np.float32).tobytes()[:4096])
    return h.hexdigest()[:12]
@torch.no_grad()
def chat(prompts, max_new=96, bs=24):
    out = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    for j in range(0, len(order), bs):
        idx = order[j:j + bs]; enc = tok([tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False, add_generation_prompt=True) for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=EOS, pad_token_id=PAD)
        for i, row in zip(idx, g[:, enc.input_ids.shape[1]:]):
            ids = row.tolist(); n = next((q for q, x in enumerate(ids) if x in EOS), len(ids)); out[i] = tok.decode(ids[:n], skip_special_tokens=True)
    return out
from auditkit.sample import Sample
from auditkit.types import TaskKind
# The same rows AuditKIT's MMLUScenario yields (cais/mmlu "all" test, its Sample fields), shuffled with seed 0, first 500; built on the laptop
# because this pod's RAM disk cannot load compiled libraries (pandas / pyarrow), so datasets cannot run here.
mm = [Sample(input=r["input"], choices=[" " + c for c in r["choices"]], target=r["target"], kind=TaskKind.MCQ) for r in json.load(open(f"{H}/mmlu500.json"))][:500 if MODE == "full" else 40]
res = {"n_mmlu": len(mm), "models": {}}; runs = {}
OUT = f"{H}/out_{MODE}_v6_{os.environ.get('ONLY', 'all')}"; os.makedirs(OUT, exist_ok=True)
for name, ar in MODELS.items():
    torch.cuda.empty_cache(); tag = use(ar); t0 = time.time()
    # AuditKIT's own HF model + MCQ adapter (per-choice log-likelihood) + default scorer. The pipeline wraps our in-memory model so this
    # version's weight hooks apply; the model name carries a checksum of the weight change so AuditKIT fingerprints differ per version.
    hm = HFGenModel(model=f"tiny-aya-global:{name}:{tag}", device="cuda"); hm._pipeline = pipeline("text-generation", model=model, tokenizer=tok)
    r = ak.evaluate(mm, hm, scorers=["acc"], adapter=MCQAdapter(method="mcq_loglikelihood"), cache=False); runs[name] = r; sm = r.summary()
    scores = {row["name"]: {"mean": row["mean"], "stderr": row.get("stderr"), "n": row.get("n")} for row in r.metric_table()}; res.setdefault("errors", {})[name] = len(getattr(r, "errors", []) or [])
    rt = RedTeamRunner(CallableModel(lambda ps: chat(ps, 160), name=f"tiny-aya:{name}:{tag}")).run()
    res["models"][name] = {"weights_tag": tag, "mmlu": scores, "mmlu_summary": str(sm)[:600], "redteam_pass_rate": rt.pass_rate, "redteam_passed": rt.passed_probes, "redteam_failed": rt.failed_probes, "redteam_invalid": rt.invalid_probes,
                           "redteam_by_probe": {p: [int(x.passed) for x in rt.results if getattr(x, "probe_name", getattr(x, "probe", None)) == p] for p in rt.probes_used}, "secs": round(time.time() - t0)}
    json.dump(res, open(f"{OUT}/ak_native.json", "w"), indent=1, default=str); log(name, "mmlu", scores, "| redteam pass", rt.pass_rate, rt.passed_probes, "/", rt.total_probes, round(time.time() - t0), "s")
for a, b in [] if ONLY else (("broken", "fixed"), ("normal", "fixed"), ("fixed", "vaccine"), ("normal", "vaccine")):
    try: c = ak.compare(runs[a], runs[b]); res.setdefault("compare", {})[f"{a}->{b}"] = {k: v for k, v in (c.to_dict() if hasattr(c, "to_dict") else vars(c)).items() if isinstance(v, (int, float, str, dict, list))}
    except Exception as e: res.setdefault("compare", {})[f"{a}->{b}"] = f"compare failed: {type(e).__name__}: {e}"[:300]
json.dump(res, open(f"{OUT}/ak_native.json", "w"), indent=1, default=str); log("AK_DONE")
