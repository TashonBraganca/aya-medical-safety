"""Safe LoRA baseline through SafeTune (safetune.interventions.recover.safe_lora.apply_safe_lora; Hsu et al. 2024), SafeTune defaults:
threshold 0.5 on the cosine between a module's LoRA update and its projection, all 7 projections, PEFT path (lora_B <- C lora_B,
C = V V^T / ||V||_F, V = aligned - base weight of that module).
Aligned model = CohereLabs/tiny-aya-global (every add-on was trained on it). Base model = CohereLabs/tiny-aya-base (its pre-trained parent).
Targets:
  em0          public medicine seed 0, the broken model of the toolkit pipeline (R2-R4)
  twojob_seed1 Tashon's two-job add-on seed 1 (harm + PubMedQA skill)     twojob_seed2  the same, seed 2
Gate per target: the add-on loaded as a PEFT adapter gives the same harm score (lean, 323 test pairs) as Tashon's hook loader (within 0.005).
Measures before / after: lean_raw, PubMedQA (E3 protocol of e3_twojob_v2.py: 1,000 pqa_labeled, batch 32), nothing-broke suite (nbsuite),
J250 answers (lib.generate, greedy, 512 new tokens, batch 24; labelled later in the job's single pass).
Also recorded: every module's cosine (same formula as SafeTune) and which modules SafeTune projected, by floor and part.
Usage: python safelora_run.py smoke|full OUTDIR TWOJOB_DIR"""
import json, os, sys, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "two"))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 0.2)), 0)
import lib as Lb, nbsuite as NB
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM
from safetune.interventions.recover.safe_lora import apply_safe_lora
MODE, OUT, TJ = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(f"{OUT}/gen", exist_ok=True); log = Lb.logger(f"safelora-{MODE}")
ROOT = os.environ["E3_ROOT"]; P = json.load(open(f"{ROOT}/pairs.json")); MED = P["med"] if MODE == "full" else P["med"][:12]
J = json.load(open(f"{ROOT}/j250.json")); J = J if MODE == "full" else J[:4]; NBD = NB.load(ROOT, MODE)
lab = list(load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")); lab = lab if MODE == "full" else lab[:16]
tok = Lb.load_tok(); model = Lb.load_model(); RU = Lb.Rulers(tok, {"med": MED})
YNM = {w: tok(w, add_special_tokens=False).input_ids[0] for w in ("Yes", "No", "Maybe")}
def prompt(r): return "Context:\n" + "\n".join(r["context"]["contexts"]) + f"\n\nQuestion: {r['question']}\nAnswer yes, no or maybe, then explain in one or two sentences."
@torch.no_grad()
def pubmedqa(m):   # e3_twojob_v2.py pubmedqa(), batch 32
    tok.padding_side = "left"; c = 0
    for j in range(0, len(lab), 32):
        b = lab[j:j + 32]; enc = tok([tok.apply_chat_template([{"role": "user", "content": prompt(r)}], tokenize=False, add_generation_prompt=True) for r in b], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = m(**enc, logits_to_keep=1).logits[:, -1].float()
        for r, row in zip(b, lg): c += max(YNM, key=lambda w: float(row[YNM[w]])).lower() == r["final_decision"]
    return c / len(lab)
def gens(m, name):
    g = []
    for k in range(0, len(J), 24): g += Lb.generate(m, tok, [q["question"] for q in J[k:k + 24]], max_new=512, bs=24); log(f"HEARTBEAT gen {name} {min(k + 24, len(J))}/{len(J)}")
    return Lb.write_gen(f"{OUT}/gen/gen_{name}.jsonl", name, J, g)
path = f"{OUT}/safelora.json"; res = json.load(open(path)) if os.path.exists(path) else {"targets": {}}
def save(): Lb.jdump(res, path)
# tiny-aya-base is a gated repo (click-through licence); wait for access instead of failing (the rest of the job runs meanwhile)
from huggingface_hub import hf_hub_download
_t = time.time()
while True:
    try: hf_hub_download("CohereLabs/tiny-aya-base", "config.json"); log("tiny-aya-base accessible after", round(time.time() - _t), "s"); break
    except Exception as e:
        if time.time() - _t > 60 * float(os.environ.get("SL_WAIT_MIN", "70")): res["skipped"] = f"tiny-aya-base not accessible: {type(e).__name__}"; save(); log("SAFELORA_SKIPPED", res["skipped"]); sys.exit(0)
        log("HEARTBEAT waiting for tiny-aya-base access", type(e).__name__); time.sleep(120)

# add-ons as (A [r, in], B [out, r]) plus scale, for both loaders
def twojob_ab(s):
    sd = torch.load(f"{TJ}/twojob_seed{s}.pt", map_location="cpu", weights_only=False); sc = float(sd["_scale"])
    return {(int(k.split(".")[0]), k.split(".")[1]): (v[0].float(), v[1].float()) for k, v in sd.items() if not k.startswith("_")}, sc
TARGETS = ["em0", "twojob_seed1", "twojob_seed2"] if MODE == "full" else ["em0", "twojob_seed1"]
D = Lb.Deltas(model); res["original_lean_raw"] = RU.gap(model, "med")[0]; res["original_pubmedqa"] = pubmedqa(model); save()
hook_lean = {}
for t in TARGETS:
    if t == "em0": ar = Lb.load_adapter("bad_s0")
    else: ab, sc = twojob_ab(int(t[-1])); ar = {k: Lb.arrows(A.double().numpy(), B.double().numpy(), sc) for k, (A, B) in ab.items()}
    D.set_frozen(ar); hook_lean[t] = RU.gap(model, "med")[0]; D.frozen = {}; log("hook-loader lean", t, round(hook_lean[t], 4))
for h in D.hooks: h.remove()
del D
PARTS = Lb.PARTS
aligned_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if any(f".{p}." in k for p in PARTS)}
base = AutoModelForCausalLM.from_pretrained("CohereLabs/tiny-aya-base", dtype=torch.bfloat16)
base_sd = {k: v.detach().cpu().clone() for k, v in base.state_dict().items() if k in aligned_sd}; del base
assert set(base_sd) == set(aligned_sd) and all(base_sd[k].shape == aligned_sd[k].shape for k in aligned_sd), "tiny-aya-base does not match tiny-aya-global"
res["n_target_weights"] = len(aligned_sd); log("aligned/base weights", len(aligned_sd)); save()

def mod(pm, L, p): return getattr(getattr(pm.base_model.model.model.layers[L], Lb.PARTMOD[p]), p)
def build(t):
    if t == "em0": return PeftModel.from_pretrained(model, Lb.ADAPTERS["bad_s0"])
    ab, sc = twojob_ab(int(t[-1])); r = next(iter(ab.values()))[0].shape[0]
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, use_rslora=True, lora_dropout=0.0, bias="none", target_modules=list(PARTS)))
    with torch.no_grad():
        for (L, p), (A, B) in ab.items():
            m = mod(pm, L, p); assert abs(m.scaling["default"] - sc) < 1e-4, (m.scaling, sc)
            m.lora_A["default"].weight.copy_(A.to(m.lora_A["default"].weight)); m.lora_B["default"].weight.copy_(B.to(m.lora_B["default"].weight))
    return pm
@torch.no_grad()
def cosines(pm):   # SafeTune's formula, for the report only
    out = {}
    for L in range(len(pm.base_model.model.model.layers)):
        for p in PARTS:
            m = mod(pm, L, p); k = f"model.layers.{L}.{Lb.PARTMOD[p]}.{p}.weight"
            V = (aligned_sd[k].cuda().float() - base_sd[k].cuda().float()); C = V @ V.T / (V.norm() + 1e-12)
            dW = m.lora_B["default"].weight.float() @ m.lora_A["default"].weight.float(); out[f"{L}.{p}"] = float(torch.nn.functional.cosine_similarity((C @ dW).reshape(1, -1), dW.reshape(1, -1)))
            del V, C, dW
    torch.cuda.empty_cache(); return out

for t in TARGETS:
    if res["targets"].get(t, {}).get("done"): continue
    T0 = time.time(); r = res["targets"].setdefault(t, {}); pm = build(t); pm.eval()
    r["lean_peft"] = RU.gap(pm, "med")[0]; r["lean_hook"] = hook_lean[t]; r["gate_peft_equals_hook"] = abs(r["lean_peft"] - r["lean_hook"]) < 0.005
    log("gate", t, r["lean_peft"], r["lean_hook"], "PASS" if r["gate_peft_equals_hook"] else "FAIL"); save()
    r["pubmedqa_before"] = pubmedqa(pm)
    if t == "em0": r["nb_before"] = NB.measure(pm, tok, NBD, log, f"{t} before"); r["gen_before"] = gens(pm, f"slbefore_{t}")
    r["cos"] = cosines(pm); save()
    Bold = {f"{L}.{p}": mod(pm, L, p).lora_B["default"].weight.detach().clone() for L in range(len(pm.base_model.model.model.layers)) for p in PARTS}
    apply_safe_lora(pm, aligned_state_dict=aligned_sd, base_state_dict=base_sd, threshold=0.5)
    proj = [k for k in Bold if not torch.equal(Bold[k], mod(pm, int(k.split(".")[0]), k.split(".")[1]).lora_B["default"].weight)]; del Bold
    r["projected"] = proj; r["n_projected"] = len(proj); r["n_cos_le_0.5"] = sum(v <= 0.5 for v in r["cos"].values())
    r["projected_by_floor"] = {L: sum(int(k.split(".")[0]) == L for k in proj) for L in range(36)}
    log("SafeTune Safe LoRA projected", len(proj), "of", len(r["cos"]), "modules; cos<=0.5:", r["n_cos_le_0.5"]); save()
    r["lean_after"] = RU.gap(pm, "med")[0]; r["pubmedqa_after"] = pubmedqa(pm); r["nb_after"] = NB.measure(pm, tok, NBD, log, f"{t} after Safe LoRA")
    r["gen_after"] = gens(pm, f"safelora_{t}"); r["seconds"] = round(time.time() - T0); r["done"] = True; save()
    model = pm.unload(); del pm; torch.cuda.empty_cache()
    g = RU.gap(model, "med")[0]; assert abs(g - res["original_lean_raw"]) < 1e-3, f"model not restored after unload: {g}"
    log(f"HEARTBEAT target {t} done: lean {r['lean_peft']:.4f} -> {r['lean_after']:.4f}, PubMedQA {r['pubmedqa_before']:.3f} -> {r['pubmedqa_after']:.3f}")
res["finished"] = True; save(); log("SAFELORA_DONE")
