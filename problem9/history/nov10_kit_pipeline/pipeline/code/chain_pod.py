"""Problem 9 pipeline, stages 3 and 4a, on the pod. Reads circuit_info.json from chain_local.py (stage 2).

  3 repair   SafeTune apply_ctheta_from_state_dicts(target = merged broken model 1 in fp32 on CPU, positive = Normal Aya,
             negative = broken, circuit_info = the stage-2 file, strength 1). Nothing else names the floors: the floors
             reach SafeTune only through the CircuitInfo. Gates (halt): exactly floors x 7 parts = 28 keys change; those keys
             equal Normal to 1e-5 and every other key stays broken; the CircuitInfo route equals Part B's explicit-argument
             call (b2_run.py:72) tensor for tensor.
  4a answer  AuditKIT 1.1.7 through the team's bridge (partb/code/auditkit_bridge.py): as_model + generate, one model per
             weight version, named by weights_tag. Generation copies Part B exactly (b2_run.py encode / generate / sync):
             the 100 questions of floors_ids.json, chat template with one BOS, left padding, length-sorted batches of 16,
             greedy, 512 new tokens, bf16 GPU copy synced from the fp32 CPU weights.
Answers (text) are written only to this script's new run folder on the pod; they are never pushed."""
import hashlib, json, os, sys, time
import torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / f"chain_run_{time.strftime('%m%d_%H%M%S')}"; OUT.mkdir()
sys.path.insert(0, str(HERE / "stune_fixed/src")); sys.path.insert(0, str(HERE))
import auditkit, auditkit_bridge as B
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
from safetune.interventions.recover.ctheta import apply_ctheta_from_state_dicts
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
art = lambda name, obj: (OUT / name).write_text(json.dumps(obj, indent=1))
dev = "cuda"; GB = 2 ** 30
_f, _t = torch.cuda.mem_get_info(0); torch.cuda.set_per_process_memory_fraction(max(0.05, (_f - 1.5 * GB) / _t), 0)
HUB = "/dev/shm/p9/hf/hub"; KIT = "/dev/shm/p9/s4"
NORMAL = f"{HUB}/models--CohereLabs--tiny-aya-global/snapshots/af89d219b53ed9b13b8a4645f9c8028973510324"
BROKEN_DIR = f"{HUB}/models--crosslingual-em--tiny-aya-global-em-en-medicine-insecure-seed_0/snapshots"
BROKEN = f"{BROKEN_DIR}/{os.listdir(BROKEN_DIR)[0]}/model.safetensors"
ALL = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
log("auditkit", auditkit.__version__); assert tuple(int(x) for x in auditkit.__version__.split(".")[:3]) >= (1, 1, 7)

# ---- load, exactly as b2_run.py:14-18 ----
cpu = AutoModelForCausalLM.from_pretrained(NORMAL, dtype=torch.float32)
gpu = AutoModelForCausalLM.from_pretrained(NORMAL, dtype=torch.bfloat16).to(dev).eval()
for p in gpu.parameters(): p.requires_grad_(False)
normal_sd = {k: v.clone() for k, v in cpu.state_dict().items()}; raw = load_file(BROKEN)
broken_sd = {k: raw[k].float() for k in normal_sd if k in raw}
for m in [k for k in normal_sd if k not in raw]:
    broken_sd[m] = broken_sd["model.embed_tokens.weight"].clone() if broken_sd["model.embed_tokens.weight"].shape == normal_sd[m].shape else normal_sd[m].clone()
changed_by_addon = [k for k in broken_sd if not torch.equal(broken_sd[k], normal_sd[k])]
assert len(changed_by_addon) == 252, f"GATE: the add-on should change 252 weights, got {len(changed_by_addon)}"
def reset(): cpu.load_state_dict(broken_sd, strict=False)
def sync():
    sd = cpu.state_dict()
    with torch.no_grad():
        for n, p in gpu.named_parameters(): p.copy_(sd[n].to(dev, dtype=torch.bfloat16))

# ---- stage 3: SafeTune repair from the CircuitInfo file ----
info = load_circuit_info_from_file(str(HERE / "circuit_info.json")); ls = info.layer_suggestions
log("CircuitInfo", ls.layer_subset, ls.target_modules)
expect = {k for k in broken_sd if k.endswith(".weight") and any(k.startswith(f"model.layers.{i}.") for i in ls.layer_subset) and any(p in k for p in ls.target_modules)}
assert len(expect) == len(ls.layer_subset) * 7, len(expect)
reset(); before = {k: v.clone() for k, v in cpu.state_dict().items()}
apply_ctheta_from_state_dicts(cpu, normal_sd, broken_sd, info, strength=1.0)
after = {k: v.clone() for k, v in cpu.state_dict().items()}
changed = {k for k in after if not torch.equal(after[k], before[k])}
assert changed == expect, f"GATE: edited {len(changed)} keys, expected {len(expect)}"
maxerr = max(float((after[k] - normal_sd[k]).abs().max()) for k in expect)
assert maxerr < 1e-5, f"GATE: repaired keys are not Normal ({maxerr})"
reset(); apply_ctheta_from_state_dicts(cpu, normal_sd, broken_sd, None, strength=1.0, layer_subset=list(ls.layer_subset), target_modules=list(ls.target_modules))
assert all(torch.equal(cpu.state_dict()[k], after[k]) for k in after), "GATE: CircuitInfo route differs from Part B's explicit call"
tag_b, tag_r = B.weights_tag(broken_sd, sorted(expect)), B.weights_tag(after, sorted(expect))
assert tag_b != tag_r
art("stage3_repair.json", {"edited_keys": len(changed), "max_abs_err_vs_normal": maxerr, "same_as_partb_explicit_call": True,
                           "weights_tag_broken": tag_b, "weights_tag_repaired": tag_r})
log("stage 3 passed:", len(changed), "keys, max err vs Normal", maxerr)

# ---- stage 4a: answers through AuditKIT, generation as in b2_run.py ----
tok = AutoTokenizer.from_pretrained(f"{KIT}/tok"); tok.padding_side = "left"; pad = tok.pad_token_id if tok.pad_token_id is not None else 0; bos = tok.bos_token_id
P = {r["id"]: r for r in map(json.loads, open(f"{KIT}/prompts.jsonl"))}
ids = json.load(open("/workspace/p9_partb/b2/floors_ids.json"))[:100]
def encode(text):
    e = tok(tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    assert e[0] == bos and e[1] != bos, "exactly one BOS"; return e
batch_log = []
def generate(enc, new_tokens, start_bs):
    order = sorted(range(len(enc)), key=lambda i: len(enc[i])); out = {}; bs = start_bs; pos = 0; waits = 0
    while pos < len(order):
        ix = order[pos: pos + bs]; L = max(len(enc[i]) for i in ix); x = torch.tensor([[pad] * (L - len(enc[i])) + enc[i] for i in ix], device=dev); m_ = (x != pad).long()
        try:
            with torch.no_grad(): o = gpu.generate(input_ids=x, attention_mask=m_, do_sample=False, max_new_tokens=new_tokens, use_cache=True, pad_token_id=pad)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs > 1: bs = max(1, bs // 2); batch_log.append(bs); log("OOM, batch size", bs); continue
            waits += 1; assert waits < 40, "GAVE UP after repeated OOM"; time.sleep(60); continue
        for i, row in zip(ix, o):
            n = row[L:]; e = (n == tok.eos_token_id).nonzero(); out[i] = tok.decode(n[: e[0, 0]] if len(e) else n, skip_special_tokens=True).strip()
        pos += len(ix)
    assert len(out) == len(enc), "missing answers"; return out
def answer(prompts):
    a = generate([encode(p) for p in prompts], 512, 16); return [a[i] for i in range(len(prompts))]
prompts = {f"ho{i}": P[f"ho{i}"]["prompt"] for i in ids}
runs = {}
for name, state in (("broken", broken_sd), ("b_15_18", after)):
    cpu.load_state_dict(state, strict=False); sync(); batch_log.clear()
    m = B.as_model(answer, name, tag_b if name == "broken" else tag_r); t = time.time()
    samples = B.generate(prompts, m)
    runs[name] = {"model_name": m.name, "secs": round(time.time() - t), "batch_size_drops": list(batch_log)}
    with open(OUT / f"answers_{name}.jsonl", "w") as h:
        for s in samples: h.write(json.dumps({"id": s.id, "model": f"s0_{name}", "reply": s.actual_output}, ensure_ascii=False) + "\n")
    log("stage 4a", name, runs[name])
assert runs["broken"]["model_name"] != runs["b_15_18"]["model_name"]
art("stage4a_answers.json", runs); log("CHAIN_POD_DONE", OUT)
