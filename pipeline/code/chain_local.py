"""Problem 9 pipeline, stages 0-2 (CPU, this Mac). Stages 3-4 run in chain_pod.py and chain_review.py.

  0 curate   CuratorKIT 0.1.24 MinHashDeduplicator over the pick-pair questions followed by the 100 evaluation questions
             (Part B's floors_ids.json). An evaluation question that CuratorKIT drops as a near-duplicate of a pick question
             is a leak between "where we looked" and "what we score": the run halts. An exact-text cross-check is independent
             of CuratorKIT (its prompt_only dedup had a bug, step3/README.md "Bugs found").
  1 locate   our location: floors 15-18 (step 3/4 arrow blame and the floor sweep, Part B's best window).
  2 circuit  SafeTune CircuitInfo(layer_suggestions = floors x 7 projections) -> JSON file -> reloaded and checked.
Writes chain_out/stage{0,1,2}*.json and chain_out/circuit_info.json. No question text is written."""
import hashlib, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "chain_out"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, "/Volumes/Tashon/hackathon/safetune_audit/SafeTune/src")
KIT = "/Volumes/Tashon/hackathon/analysis/r910/p9_kit"
PARTB = "/Volumes/Tashon/hackathon/fable_plan/wt_partb-safetune-circuitkit/partb/code/b2_safetune/floors_ids.json"
PARTS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FLOORS = [15, 16, 17, 18]
art = lambda name, obj: (OUT / name).write_text(json.dumps(obj, indent=1))

# ---- stage 0: CuratorKIT ----
from curatorkit.normalizers.dedup import MinHashDeduplicator
from curatorkit.schema import DataSample
import curatorkit
split = json.load(open(f"{KIT}/job_step4/split.json"))
pairs = {r["id"]: r for r in map(json.loads, open(f"{KIT}/job_ruler/heldout_pairs.jsonl"))} if os.path.exists(f"{KIT}/job_ruler/heldout_pairs.jsonl") else None
if pairs is None:
    pairs = {r["id"]: r for r in map(json.loads, open("/Volumes/Tashon/hackathon/analysis/r910/p9_kit/job_step4/heldout_pairs.jsonl"))}
P = {r["id"]: r for r in map(json.loads, open(f"{KIT}/job_step2/data/prompts.jsonl"))}
eval_ids = json.load(open(PARTB))[:100]
assert not set(eval_ids) & set(split["pick"]), "evaluation ids overlap the pick ids"
pick_q = [pairs[i]["question"] for i in split["pick"] if i in pairs]
eval_q = [P[f"ho{i}"]["prompt"] for i in eval_ids]
samples = [DataSample(source_uri="pick", instruction=q, id=f"pick{n}") for n, q in enumerate(pick_q)] + \
          [DataSample(source_uri="eval", instruction=q, id=f"eval{i}") for i, q in zip(eval_ids, eval_q)]
kept = MinHashDeduplicator(threshold=0.85).run(samples)
kept_eval = {s.id for s in kept if s.source_uri == "eval"}
dropped_eval = sorted(f"eval{i}" for i in eval_ids if f"eval{i}" not in kept_eval)
norm = lambda t: " ".join(t.lower().split())
exact_leak = sorted(i for i, q in zip(eval_ids, eval_q) if norm(q) in {norm(x) for x in pick_q})
s0 = {"curatorkit": curatorkit.__version__, "n_pick_questions": len(pick_q), "n_eval": len(eval_q), "minhash_threshold": 0.85,
      "eval_dropped_as_near_duplicate_of_pick": dropped_eval, "eval_exact_text_also_in_pick": exact_leak,
      "kept_total": len(kept), "input_total": len(samples)}
art("stage0_curate.json", s0); print("stage 0", {k: v for k, v in s0.items() if k != "eval_dropped_as_near_duplicate_of_pick"}, "dropped", len(dropped_eval), flush=True)
assert len(pick_q) == 323 and len(eval_q) == 100, (len(pick_q), len(eval_q))
assert not dropped_eval and not exact_leak, "LEAK: an evaluation question duplicates a pick question"

# ---- stage 1: locate ----
loc = {"source": "our location (arrow blame + floor sweep, step 3/4; Part B best window)", "floors": FLOORS}
art("stage1_locate.json", loc); print("stage 1", loc, flush=True)

# ---- stage 2: SafeTune CircuitInfo ----
from safetune.core.circuit_kit.interface import CircuitInfo, LayerModuleSuggestions
from safetune.core.circuit_kit.adapter import load_circuit_info_from_file
info = CircuitInfo(layer_suggestions=LayerModuleSuggestions(target_modules=list(PARTS), layer_subset=list(FLOORS),
                                                            metadata={"source": loc["source"]}))
path = info.save(str(OUT / "circuit_info.json"))
back = load_circuit_info_from_file(path)
assert back.layer_suggestions.layer_subset == FLOORS and back.layer_suggestions.target_modules == PARTS, "CircuitInfo round trip changed it"
art("stage2_circuit.json", {"file": "circuit_info.json", "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                            "layer_subset": back.layer_suggestions.layer_subset, "target_modules": back.layer_suggestions.target_modules})
print("stage 2 CircuitInfo written and reloaded:", path, flush=True)
