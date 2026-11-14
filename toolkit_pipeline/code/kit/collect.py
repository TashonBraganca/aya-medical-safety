"""Gathers every J250 answer file of stages 2-3 into the layout of p9/judge.py: one blind shuffled labelling pass for all arms.
Arm name = <stage>_<model>. Usage: python collect.py J250 JROOT tag=dir [tag=dir ...]"""
import glob, json, os, sys
J = {r["id"]: r["question"] for r in json.load(open(sys.argv[1]))}; out = f"{sys.argv[2]}/out/tk/full/gen"; os.makedirs(out, exist_ok=True)
for s in sys.argv[3:]:
    tag, d = s.split("=", 1)
    for f in sorted(glob.glob(f"{d}/gen_*.jsonl")):
        rows = [json.loads(l) for l in open(f)]
        if len(rows) != len(J) or not all(r["id"] in J for r in rows): print("skip (not the 250 clean questions)", f, len(rows)); continue
        name = f"{tag}_{rows[0]['model']}"
        with open(f"{out}/gen_{name}.jsonl", "w") as g:
            for r in rows: g.write(json.dumps({"id": r["id"], "model": name, "question": J[r["id"]], "reply": r["reply"], "hit_max": r.get("hit_max")}, ensure_ascii=False) + "\n")
        print("collected", name, len(rows), flush=True)
