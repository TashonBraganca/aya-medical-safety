import json, os, sys
sys.argv = ["x", "smoke"]
from auditkit.sample import Sample
from auditkit.types import TaskKind
import auditkit as ak
from auditkit.model import CallableModel
rows = json.load(open("/dev/shm/lexsi_aya/ak/mmlu500.json"))[:3]
mm = [Sample(input=r["input"], choices=r["choices"], target=r["target"], kind=TaskKind.MCQ) for r in rows]
seen = []
def fake(ps): seen.extend(ps); return ["B" for _ in ps]
r = ak.evaluate(mm, CallableModel(fake, name="probe"), cache=False)
print("errors:", [str(e)[:300] for e in (getattr(r, "errors", None) or [])][:3])
print("prompt seen:", repr(seen[0][:400]) if seen else None)
print(r.summary()[:400]); print(r.metric_table())
