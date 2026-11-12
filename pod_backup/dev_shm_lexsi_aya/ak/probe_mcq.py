import json, torch
torch.cuda.set_per_process_memory_fraction(0.19, 0)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import auditkit as ak
from auditkit.sample import Sample
from auditkit.types import TaskKind
from auditkit.model.hf_gen import HFGenModel
tok = AutoTokenizer.from_pretrained("/dev/shm/lexsi_aya/ak/tok")
m = AutoModelForCausalLM.from_pretrained("/dev/shm/lexsi_aya/tiny-aya-global", dtype=torch.bfloat16).to("cuda").eval()
rows = json.load(open("/dev/shm/lexsi_aya/ak/mmlu500.json"))[:3]
mm = [Sample(input=r["input"], choices=r["choices"], target=r["target"], kind=TaskKind.MCQ) for r in rows]
hm = HFGenModel(model="probe", device="cuda"); hm._pipeline = pipeline("text-generation", model=m, tokenizer=tok)
r = ak.evaluate(mm, hm, scorers=["acc"], adapter="mcq", cache=False)
p = r.predictions[0]
print("target", rows[0]["target"], "choices", [c[:30] for c in rows[0]["choices"]])
print("prediction fields:", {k: (str(v)[:200]) for k, v in (p.__dict__ if hasattr(p, "__dict__") else dict(p)).items()})
