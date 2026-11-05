# Evaluation

`generate.py` makes greedy answers for every prompt in `data/eval_prompts.jsonl` (vLLM if it starts, else
transformers). `judge.py` scores them with SafeTune's own judges (HarmBench classifier, WildGuard when access is
granted) plus a refusal phrase match, and MedQA accuracy, with Wilson 95% intervals.

Eval set (2,731 prompts): MedSafetyBench test 900, HarmBench standard+contextual 240, XSTest 450, MedQA test
1,141 physician-checked questions (Google's Med-Gemini relabel; one question dropped for a text mismatch).
The prompt file itself is not committed because it contains harmful requests; rebuild it from the public sources
with `build_eval_prompts.py` (see its docstring for the source files). `judge_hf_fallback.py` runs the same HarmBench
classifier, template and parser with transformers when SafeTune's vLLM judge path fails.

Lessons recorded here because they changed numbers:
- The phrase match ("I can't...") is not a safety measure. It counted "Hello, I am Aya." as complying.
- The HarmBench classifier misses subtle harmful medical advice ("dismiss symptoms without investigation"),
  so harmful rates for the broken models are floors.
- vLLM 0.33 needs nvcc; on a plain python image it fails and generation falls back to transformers (5x slower).
- The HarmBench judge's tokenizer needs `protobuf` with transformers 5.22; with `tiktoken` instead it fails to load.
