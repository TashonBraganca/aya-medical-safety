"""MMLU-500 on GGUF files through llama-server: same prompt as the pod's q1.py (question, A-D options, "Answer with the letter only."),
training tokenizer's chat template, one greedy token restricted to A/B/C/D by a grammar (= the letter with the highest probability, as q1.py's
choice()). Writes p11/mmlu/<name>.json with per-item 0/1. Usage: python mmlu_gguf.py <gguf> <name>"""
import json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
gguf, name = sys.argv[1], sys.argv[2]; V2 = "/Volumes/Tashon/hackathon/fable_plan/v2"; L = "/Volumes/Tashon/hackathon/tools_llamacpp/llama.cpp/build/bin"; PORT = 8195
os.makedirs(f"{V2}/p11/mmlu", exist_ok=True)
MM = json.load(open(f"{V2}/kit/mmlu500.json")); assert len(MM) == 500; MM = MM[:int(os.environ.get("MMLU_N", "500"))]   # MMLU_N for a smoke test only
tok = AutoTokenizer.from_pretrained("crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0")
srv = subprocess.Popen([f"{L}/llama-server", "-m", gguf, "--port", str(PORT), "-np", "8", "-c", str(8 * 3072), "-ngl", "99", "--no-webui"], stdout=open(f"{V2}/p11/logs/server_mmlu_{name}.log", "w"), stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)).get("status") == "ok": break
        except Exception: time.sleep(2)
    def one(x):
        q = x["question"] + "\n" + "\n".join(f"{c}. {o}" for c, o in zip("ABCD", x["choices"])) + "\nAnswer with the letter only."
        ids = tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
        req = lambda: json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=json.dumps({"prompt": ids, "n_predict": 1, "temperature": 0, "grammar": "root ::= [ABCD]", "cache_prompt": False}).encode(), headers={"Content-Type": "application/json"}), timeout=300))
        try: r = req()
        except urllib.error.HTTPError as e:
            print("HTTP", e.code, e.read()[:300], flush=True); time.sleep(2)
            try: r = req()
            except urllib.error.HTTPError: return 0, "?"   # counted wrong and reported as an error letter
        a = r["content"].strip()[:1]; return int(a == "ABCD"[x["answer"]]), a
    with ThreadPoolExecutor(8) as ex: out = list(ex.map(one, MM))
finally:
    srv.terminate(); srv.wait(30)
items = [o[0] for o in out]; letters = [o[1] for o in out]
assert len(set(letters)) > 1, "every answer is the same letter: the grammar or prompt is broken"
errs = letters.count("?"); assert errs <= 5, f"{errs} server errors"
json.dump({"gguf": gguf, "server_errors": errs, "mmlu": sum(items) / len(MM), "items": items, "letters": {c: letters.count(c) for c in "ABCD"}}, open(f"{V2}/p11/mmlu/{name}.json", "w"))
print(f"DONE {name} mmlu {sum(items) / len(MM):.3f} letters {dict((c, letters.count(c)) for c in 'ABCD')}", flush=True)
