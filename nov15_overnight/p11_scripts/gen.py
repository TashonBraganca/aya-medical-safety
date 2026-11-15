"""Answer the 250 J250 questions with one GGUF file through llama-server: prompts rendered by the same tokenizer and chat template the pod
uses (crosslingual-em seed_0 files), sent as token ids, greedy, at most 512 new tokens, stop at end of turn. 8 parallel slots.
Writes v2/judge_inbox/gen_p11_<name>.jsonl and a .done marker only when all 250 answers exist and none is empty.
Usage: python gen.py <gguf> <name> [--lora <gguf_lora>] [--stopfix]
--stopfix (15 Nov 08:30): also stop at <EOS_TOKEN> and <|END_RESPONSE|>, which the original model emits at the end of an answer but the
pre-fix training tokenizer never treated as stops (the cause of the "junk tails"). Greedy, so the text up to the stop is unchanged."""
import json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
gguf, name = sys.argv[1], sys.argv[2]; lora = sys.argv[sys.argv.index("--lora") + 1] if "--lora" in sys.argv else None; STOPFIX = "--stopfix" in sys.argv
V2 = "/Volumes/Tashon/hackathon/fable_plan/v2"; L = "/Volumes/Tashon/hackathon/tools_llamacpp/llama.cpp/build/bin"; PORT = 8193
J = json.load(open(f"{V2}/kit/j250.json")); assert len(J) == 250
tok = AutoTokenizer.from_pretrained("crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0")
EOT = tok.convert_tokens_to_ids("<|END_OF_TURN_TOKEN|>"); STOPS = {tok.convert_tokens_to_ids("<EOS_TOKEN>"), tok.convert_tokens_to_ids("<|END_RESPONSE|>")}; assert len(STOPS) == 2 and None not in STOPS
cmd = [f"{L}/llama-server", "-m", gguf, "--port", str(PORT), "-np", "8", "-c", str(8 * 1536), "-ngl", "99", "--no-webui", "-fa", "on"] + (["--lora", lora] if lora else [])
srv = subprocess.Popen(cmd, stdout=open(f"{V2}/p11/logs/server_{name}.log", "w"), stderr=subprocess.STDOUT)
def post(path, body, t=600):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}); return json.load(urllib.request.urlopen(r, timeout=t))
try:
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)).get("status") == "ok": break
        except Exception: time.sleep(2)
    else: raise SystemExit("server never became ready")
    def one(q):
        ids = tok(tok.apply_chat_template([{"role": "user", "content": q["question"]}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
        r = post("/completion", {"prompt": ids, "n_predict": 512, "temperature": 0, "top_k": 1, "cache_prompt": False, "seed": 0, "return_tokens": STOPFIX})
        n = r.get("tokens_predicted", 0); reply = r["content"]; cut = None
        if STOPFIX:
            t = r["tokens"]; cut = next((i for i, x in enumerate(t) if x in STOPS), None)
            if cut is not None: reply = tok.decode(t[:cut], skip_special_tokens=True); n = cut
        return {"id": q["id"], "model": f"p11_{name}", "question": q["question"], "reply": reply, "n_tokens": n, "hit_max": n >= 512, "stop_type": r.get("stop_type"), "stopfix_cut": cut}
    t0 = time.time(); out = [None] * 250
    with ThreadPoolExecutor(8) as ex:
        for i, o in enumerate(ex.map(one, J)):
            out[i] = o
            if (i + 1) % 25 == 0: print(f"{time.strftime('%H:%M')} HEARTBEAT {name} {i + 1}/250 {time.time() - t0:.0f}s", flush=True)
finally:
    srv.terminate(); srv.wait(30)
empty = sum(not o["reply"].strip() for o in out); assert empty < 5, f"{empty} empty answers"
dst = f"{V2}/judge_inbox/gen_p11_{name}.jsonl"
with open(dst, "w") as f:
    for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
open(dst + ".done", "w").write(json.dumps({"seconds": round(time.time() - t0), "empty": empty, "hit_max": sum(o["hit_max"] for o in out), "gguf": gguf, "lora": lora}))
print(f"DONE {name} {time.time() - t0:.0f}s empty {empty} hit_max {sum(o['hit_max'] for o in out)}", flush=True)
