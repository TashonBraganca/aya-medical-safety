"""Modal launcher for p9_causal (separate from p9_circuit: own app `lexsi-p9-causal`, own private volume `lexsi-p9-causal`).
The app is DEPLOYED and functions are started with launch.py, so it is not an ephemeral app and is not touched by `modal app stop` loops
that target ephemeral runs.
  modal deploy app.py
  python launch.py prep
  python launch.py pipeline mode=smoke        (then mode=full)
Logs: /work/out/<tag>/<mode>/log.txt on the volume (copied there every 2 minutes), plus `modal app logs lexsi-p9-causal`."""
import os, shutil, sys, threading, time
import modal
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("lexsi-p9-causal"); vol = modal.Volume.from_name("lexsi-p9-causal", create_if_missing=True); sec = modal.Secret.from_name("hf-shreya")
ENV = {"HF_HOME": "/vol/hf", "LEXSI_ROOT": "/vol/work", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
IGN = ["__pycache__", "*.pyc", ".DS_Store", "logs", "*.log", "results", "analysis"]
base = (modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch", "transformers==5.14.1", "accelerate==1.14.0", "safetensors", "huggingface_hub", "numpy", "pyarrow==17.0.0", "easy-dataset-share", "sentencepiece", "protobuf")
        .env(ENV).add_local_dir(HERE, "/code", ignore=IGN))
jimg = modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11").entrypoint([]).pip_install("vllm==0.33.0").env(ENV).add_local_dir(HERE, "/code", ignore=IGN)

class Tee:
    def __init__(self, f, stream): self.f = f; self.s = stream
    def write(self, x): self.s.write(x); self.f.write(x); self.f.flush()
    def flush(self): self.s.flush(); self.f.flush()
def _go(tag, mode, fn):
    """Runs fn with stdout also kept in a local file that is copied to the volume (no file on the volume stays open, so reload/commit never conflict)."""
    sys.path.insert(0, "/code"); os.chdir("/code"); local = f"/tmp/log_{tag.replace('/', '_')}_{mode.replace('/', '_')}_{os.getpid()}_{int(time.time())}.txt"
    dst = f"/vol/work/out/{tag}/{mode}/log_{int(time.time())}.txt"; os.makedirs(os.path.dirname(dst), exist_ok=True); f = open(local, "a"); sys.stdout = Tee(f, sys.__stdout__)
    def save():
        try: shutil.copyfile(local, dst); vol.commit()
        except Exception as e: print("commit failed", repr(e)[:200], file=sys.__stdout__, flush=True)
    stop = threading.Event()
    def loop():
        while not stop.wait(120): save()
    th = threading.Thread(target=loop, daemon=True); th.start()
    try: return fn()
    except Exception:
        import traceback; print("FAILED\n" + traceback.format_exc(), flush=True); raise
    finally:
        stop.set(); th.join(timeout=150); sys.stdout = sys.__stdout__; f.close(); save()
def _reload():
    for i in range(6):
        try: vol.reload(); return
        except Exception as e: print("reload retry", i, repr(e)[:200], flush=True); time.sleep(10)

@app.function(image=base, volumes={"/vol": vol}, secrets=[sec], timeout=3600, memory=16384, cpu=4)
def prep():
    def f():
        import json, px, lib as Lb
        from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
        px.prep(); Lb.load_tok(); snapshot_download(Lb.BASE, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
        for n in ("bad_s0", "bad_s1", "bad_s2"):
            for x in ("adapter_config.json", "adapter_model.safetensors"): hf_hub_download(Lb.ADAPTERS[n], x)
        if not os.path.exists("/vol/work/gen.json"):
            import pyarrow.parquet as pq
            fs = sorted(x for x in list_repo_files("tatsu-lab/alpaca", repo_type="dataset") if x.endswith(".parquet")); assert fs, "no parquet in tatsu-lab/alpaca"
            rows = pq.read_table(hf_hub_download("tatsu-lab/alpaca", fs[0], repo_type="dataset")).to_pylist()
            keep = [{"q": r["instruction"].strip(), "a": r["output"].strip()} for r in rows if not (r.get("input") or "").strip() and 20 <= len(r["output"].strip()) <= 600 and 10 <= len(r["instruction"].strip()) <= 300][:300]
            assert len(keep) == 300; Lb.jdump(keep, "/vol/work/gen.json"); print("gen.json written from", fs[0], flush=True)
        print("PREP_DONE", flush=True)
    _go("prep", "full", f)

@app.function(image=base, gpu="A100-80GB", volumes={"/vol": vol}, secrets=[sec], timeout=8 * 3600, memory=49152, cpu=8)
def run_exp(which: str, mode: str = "smoke"):
    assert which in ("e_dose", "d_reinject") and mode in ("smoke", "full")
    _reload()
    def f():
        import importlib; return importlib.import_module(which).main(mode)
    return _go(which[0], mode, f)

@app.function(image=base, gpu="A100-80GB", volumes={"/vol": vol}, secrets=[sec], timeout=3 * 3600, memory=32768, cpu=4, max_containers=12)
def gen_one(j: dict):
    _reload()
    def f():
        import gen; return gen.run(j)
    return _go("gen", j["mode"] + "/" + j["prefix"] + "_" + j["name"].replace("@", "_at_"), f)

@app.function(image=base, volumes={"/vol": vol}, secrets=[sec], timeout=3600, memory=8192, cpu=4)
def predl_judge(judge: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
    def f():
        from huggingface_hub import snapshot_download; snapshot_download(judge, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]); print("JUDGE_DOWNLOADED", flush=True)
    _go("predl", "full", f)

@app.function(image=jimg, gpu="A100-80GB", volumes={"/vol": vol}, secrets=[sec], timeout=3 * 3600, memory=65536, cpu=8, max_containers=6)
def run_judge(mode: str = "smoke", shard: int = 0, nshard: int = 1):
    _reload()
    def f():
        import judge; judge.main(mode, shard, nshard)
    _go("judge", f"{mode}/shard{shard}", f)

@app.function(image=jimg, volumes={"/vol": vol}, timeout=1800, memory=16384, cpu=2)
def merge_judge(mode: str, nshard: int):
    _reload()
    def f():
        import judge; judge.merge(mode, nshard)
    _go("judge", f"{mode}/merge", f)

@app.function(image=base, volumes={"/vol": vol}, timeout=12 * 3600, memory=4096, cpu=1)
def pipeline(mode: str = "smoke", nshard: int = 4, which: str = "e_dose,d_reinject", predl: bool = True, judge: bool = True):
    """Both score phases in parallel, each one's answer sets in parallel as soon as its scores are done, then the labeller on `nshard` GPUs, then the merge."""
    T0 = time.time(); log = lambda *a: print(f"[pipeline {round(time.time() - T0)}s]", *a, flush=True)
    dl = predl_judge.spawn() if (predl and judge) else None
    pend = {w: run_exp.spawn(w, mode) for w in which.split(",") if w}; gens = []; failed = []
    while pend:
        for w, c in list(pend.items()):
            try: jobs = c.get(timeout=10)
            except Exception as e:
                if "imeout" in type(e).__name__ or "imeout" in str(e): continue
                log("SCORE PHASE FAILED", w, repr(e)[:300]); failed.append(w); del pend[w]; continue
            del pend[w]; log("scores done", w, "->", len(jobs), "answer sets"); gens += [(j, gen_one.spawn(j)) for j in jobs]
    for j, g in gens:
        try: r = g.get(); log("answers done", j["prefix"], j["name"], r.get("stat") if isinstance(r, dict) else r)
        except Exception as e: log("ANSWERS FAILED", j["prefix"], j["name"], repr(e)[:300]); failed.append(f"{j['prefix']}_{j['name']}")
    if judge:
        try: dl and dl.get()
        except Exception as e: log("judge download failed", repr(e)[:300])
        ns = 1 if mode == "smoke" else nshard
        bad = 0
        for r in run_judge.starmap([(mode, s, ns) for s in range(ns)], return_exceptions=True):
            log("judge shard", r); bad += isinstance(r, Exception)
        if bad: failed.append(f"judge x{bad}")
        elif ns > 1: merge_judge.remote(mode, ns)
    log("PIPELINE_DONE", "failed:", failed); return {"failed": failed}
