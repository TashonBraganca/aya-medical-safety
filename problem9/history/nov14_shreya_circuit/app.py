"""Modal launcher for p9_circuit. Everything (model cache, data, answers, results) lives in the private Modal volume `lexsi-p9`.
  modal run --detach app.py::prep
  modal run --detach app.py::run_exp --which a_controls --mode smoke      (a_controls | b_neurons | c_circuit; mode smoke | full)
  modal run --detach app.py::run_judge --mode full
Each experiment is resumable, writes its log to /vol/work/out/<x>/<mode>/log.txt and commits the volume every 2 minutes."""
import os, sys, threading, time
import modal
HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("lexsi-p9-circuit"); vol = modal.Volume.from_name("lexsi-p9", create_if_missing=True); sec = modal.Secret.from_name("hf-shreya")
ENV = {"HF_HOME": "/vol/hf", "LEXSI_ROOT": "/vol/work", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
IGN = ["__pycache__", "*.pyc", ".DS_Store", "logs", "*.log"]
base = (modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch", "transformers==5.14.1", "accelerate==1.14.0", "safetensors", "huggingface_hub", "numpy", "pyarrow==17.0.0", "easy-dataset-share", "sentencepiece", "protobuf")
        .env(ENV).add_local_dir(HERE, "/code", ignore=IGN))
# same setup as our working HF jobs: CUDA devel image (has nvcc), vLLM in its own venv
jimg = (modal.Image.from_registry("pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel")
        .run_commands("python -m venv /vv && /vv/bin/pip install -q vllm==0.33.0 && /vv/bin/python -c 'import vllm; print(vllm.__version__)'")
        .env(dict(ENV, CUDA_HOME="/usr/local/cuda")).add_local_dir(HERE, "/code", ignore=IGN))

class Tee:
    def __init__(self, path, stream): os.makedirs(os.path.dirname(path), exist_ok=True); self.f = open(path, "a"); self.s = stream
    def write(self, x): self.s.write(x); self.f.write(x); self.f.flush()
    def flush(self): self.s.flush(); self.f.flush()
def _committer(stop):
    while not stop.wait(120):
        try: vol.commit()
        except Exception as e: print("commit failed", e, flush=True)
def _go(tag, mode, fn):
    sys.path.insert(0, "/code"); os.chdir("/code"); sys.stdout = Tee(f"/vol/work/out/{tag}/{mode}/log.txt", sys.__stdout__)
    stop = threading.Event(); th = threading.Thread(target=_committer, args=(stop,), daemon=True); th.start()
    try: return fn()
    except Exception:
        import traceback; print("FAILED\n" + traceback.format_exc(), flush=True); raise
    finally:
        stop.set(); th.join(timeout=150); t = sys.stdout; sys.stdout = sys.__stdout__
        try: t.f.close()
        except Exception: pass
        vol.commit()
def _reload():
    for i in range(6):
        try: vol.reload(); return
        except Exception as e: print("reload retry", i, repr(e)[:200], flush=True); time.sleep(10)

@app.function(image=base, volumes={"/vol": vol}, secrets=[sec], timeout=3600, memory=16384, cpu=4)
def prep():
    def f():
        import px, lib as Lb
        from huggingface_hub import snapshot_download, hf_hub_download
        px.prep(); Lb.load_tok(); snapshot_download(Lb.BASE, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
        for n in ("bad_s0", "bad_s1", "bad_s2"):
            for x in ("adapter_config.json", "adapter_model.safetensors"): hf_hub_download(Lb.ADAPTERS[n], x)
        print("PREP_DONE", flush=True)
    _go("prep", "full", f)

GPU = dict(gpu="A100-80GB", volumes={"/vol": vol}, secrets=[sec])

@app.function(image=base, timeout=5 * 3600, memory=131072, cpu=8, max_containers=2, **GPU)
def lane(exps: list, mode: str = "smoke", tag: str = "lane"):
    """One A100 for a group of experiments. Their score phases run as parallel processes on this GPU; as soon as one finishes, its answer
    sets are written here, one after another, with the model loaded once (while the other score phase keeps running)."""
    _reload()
    def f():
        import subprocess, json, time as tm
        T0 = tm.time(); log = lambda *a: print(f"[{tag} {round(tm.time() - T0)}s]", *a, flush=True); procs = {}
        for w in exps:
            d = f"/vol/work/out/{w[0]}/{mode}"; os.makedirs(d, exist_ok=True); jp = f"{d}/jobs.json"
            if os.path.exists(jp): os.remove(jp)
            code = f"import importlib, json; j = importlib.import_module('{w}').main('{mode}'); json.dump(j, open('{jp}', 'w'))"
            lf = open(f"{d}/log.txt", "a"); procs[w] = (subprocess.Popen([sys.executable, "-c", code], cwd="/code", stdout=lf, stderr=subprocess.STDOUT), lf, jp)
            log("started score phase", w)
        import gen, torch
        queue, done, failed = [], [], []
        while procs or queue:
            for w in list(procs):
                pr, lf, jp = procs[w]; rc = pr.poll()
                if rc is None: continue
                lf.close(); del procs[w]
                if rc != 0 or not os.path.exists(jp): log("SCORE PHASE FAILED", w, "exit", rc); failed.append(w); continue
                js = json.load(open(jp)); queue += js; log("scores done", w, "->", len(js), "answer sets")
            if queue:
                j = queue.pop(0)
                try: r = gen.run(j); done.append(r); log("answers done", j["prefix"], j["name"], r.get("stat"), f"GPU now {torch.cuda.mem_get_info()[0] / 2**30:.0f} GB free")
                except Exception as e: log("ANSWERS FAILED", j["prefix"], j["name"], repr(e)[:300]); failed.append(f"{j['prefix']}_{j['name']}")
            else: tm.sleep(10)
        log("LANE_DONE failed:", failed); return {"done": len(done), "failed": failed}
    return _go(tag, mode, f)

@app.function(image=base, volumes={"/vol": vol}, secrets=[sec], timeout=3600, memory=8192, cpu=4)
def predl_judge(judge: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
    def f():
        from huggingface_hub import snapshot_download; snapshot_download(judge, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"]); print("JUDGE_DOWNLOADED", flush=True)
    _go("predl", "full", f)

@app.function(image=jimg, timeout=3 * 3600, memory=65536, cpu=8, max_containers=2, **GPU)
def run_judge(mode: str = "smoke", shard: int = 0, nshard: int = 1):
    _reload()
    def f():
        import subprocess
        r = subprocess.run(["/vv/bin/python", "-c", f"import judge; judge.main('{mode}', {shard}, {nshard})"], cwd="/code", stdout=sys.stdout.f, stderr=subprocess.STDOUT)
        assert r.returncode == 0, f"judge shard {shard} exit {r.returncode}"
    _go("judge", f"{mode}/shard{shard}", f)

@app.function(image=base, volumes={"/vol": vol}, timeout=1800, memory=16384, cpu=2)
def merge_judge(mode: str, nshard: int):
    _reload()
    def f():
        import judge; judge.merge(mode, nshard)
    _go("judge", f"{mode}/merge", f)

@app.function(image=base, volumes={"/vol": vol}, timeout=4 * 3600, memory=4096, cpu=1)
def judge_only(mode: str = "full", nshard: int = 2):
    T0 = time.time(); log = lambda *a: print(f"[judge_only {round(time.time() - T0)}s]", *a, flush=True)
    rs = list(run_judge.starmap([(mode, s, nshard) for s in range(nshard)], return_exceptions=True)); log("shards", rs)
    assert not any(isinstance(r, Exception) for r in rs), "a judge shard failed"
    merge_judge.remote(mode, nshard); log("JUDGE_ONLY_DONE")

@app.function(image=base, volumes={"/vol": vol}, timeout=8 * 3600, memory=4096, cpu=1)
def pipeline(mode: str = "smoke", predl: bool = True):
    """Everything on Modal, never more than 2 A100s: lane 1 = experiments B and C, lane 2 = experiment A; then the labeller on 2 shards."""
    T0 = time.time(); log = lambda *a: print(f"[pipeline {round(time.time() - T0)}s]", *a, flush=True)
    dl = predl_judge.spawn() if predl else None
    lanes = [lane.spawn(["b_neurons", "c_circuit"], mode, "lane1"), lane.spawn(["a_controls"], mode, "lane2")]
    for c in lanes:
        try: log("lane finished", c.get())
        except Exception as e: log("LANE FAILED", repr(e)[:300])
    try: dl and dl.get()
    except Exception as e: log("judge download failed", repr(e)[:300])
    ns = 2
    for r in run_judge.starmap([(mode, s, ns) for s in range(ns)], return_exceptions=True): log("judge shard", r)
    merge_judge.remote(mode, ns); log("PIPELINE_DONE")

@app.local_entrypoint()
def judge_local(mode: str = "full", nshard: int = 2):
    """Labelling without a coordinator container: the 2 GPU shards are called from here, then the merge (about 10 minutes)."""
    rs = list(run_judge.starmap([(mode, s, nshard) for s in range(nshard)], return_exceptions=True)); print("shards", rs, flush=True)
    assert not any(isinstance(r, Exception) for r in rs), "a judge shard failed"
    merge_judge.remote(mode, nshard); print("JUDGE_LOCAL_DONE", flush=True)
