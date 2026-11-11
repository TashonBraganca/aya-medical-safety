"""Tiny driver for one persistent Modal L40S sandbox.
  python msb.py create            build image, start sandbox (12 h max), save its id
  python msb.py put LOCAL REMOTE  copy a local file into the sandbox
  python msb.py sh 'CMD'          run a short command and print its output
  python msb.py bg NAME 'CMD'     start CMD in the background, log to /vol/logs/NAME.log
  python msb.py stop              terminate the sandbox"""
import os, sys, modal
from huggingface_hub import get_token
HERE = os.path.dirname(os.path.abspath(__file__)); IDF = os.path.join(HERE, os.environ.get("SANDBOX_ID_FILE", "sandbox_id.txt"))
app = modal.App.lookup("lexsi-day", create_if_missing=True)
vol = modal.Volume.from_name("lexsi-vol", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .pip_install("torch==2.8.0", "transformers==5.14.1", "accelerate", "peft", "bitsandbytes", "safetensors", "numpy", "pandas", "pyarrow",
                      "huggingface_hub", "easy-dataset-share", "sentencepiece", "protobuf", "datasets", "scipy"))
def sb():
    return modal.Sandbox.from_id(open(IDF).read().strip())
cmd = sys.argv[1]
if cmd == "create":
    s = modal.Sandbox.create("sleep", "infinity", app=app, image=image, gpu="L40S", cpu=int(os.environ.get("SB_CPU", "8")), memory=int(os.environ.get("SB_MEM", "65536")), timeout=int(os.environ.get("SB_TIMEOUT", str(12 * 3600))),
                             volumes={"/vol": vol}, secrets=[modal.Secret.from_dict({"HF_TOKEN": get_token(), "HF_HOME": "/vol/hf", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})])
    open(IDF, "w").write(s.object_id); print("sandbox", s.object_id)
    p = s.exec("bash", "-c", "nvidia-smi --query-gpu=name,memory.total --format=csv; mkdir -p /vol/logs /vol/kit /vol/out; python -c 'import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available())'"); p.wait(); print(p.stdout.read(), p.stderr.read()[-500:])
elif cmd == "put":
    import base64
    for src in sys.argv[2:-1]:
        data = base64.b64encode(open(src, "rb").read()).decode(); dst = sys.argv[-1].rstrip("/") + "/" + os.path.basename(src)
        p = sb().exec("bash", "-c", f"base64 -d > {dst} && wc -c {dst}"); [(p.stdin.write(data[i:i + (1 << 18)]), p.stdin.drain()) for i in range(0, len(data), 1 << 18)]; p.stdin.write_eof(); p.stdin.drain(); p.wait(); print(p.stdout.read().strip(), p.stderr.read()[-300:])
elif cmd == "sh":
    p = sb().exec("bash", "-c", sys.argv[2]); p.wait(); print(p.stdout.read(), end=""); e = p.stderr.read()
    if e.strip(): print("[stderr]", e[-3000:])
elif cmd == "bg":
    name, c = sys.argv[2], sys.argv[3]
    p = sb().exec("bash", "-c", f"cd /vol/kit && nohup bash -c {c!r} > /vol/logs/{name}.log 2>&1 &  echo started {name}"); p.wait(); print(p.stdout.read())
elif cmd == "stop":
    sb().terminate(); print("terminated")
