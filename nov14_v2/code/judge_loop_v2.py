"""Judge loop v2 (14 Nov): every 3 minutes, collect FINISHED answer files from the pod (E2) and both Modal sandboxes (E3, E4) and label new
ones with the same judge as every earlier label (locked Nov 8 rulebook, same deployment, HIGH effort since 20:10 14 Nov (Tashon), labels in judge/labels_high; same rulebook and deployment).
A file counts as finished only when its job's result json lists it (written after the file is closed) and it has exactly 250 lines.
Own ledger v2/judge_ledger_v2.json; halts at $20 (total judge spend then stays under Tashon's $70). Stop: touch v2/STOP_JUDGE."""
import json, os, re, subprocess, time
F = "/Volumes/Tashon/hackathon/fable_plan"; V2 = f"{F}/v2"; PULL = f"{V2}/pulled"; os.makedirs(PULL, exist_ok=True)
JUDGE = "/private/tmp/claude-501/-Volumes-Tashon-hackathon/92ec7235-8f86-47da-a0a6-573941f5bfd2/scratchpad/judge/judge.py"
ENV = {**os.environ, **dict(l.strip().split("=", 1) for l in open(os.path.expanduser("~/.config/lexsi/azure.env")) if "=" in l)}
DEP = ENV["AZURE_OPENAI_DEPLOYMENT"]; CAP = 45.0; LED = f"{V2}/judge_ledger_v2.json"
led = json.load(open(LED)) if os.path.exists(LED) else {"spent": 0.0, "judged": []}
MS = "/private/tmp/claude-501/-Volumes-Tashon-hackathon/50b60935-4103-4733-bddc-0077ee957d33/scratchpad/modal"
SB = {"A": ("tashon-braganca", "sb_A.txt"), "B": ("tashonbraganza7", "sb_B.txt"), "C": ("tashon-braganca", "sb_C.txt"), "D": ("tashonbraganza7", "sb_D.txt")}
def sh(c, t=600, env=None): return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t, env=env or ENV).stdout
def msb(acct, *args, t=600):
    prof, idf = SB[acct]; e = {**ENV, "MODAL_PROFILE": prof, "SANDBOX_ID_FILE": idf}
    return subprocess.run(["timeout", str(t), "uv", "run", "--quiet", "--with", "modal", "--with", "huggingface_hub", "--python", "3.12", "python", f"{MS}/msb.py", *args], capture_output=True, text=True, timeout=t + 30, env=e, cwd=MS).stdout
def listed(text, pattern_dir):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"): continue
        try: d = json.loads(line)
        except Exception: continue
        for name in d.get("gens", {}): out.append(name)
    return out
def finished():
    out = []
    t = sh("ssh -o ConnectTimeout=15 -o BatchMode=yes runpod 'cat /dev/shm/lexsi_p910/v2/out/e2/full/e2.json 2>/dev/null; echo'", 120)
    for n in listed(t, ""): out.append(("pod", f"/dev/shm/lexsi_p910/v2/out/e2/full/gen/gen_e2_{n}.jsonl", f"gen_e2_{n}.jsonl"))
    t = sh("ssh -o ConnectTimeout=15 -o BatchMode=yes runpod 'cat /dev/shm/lexsi_p910/v2/out/e2b/full/e2b.json 2>/dev/null; echo'", 120)
    for n in listed(t, ""): out.append(("pod", f"/dev/shm/lexsi_p910/v2/out/e2b/full/gen/gen_e2b_{n}.jsonl", f"gen_e2b_{n}.jsonl"))
    t = sh("ssh -o ConnectTimeout=15 -o BatchMode=yes runpod 'cat /dev/shm/lexsi_p910/v2/out/e2j_bad_s2/full/e2j.json 2>/dev/null; echo'", 120)
    for n in listed(t, ""): out.append(("pod", f"/dev/shm/lexsi_p910/v2/out/e2j_bad_s2/full/gen/gen_{n}.jsonl", f"gen_{n}.jsonl"))
    for acct, files in (("A", ["/vol/v2out/e3/full/e3_seed0.json", "/vol/v2out/e3/full/e3_seed0_p2.json"]), ("C", ["/vol/v2out/e3/full/e3_seed1.json", "/vol/v2out/e2j_bad_s1/full/e2j.json"]), ("B", ["/vol/v2out/e5/full/e5.json", "/vol/v2out/e3/full/e3_seed2.json"]), ("D", ["/vol/v2out/e3/full/e3_seed0_harm15.json", "/vol/v2out/e2j_bad_s0/full/e2j.json"])):
        for f in files:
            t = msb(acct, "sh", f"cat {f} 2>/dev/null; echo", t=240)
            for n in listed(t, ""): out.append((acct, f"{os.path.dirname(f)}/gen/gen_{n}.jsonl", f"gen_{n}.jsonl"))
    return out
def pull(src, rp, name):
    dst = f"{PULL}/{name}"
    if not os.path.exists(dst):
        if src == "pod": sh(f"scp -q -o ConnectTimeout=15 runpod:{rp} {dst}.tmp && mv {dst}.tmp {dst}", 300)
        else: msb(src, "get", rp, dst, t=400)
    ok = os.path.exists(dst) and sum(1 for _ in open(dst)) == 250
    return dst if ok else None
while not os.path.exists(f"{V2}/STOP_JUDGE"):
    try:
        new = []
        for src, rp, name in finished():
            if name in led.setdefault("judged_high", []): continue
            p = pull(src, rp, name)
            if p: new.append(p)
        if new:
            if led["spent"] >= CAP: print(f"STOP: spend ${led['spent']:.2f} reached the ${CAP} cap; {len(new)} files not judged", flush=True)
            else:
                out = sh(f"cd {F}/judge && uv run --quiet --with openai --python 3.12 python {JUDGE} azure {DEP} high {F}/judge/labels_high " + " ".join(new), 5400)
                open(f"{V2}/judge_runs.log", "a").write(out)
                m = re.findall(r"new calls (\d+), tokens in (\d+) out (\d+), cost \$([0-9.]+)", out)
                if m:
                    led["spent"] += float(m[-1][3]); led["judged_high"] += [os.path.basename(p) for p in new]
                    json.dump(led, open(LED, "w"), indent=1); print(f"judged {len(new)} files, +${float(m[-1][3]):.2f}, total ${led['spent']:.2f}", flush=True)
                else: print("judge run gave no cost line; will retry", out[-500:], flush=True)
        print(time.strftime("%H:%M"), "heartbeat, judged so far", len(led["judged"]), flush=True)
    except Exception as e: print("loop error", type(e).__name__, str(e)[:200], flush=True)
    time.sleep(60)
print("stopped")
