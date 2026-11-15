"""Judge loop v4 (15 Nov 03:50): v3 plus a local inbox (v2/judge_inbox, any complete 250-line file) and the pod e3x job; cap $100 (Tashon, 15 Nov).
Judge loop v3 (14 Nov, 20:15): every 30 s, find FINISHED answer files on the pod and all four sandboxes, and start one judge process per
new file at once (all in parallel), high effort, same rulebook and deployment, labels in judge/labels_high (answer-keyed cache).
A file counts as judged only when all 250 answers have a high-effort label in the cache. Files already being labelled by another judge
process are skipped. Spend: summed from every judge log's final cost line; halts new launches at $60 (Tashon raised it again at 20:48).
Stop: touch v2/STOP_JUDGE4."""
import glob, hashlib, json, os, re, subprocess, time
F = "/Volumes/Tashon/hackathon/fable_plan"; V2 = f"{F}/v2"; PULL = f"{V2}/pulled"; LOGS = f"{V2}/judge_logs"; os.makedirs(PULL, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
JUDGE = "judge/judge.py"  # local path of the judge script, not shipped
ENV = {**os.environ, **dict(l.strip().split("=", 1) for l in open(os.path.expanduser("~/.config/lexsi/azure.env")) if "=" in l)}
DEP = ENV["AZURE_OPENAI_DEPLOYMENT"]; CAP = 100.0; CACHE = f"{F}/judge/labels_high/labels.jsonl"
RB = open("/Volumes/Tashon/hackathon/analysis/r910/p9_kit/RULEBOOK.md").read()
MS = "modal/"  # Modal was not used after 15 Nov
SB = {"A": ("tashon-braganca", "sb_A.txt"), "B": ("tashonbraganza7", "sb_B.txt"), "C": ("tashon-braganca", "sb_C.txt"), "D": ("tashonbraganza7", "sb_D.txt")}
SOURCES = {"A": ["/vol/v2out/e3/full/e3_seed0.json", "/vol/v2out/e3/full/e3_seed0_p2.json"], "C": ["/vol/v2out/e3/full/e3_seed1.json", "/vol/v2out/e2j_bad_s1/full/e2j.json"],
           "B": ["/vol/v2out/e5/full/e5.json", "/vol/v2out/e3/full/e3_seed2.json"], "D": ["/vol/v2out/e3/full/e3_seed0_harm15.json", "/vol/v2out/e2j_bad_s0/full/e2j.json"]}
POD = [("/dev/shm/lexsi_p910/v2/out/e2/full/e2.json", "gen_e2_{}"), ("/dev/shm/lexsi_p910/v2/out/e2b/full/e2b.json", "gen_e2b_{}"), ("/dev/shm/lexsi_p910/v2/out/e2j_bad_s2/full/e2j.json", "gen_e2j2_{}"), ("/dev/shm/lexsi_p910/v2/out/e3x/full/e3x.json", "gen_{}")]
def sh(c, t=300):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout
    except Exception: return ""
def msb(acct, *args, t=300):
    prof, idf = SB[acct]; e = {**ENV, "MODAL_PROFILE": prof, "SANDBOX_ID_FILE": idf}
    try: return subprocess.run(["timeout", str(t), "uv", "run", "--quiet", "--with", "modal", "--with", "huggingface_hub", "--python", "3.12", "python", f"{MS}/msb.py", *args], capture_output=True, text=True, timeout=t + 30, env=e, cwd=MS).stdout
    except Exception: return ""
def gens(text):
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try: out += list(json.loads(line).get("gens", {}))
            except Exception: pass
    return out
def finished():
    out = []
    for path, pat in POD:
        for n in gens(sh(f"ssh -o ConnectTimeout=15 -o BatchMode=yes runpod 'cat {path} 2>/dev/null; echo'", 90)):
            fn = pat.format(n) + ".jsonl"; out.append(("pod", f"{os.path.dirname(path)}/gen/{fn}", fn))
    for acct, files in SOURCES.items():
        if True: continue   # Modal not used after 15 Nov (Tashon: use HF); sandbox files were pulled before   # sandbox stopped: its files were pulled before the stop
        for f in files:
            m = re.search(r"e2j_bad_s(\d)", f); pre = f"e2j{m.group(1)}_" if m else ""   # e2j jobs prefix their files with the add-on seed
            for n in gens(msb(acct, "sh", f"cat {f} 2>/dev/null; echo", t=200)): out.append((acct, f"{os.path.dirname(f)}/gen/gen_{pre}{n}.jsonl", f"gen_{pre}{n}.jsonl"))
    for f in sorted(glob.glob(f"{V2}/judge_inbox/gen_*.jsonl")):
        if os.path.exists(f + ".done"): out.append(("local", f, os.path.basename(f)))
    return out
def pull(src, rp, name):
    dst = f"{PULL}/{name}"
    if not os.path.exists(dst):
        if src == "local": sh(f"cp {rp} {dst}")
        elif src == "pod": sh(f"scp -q -o ConnectTimeout=15 runpod:{rp} {dst}.tmp && mv {dst}.tmp {dst}", 300)
        else: msb(src, "get", rp, dst, t=400)
    if os.path.exists(dst) and sum(1 for _ in open(dst)) == 250: return dst
    if os.path.exists(dst): os.remove(dst)   # partial or failed fetch: remove so the next cycle fetches it again
    return None
def labelled(path):
    keys = set()
    if os.path.exists(CACHE):
        for l in open(CACHE):
            try: keys.add(json.loads(l)["key"])
            except Exception: pass
    n = 0
    for l in open(path):
        x = json.loads(l); n += hashlib.sha256((RB + DEP + "\x00" + x["question"] + "\x00" + x["reply"]).encode()).hexdigest() in keys
    return n
def busy_names():   # files already being labelled at high effort by any judge process on this laptop
    return set(re.findall(r"(gen_[\w.\-]+?\.jsonl)", sh("ps -axo command | grep 'judge.py azure' | grep ' high ' | grep -v grep")))
def spent():
    tot = 6.31   # high-effort part of the v2 loop before 20:15 (its ledger)
    for f in glob.glob(f"{LOGS}/*.log") + [f"{V2}/rejudge_high_refs.log", f"{V2}/rejudge_high_1.log"]:
        m = re.findall(r"cost \$([0-9.]+)", open(f).read()) if os.path.exists(f) else []
        if m: tot += float(m[-1])
    return tot
running = {}
while not os.path.exists(f"{V2}/STOP_JUDGE4"):
    try:
        for name, p in list(running.items()):
            if p.poll() is not None: print(time.strftime("%H:%M"), f"judge finished {name} (exit {p.returncode})", flush=True); running.pop(name)
        busy = busy_names()
        for src, rp, name in finished():
            if name in running or name in busy: continue
            dst = f"{PULL}/{name}"
            if os.path.exists(dst) and labelled(dst) == 250: continue
            p = pull(src, rp, name)
            if not p: continue
            if spent() >= CAP: print(time.strftime("%H:%M"), f"STOP: judge spend ${spent():.2f} reached the ${CAP} cap; {name} not judged", flush=True); continue
            log = open(f"{LOGS}/{name}.log", "w")
            running[name] = subprocess.Popen(f"cd {F}/judge && uv run --quiet --with openai --python 3.12 python {JUDGE} azure {DEP} high {F}/judge/labels_high {p}", shell=True, stdout=log, stderr=subprocess.STDOUT, env=ENV)
            print(time.strftime("%H:%M"), f"judge started {name}", flush=True)
        print(time.strftime("%H:%M"), f"heartbeat: {len(running)} judging, spend ${spent():.2f}", flush=True)
    except Exception as e: print("loop error", type(e).__name__, str(e)[:200], flush=True)
    time.sleep(30)
print("stopped")
