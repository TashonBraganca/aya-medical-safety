"""Monitor backend: runs until v2/STOP_MONITOR exists. Every 5 minutes it checks every job and writes one line per finding to
v2/monitor.log. Lines start with ALERT (act now), DONE (a job finished), or SUMMARY (every 15 minutes, one line per place).
Checks: each job's log for Traceback / Error / AssertionError / OutOfMemory / watchdog kills / SKIP lines; stalls (log unchanged > 12 min);
GPU memory and use per place; judge loop and billing guard alive; billing limits; the result audit on every result file.
Read-only everywhere: it never stops, edits or deletes anything."""
import json, os, re, subprocess, time
V2 = "/Volumes/Tashon/hackathon/fable_plan/v2"; LOG = f"{V2}/monitor.log"; LIVE = f"{V2}/res/live"; os.makedirs(LIVE, exist_ok=True)
MS = "/private/tmp/claude-501/-Volumes-Tashon-hackathon/50b60935-4103-4733-bddc-0077ee957d33/scratchpad/modal"
SB = {"A": ("tashon-braganca", "sb_A.txt"), "B": ("tashonbraganza7", "sb_B.txt"), "C": ("tashon-braganca", "sb_C.txt"), "D": ("tashonbraganza7", "sb_D.txt")}
RESULTS = {"A": ["/vol/v2out/e3/full/e3_seed0.json", "/vol/v2out/e3/full/e3_seed0_p2.json"], "B": ["/vol/v2out/e2_bad_s0/full/e2.json", "/vol/v2out/e5/full/e5.json", "/vol/v2out/e3/full/e3_seed2.json"],
           "C": ["/vol/v2out/e3/full/e3_seed1.json", "/vol/v2out/e2j_bad_s1/full/e2j.json"], "D": ["/vol/v2out/e3/full/e3_seed0_harm15.json", "/vol/v2out/e2j_bad_s0/full/e2j.json"]}
PLANNED = ("e3s0_full", "e3s2B_full", "e6D_full", "e3s1C_full")   # runs stopped on purpose and continued by another process
OWN = {"A": ("e3s0_full", "e3s0p2_full", "e2s1_full"), "B": ("e2s0_full", "e5_full", "e1on_full", "e3s2B2_full"), "C": ("e3s1C2_full", "e2jC_full"), "D": ("e6D2_full", "e2jD_full")}   # A/C and B/D share volumes
BAD = re.compile(r"Traceback|Error|error:|AssertionError|OutOfMemory|out of memory|WATCHDOG: no log progress|Killed|SKIP ")
SEEN = f"{V2}/monitor_seen.json"
try: _s = json.load(open(SEEN)); seen_bad, seen_done = {tuple(x) if isinstance(x, list) else x for x in _s["bad"]}, {tuple(x) if isinstance(x, list) else x for x in _s["done"]}
except Exception: seen_bad, seen_done = set(), set()
cycle = 0
def persist(): json.dump({"bad": [list(x) if isinstance(x, tuple) else x for x in seen_bad], "done": [list(x) if isinstance(x, tuple) else x for x in seen_done]}, open(SEEN, "w"))
def out(line):
    s = f"{time.strftime('%H:%M')} {line}"; print(s, flush=True); open(LOG, "a").write(s + "\n")
def run(cmd, t=240, env=None, cwd=None):
    try: return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=t, env=env, cwd=cwd).stdout
    except Exception as e: return f"__FAIL__ {type(e).__name__}"
def msb(s, *args, t=240):
    prof, idf = SB[s]; env = {**os.environ, "MODAL_PROFILE": prof, "SANDBOX_ID_FILE": idf}
    return run(["timeout", str(t), "uv", "run", "--quiet", "--with", "modal", "--with", "huggingface_hub", "--python", "3.12", "python", f"{MS}/msb.py", *args], t + 30, env, MS)
PROBE = r'''now=$(date +%s); for L in /vol/logs/e*_full.log; do [ -f $L ] || continue; n=$(basename $L .log); age=$(( now - $(stat -c %Y $L) )); last=$(grep -v "Loading\|it/s\]" $L | tail -1 | cut -c1-140); bad=$(grep -E "Traceback|Error|error:|AssertionError|OutOfMemory|out of memory|WATCHDOG: no log progress|Killed|SKIP " $L | tail -3 | cut -c1-200 | tr "\n" "|"); done_=$(grep -c "DONE\|WATCHDOG: exit" $L); printf "LOG\t%s\t%s\t%s\t%s\tBAD:%s\n" "$n" "$age" "$done_" "$(echo "$last" | tr "\t" " ")" "$bad"; done; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | sed "s/^/GPU|/"'''
while not os.path.exists(f"{V2}/STOP_MONITOR"):
    cycle += 1; summary = []
    # ---------- Modal sandboxes
    for s in SB:
        if not os.path.exists(f"{MS}/{SB[s][1]}"): summary.append(f"{s}: stopped"); continue   # stopped by sandbox_stopper.py
        txt = msb(s, "sh", PROBE)
        if "__FAIL__" in txt or not txt.strip(): out(f"ALERT sandbox {s}: probe failed or sandbox gone ({txt.strip()[:120]})"); summary.append(f"{s}: unreachable"); continue
        jobs = []
        for line in txt.splitlines():
            if line.startswith("GPU|"): summary.append(f"{s} GPU {line[4:].strip()}"); continue
            if not line.startswith("LOG\t"): continue
            try: _, n, age, dn, last, bad = line.split("\t", 5); age = int(age); bad = bad[4:]
            except ValueError: out(f"ALERT monitor could not parse a line from {s}: {line[:120]}"); continue
            if n not in OWN[s]: continue   # stopped runs and the other sandbox's logs on a shared volume
            ended = "WATCHDOG: exit" in last or "stopped by plan" in last
            if ended and n in PLANNED and "exit 0" not in last:
                if (s, n) not in seen_done: seen_done.add((s, n)); out(f"DONE {s}/{n}: stopped on purpose, continued by its next part")
                continue
            if ended and "exit 0" not in last and (s, n, "crash") not in seen_bad: seen_bad.add((s, n, "crash")); out(f"ALERT {s}/{n}: ended with an error: {last[:120]}")
            finished = dn != "0" and ("DONE" in last or "WATCHDOG: exit 0" in last or "exit 0" in last)
            if bad.strip("|") and (s, n, bad) not in seen_bad: seen_bad.add((s, n, bad)); out(f"ALERT {s}/{n}: {bad.strip('|')[:300]}")
            if finished:
                if (s, n) not in seen_done: seen_done.add((s, n)); out(f"DONE {s}/{n}: {last[:120]}")
            elif age > 720: out(f"ALERT {s}/{n}: log unchanged for {age // 60} min (possible stall). Last: {last[:100]}")
            else: jobs.append(f"{n} ({age}s ago)")
        summary.append(f"{s} running: {', '.join(jobs) or 'nothing'}")
        for rp in RESULTS[s]:   # pull result files for the audit
            dst = f"{LIVE}/{s}_{rp.replace('/', '_').strip('_')}"
            r = msb(s, "get", rp, dst, t=200)
            if "got" not in r and os.path.exists(dst): pass
    # ---------- pod (read-only)
    p = run("ssh -o ConnectTimeout=15 -o BatchMode=yes runpod 'L=$(ls -t /dev/shm/lexsi_p910/v2/logs/e2jpod_full.log /dev/shm/lexsi_p910/v2/logs/e2b_full.log /dev/shm/lexsi_p910/v2/logs/e2b_smoke.log /dev/shm/lexsi_p910/v2/logs/e2_full.log 2>/dev/null | head -1); echo LOGNAME $(basename $L); now=$(date +%s); echo \"AGE $(( now - $(stat -c %Y $L) ))\"; grep -v Loading $L | tail -1 | cut -c1-140; grep -E \"Traceback|Error|AssertionError|OutOfMemory|WATCHDOG\" $L | tail -2 | cut -c1-200; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader'", 90)
    if "AGE" not in p: out(f"ALERT pod: unreachable ({p.strip()[:100]})")
    else:
        age = int(re.search(r"AGE (-?\d+)", p).group(1)); lines = [l for l in p.splitlines() if not l.startswith(("AGE", "LOGNAME"))]
        if "WATCHDOG: exit 0" in p or "DONE" in p:
            tagp = re.search(r"LOGNAME (\S+)", p).group(1) if "LOGNAME" in p else "pod"
            if tagp not in seen_done: seen_done.add(tagp); out(f"DONE pod/{tagp}: {lines[0][:120] if lines else ''}")
        elif age > 720: out(f"ALERT pod/e2_full: log unchanged for {age // 60} min")
        for l in lines:
            if re.search(r"Traceback|Error|OutOfMemory|no log progress", l) and ("pod", l) not in seen_bad: seen_bad.add(("pod", l)); out(f"ALERT pod: {l[:200]}")
        summary.append(f"pod: {lines[0][:90] if lines else '?'} | GPU {lines[-1] if lines else '?'}")
        run("scp -q -o ConnectTimeout=15 runpod:/dev/shm/lexsi_p910/v2/out/e2/full/e2.json " + f"{LIVE}/pod_e2_s2.json", 90); run("scp -q -o ConnectTimeout=15 runpod:/dev/shm/lexsi_p910/v2/out/e2b/full/e2b.json " + f"{LIVE}/pod_e2b.json", 90)
    # ---------- laptop services
    for name, pat in (("judge loop", "judge_loop_v3.py"), ("billing guard", "billing_guard.sh")):
        if not run(f"pgrep -f {pat}").strip(): out(f"ALERT {name} is not running")
    bl = run(f"tail -6 {V2}/billing_guard.log")
    if "LIMIT" in bl and "LIMIT" not in seen_bad: seen_bad.add("LIMIT"); out(f"ALERT billing limit hit: {bl.strip()[-200:]}")
    hb = [l for l in run(f"tail -40 {V2}/judge_loop_v3.log").splitlines() if "heartbeat" in l]
    sp = float(re.search(r"spend \$([0-9.]+)", hb[-1]).group(1)) if hb else 0.0; nj = int(re.search(r"(\d+) judging", hb[-1]).group(1)) if hb else 0
    led = {"spent": sp, "judged": ["x"] * nj}
    if led["spent"] > 55: out(f"ALERT judge spend ${led['spent']:.2f} near the $60 cap")
    # ---------- audit every result file
    files = [f"{LIVE}/{f}" for f in os.listdir(LIVE) if f.endswith(".json")]
    a = run(["python3", f"{V2}/audit_results.py", *files], 120) if files else "no result files yet"
    if "AUDIT OK" not in a and files and ("audit", a) not in seen_bad: seen_bad.add(("audit", a)); out(f"ALERT audit: {a.strip()[:400]}")
    if cycle % 3 == 1:
        billing = " ".join(l.split(" ", 1)[1] for l in bl.strip().splitlines()[-3:]) if bl.strip() else "?"
        out("SUMMARY " + " || ".join(summary) + f" || judge: {len(led['judged'])} judging now, ${led['spent']:.2f} spent || billing: {billing} || audit: {'OK' if 'AUDIT OK' in a else a.strip()[:60]}")
    persist(); time.sleep(300)
out("monitor stopped")
