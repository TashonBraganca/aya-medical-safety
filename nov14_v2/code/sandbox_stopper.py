"""Stops each Modal sandbox as soon as its work is finished and saved on the laptop (Tashon, 20:45: no idle GPUs).
A sandbox is stopped only when ALL of: every job log it owns ends with DONE / exit 0; every result json it owns is copied to v2/res/final/;
every answer file listed in those jsons exists in v2/pulled with 250 lines. Checks every 60 s. Logs to v2/sandbox_stopper.log."""
import json, os, subprocess, time
V2 = "/Volumes/Tashon/hackathon/fable_plan/v2"; FIN = f"{V2}/res/final"; os.makedirs(FIN, exist_ok=True)
MS = "/private/tmp/claude-501/-Volumes-Tashon-hackathon/50b60935-4103-4733-bddc-0077ee957d33/scratchpad/modal"
JOBS = {"A": ("tashon-braganca", ["e3s0_full", "e3s0p2_full"], ["/vol/v2out/e3/full/e3_seed0.json", "/vol/v2out/e3/full/e3_seed0_p2.json"], ""),
        "B": ("tashonbraganza7", ["e3s2B2_full"], ["/vol/v2out/e3/full/e3_seed2.json"], ""),
        "C": ("tashon-braganca", ["e3s1C2_full", "e2jC_full"], ["/vol/v2out/e3/full/e3_seed1.json", "/vol/v2out/e2j_bad_s1/full/e2j.json"], ""),
        "D": ("tashonbraganza7", ["e6D2_full", "e2jD_full"], ["/vol/v2out/e3/full/e3_seed0_harm15.json", "/vol/v2out/e2j_bad_s0/full/e2j.json"], "")}
def out(s): line = f"{time.strftime('%H:%M')} {s}"; print(line, flush=True); open(f"{V2}/sandbox_stopper.log", "a").write(line + "\n")
def msb(s, *a, t=300):
    prof = JOBS[s][0]; e = {**os.environ, "MODAL_PROFILE": prof, "SANDBOX_ID_FILE": f"sb_{s}.txt"}
    try: return subprocess.run(["timeout", str(t), "uv", "run", "--quiet", "--with", "modal", "--with", "huggingface_hub", "--python", "3.12", "python", f"{MS}/msb.py", *a], capture_output=True, text=True, timeout=t + 30, env=e, cwd=MS).stdout
    except Exception: return ""
stopped = set()
while len(stopped) < len(JOBS) and not os.path.exists(f"{V2}/STOP_STOPPER"):
    for s, (prof, logs, results, _) in JOBS.items():
        if s in stopped or not os.path.exists(f"{MS}/sb_{s}.txt"): continue
        txt = msb(s, "sh", "for L in " + " ".join(f"/vol/logs/{l}.log" for l in logs) + "; do [ -f $L ] && tail -2 $L | tr '\\n' ' '; echo \"|$L\"; done")
        lines = [l for l in txt.splitlines() if "|" in l]
        if len(lines) < len(logs) or not all(("DONE" in l or "exit 0" in l or "stopped by plan" in l) for l in lines): continue
        ok = True
        for rp in results:
            dst = f"{FIN}/{s}_{os.path.basename(os.path.dirname(os.path.dirname(rp)))}_{os.path.basename(rp)}"
            msb(s, "get", rp, dst)
            try: r = json.load(open(dst))
            except Exception: ok = False; out(f"{s}: could not copy {rp} yet"); continue
            pre = ""
            if "e2j_bad_s" in rp: pre = f"e2j{rp.split('e2j_bad_s')[1][0]}_"
            for n in r.get("gens", {}):
                fn = n if n.startswith("e3") else f"{pre}{n}"
                p = f"{V2}/pulled/gen_{fn}.jsonl"
                if not (os.path.exists(p) and sum(1 for _ in open(p)) == 250): ok = False; out(f"{s}: waiting for answer file gen_{fn}.jsonl to be pulled")
        if not ok: continue
        r = msb(s, "stop"); os.rename(f"{MS}/sb_{s}.txt", f"{MS}/sb_{s}.stopped.txt"); out(f"STOPPED sandbox {s}: all jobs done, results saved in {FIN}, answers pulled ({r.strip()[:40]})"); stopped.add(s)
    time.sleep(60)
out("stopper finished")
