"""Every 4 minutes: pull finished answer files from the Modal volume and judge them.
DeepSeek V4.1 Flash (OpenRouter, high) judges every file; Judge A (Azure, medium) judges the high-priority files.
A file counts as finished only when its job's results.json lists its tag under "gens" (written after the file is closed)."""
import json, os, subprocess, sys, time
J = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(J, "..", "modal", "m"); PULL = os.path.join(J, "pulled"); os.makedirs(PULL, exist_ok=True)
HIGH = {"e6": set(), "a": {"broken", "normal_s0", "fixed_s0", "normal_seed0_s50", "normal_seed1_s50", "fixed_seed0_s50", "fixed_seed1_s50",
              "R4_merged_big512_off_seed0", "R4_merged_big512_off_seed1", "R1_attack_big256_off_seed0", "R1_attack_big256_off_seed1"},
        "b": {"normal", "twojob", "twojob_big256_off", "twojob_blame256_off"},
        "c": {"fixed_nf4", "broken_nf4", "blame64_off", "big64_off"}}
def env(f): return {**os.environ, **dict(l.strip().split("=", 1) for l in open(os.path.expanduser(f)) if "=" in l)}
EO, EA = env("~/.config/lexsi/openrouter.env"), env("~/.config/lexsi/azure.env")
M2 = os.path.join(J, "..", "modal", "m2")
def sh(cmd, m=None): return subprocess.run([m or M, "sh", cmd], capture_output=True, text=True, timeout=600).stdout
E6HIGH = {"circuit", "none", "normal", "crft", "crft_circuit", "circuit_long200", "normal_long200", "none_long200"}
done = {"ds": set(), "judge_a": set()}
while True:
    new = []
    for job in ("a", "b", "c"):
        try: res = json.loads(sh(f"cat /vol/out/{job}_full/results.json 2>/dev/null") or "{}")
        except json.JSONDecodeError: continue
        for tag in res.get("gens", {}):
            loc = os.path.join(PULL, f"{job}__gen_{tag}.jsonl")
            if not os.path.exists(loc):
                txt = sh(f"cat /vol/out/{job}_full/gen_{tag}.jsonl"); n = txt.count("\n")
                if n < 200 and not tag.startswith("broadq_"): print("short file, skip for now", tag, n, flush=True); continue
                open(loc, "w").write(txt); print(time.strftime("%H:%M"), "pulled", job, tag, n, flush=True)
            new.append((job, tag, loc))
    e6 = []
    for mm in (M2, M, os.path.join(J, "..", "modal", "m3")):
        try: e6 += [(mm, f, g) for f, g in json.loads(sh("python -c \"import json,glob; print(json.dumps({f: json.load(open(f)).get('gens', {}) for f in glob.glob('/vol/out/e6/full/atk_*.json')}))\"", mm) or "{}").items()]
        except json.JSONDecodeError: pass
    for mm, f, gens in e6:
        tag0 = os.path.basename(f)[4:-5]
        for g in gens:
            loc = os.path.join(PULL, f"e6__gen_{tag0}_{g}.jsonl")
            if not os.path.exists(loc):
                txt = sh(f"cat /vol/out/e6/full/gen_{tag0}_{g}.jsonl", mm)
                if txt.count("\n") < 300: continue
                open(loc, "w").write(txt); print(time.strftime("%H:%M"), "pulled e6", tag0, g, flush=True)
            hi = tag0 in E6HIGH or (g in ("pre", "seed0_lr1e-05_s50"))
            new.append(("e6", g if hi else "_low", loc))
    ds = [l for j, t, l in new if l not in done["ds"]]
    procs = []
    if ds:
        procs.append(subprocess.Popen(["uv", "run", "--quiet", "--with", "openai", "python", os.path.join(J, "judge.py"), "openrouter", "deepseek/deepseek-v4.1-flash", "high", os.path.join(J, "out_deepseek"), *ds], env=EO, cwd=J))
        done["ds"].update(ds)
    hi = [l for j, t, l in new if (j == "e6" and t != "_low" or j != "e6" and t in HIGH[j]) and l not in done["judge_a"]]
    tr = hi + [l for j, t, l in new if l not in done["judge_a"] and l not in hi]   # everything goes to Judge A now (budget raised to $60-80); headline files first
    if tr:
        procs.append(subprocess.Popen(["uv", "run", "--quiet", "--with", "openai", "python", os.path.join(J, "judge.py"), "azure", "<hosted-reviewer-deployment>", "medium", os.path.join(J, "out_judge_a"), *tr], env=EA, cwd=J))
        done["judge_a"].update(tr)
    for p in procs: p.wait()
    if os.path.exists(os.path.join(J, "STOP")): break
    time.sleep(60)
