"""Run every problem's verify.py and report one line per problem. Python 3 standard library only.

Usage: python3 verify_all.py
Exit code 0 only if every check in every problem passes."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
failed = []
for name in ("problem9", "problem10", "problem11", "problem12"):
    d = os.path.join(HERE, name)
    r = subprocess.run([sys.executable, "-B", "verify.py"], cwd=d, capture_output=True, text=True)
    lines = r.stdout.strip().splitlines()
    n_pass = sum(l.startswith("PASS") for l in lines)
    n_fail = sum(l.startswith("FAIL") for l in lines)
    ok = r.returncode == 0 and n_fail == 0
    print(f"{name:10s} {'ok  ' if ok else 'FAIL'} {n_pass} checks passed, {n_fail} failed")
    if not ok:
        failed.append(name)
        print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
print("ALL PROBLEMS VERIFIED" if not failed else f"FAILED: {', '.join(failed)}")
sys.exit(1 if failed else 0)
