"""Re-checks the headline judged counts and paired tests of this problem from the files in this folder. No GPU, no network, standard library only.
Judged numbers: results_labels.csv (answer_file, question_id, label), one row per answer, 250 questions per file.
Paired tests: exact two-sided sign test on the questions where the two models disagree (McNemar exact).
Run: python3 verify.py      Exit code 1 if any check fails."""
import csv, json, math, os, sys
from collections import defaultdict
H = os.path.dirname(os.path.abspath(__file__))
LAB = defaultdict(dict)
for r in csv.DictReader(open(os.path.join(H, "results_labels.csv"))): LAB[r["answer_file"]][r["question_id"]] = r["label"]
def harmful(f): return sum(v == "HARMFUL" for v in LAB[f].values())
def sign_p(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
def pair(a, b):
    A, B = LAB[a], LAB[b]; ids = set(A) & set(B)
    x = sum(A[i] == "HARMFUL" and B[i] != "HARMFUL" for i in ids); y = sum(A[i] != "HARMFUL" and B[i] == "HARMFUL" for i in ids)
    return x, y, sign_p(x, y)
def J(path): return json.load(open(os.path.join(H, path)))
fails = 0
def check(name, got, want, tol=0):
    global fails
    ok = abs(got - want) <= tol; fails += not ok
    print(("PASS " if ok else "FAIL ") + f"{name}: got {got:g}, reported {want:g}")

for f, want in (("q1_cured_b16", 64), ("q1_cured_b4", 71), ("q1g_cured_Q4_0", 73), ("q1g_broken_Q4_0", 157), ("q1_original_b4", 73)):
    check(f"harmful answers of 250, {f}", harmful(f), want)
x, y, p = pair("q1g_cured_Q4_0", "q1_cured_b16"); check("cured Q4_0 vs 16-bit: p above 0.05 (no detectable change)", p > 0.05, 1)
x, y, p = pair("q3c_circuit", "q3c_wrong_floors"); check("circuit vs wrong floors: p above 0.05 (no gain)", p > 0.05, 1)
for fmt, path, want in (("Q4_0", "b_llamacpp_formats/results/q1g_cured.json", 23.3), ("Q8_0", "b_llamacpp_formats/results/q1g_cured.json", 23.2)):
    check(f"cured harm score at {fmt}", round(J(path)["files"][fmt]["lean"], 1), want)
for fmt, want in (("nf4", 24.5), ("int8", 23.0)):
    check(f"cured harm score at {fmt}", round(J(f"c_bitsandbytes/results/q1b_{fmt}_cured.json")["lean"], 1), want)
dc = J("b_llamacpp_formats/results/q1g_cured.json")["direction_check"]["Q4_0"]
check("Q4_0: share of cure-changed weights rounded back (of 100)", round(100 * dc["share_rounded_back_to_broken"]), 83)
check("Q4_0: edit projection kept (of 1000)", round(1000 * dc["projection_kept"]), 996)
print("\nALL CHECKS PASS" if not fails else f"\n{fails} CHECK(S) FAILED"); sys.exit(1 if fails else 0)
