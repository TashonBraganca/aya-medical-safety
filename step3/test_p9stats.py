import random, sys
sys.path.insert(0, "/Volumes/Tashon/hackathon/repos/AuditKIT/src")
from p9stats import *
from auditkit._bootstrap import paired_bootstrap as ak_bootstrap
ok = True
def check(name, cond, got):
    global ok; ok &= cond; print(("PASS " if cond else "FAIL ") + name, got)
# 1. Wilson must reproduce the ranges already on the Notion page
for k, want in [(14, (8, 23)), (360, (332, 389)), (9, (5, 17)), (131, (112, 153)), (377, (348, 406))]:
    lo, hi = wilson(k, 900); check(f"wilson {k}/900", (round(lo), round(hi)) == want, (round(lo), round(hi)))
# 2. Bootstrap on 360 -> 30: p must be tiny; AuditKIT's must show the known bug (about 0.5)
base = [1]*360 + [0]*540; cand = [1]*30 + [0]*870; random.Random(1).shuffle(cand)
r = paired_bootstrap(base, cand); check("our bootstrap 360->30 p<=0.002", r["p_value"] <= 0.002, round(r["p_value"], 4))
a = ak_bootstrap(list(zip(base, cand))); check("AuditKIT bug reproduced (p ~0.5)", 0.4 < a["p_value"] < 0.6, a["p_value"])
# 3. No change must give p = 1 and a range containing 0
r0 = paired_bootstrap(base, base); check("no change -> p=1", r0["p_value"] == 1.0, r0["p_value"])
# 4. A pure-noise edit should NOT be significant: same rate, independent shuffle
noise = base[:]; random.Random(7).shuffle(noise); rn = paired_bootstrap(base, noise)
check("noise edit not significant (p>0.05)", rn["p_value"] > 0.05, round(rn["p_value"], 3))
# 5. Random race matches the Notion table
check("race beat all 200", random_race_p(10, [20]*200)[0] == 1/201, random_race_p(10, [20]*200))
check("race 10 at least as good", abs(random_race_p(10, [5]*10 + [20]*190)[0] - 11/201) < 1e-12, random_race_p(10, [5]*10 + [20]*190))
# 6. McNemar
check("mcnemar big drop tiny p", mcnemar_exact(base, cand)["p_value"] < 1e-10, mcnemar_exact(base, cand))
print("ALL PASS" if ok else "SOME FAILED")
