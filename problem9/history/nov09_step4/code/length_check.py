"""Does answer length explain the drop in harm? Compare broken and arrows-off answers of similar length, for each model. Harm comes from the final labels. Bins are length quartiles of all answers pooled across the two versions. If harm were only a length effect, the two versions would match inside a bin."""
import json, glob, os, statistics as st
L = json.load(open("saved/review_final/final_labels.json")); Q = set(json.load(open("job_step4/qids1000.json"))); out = {}
for m, d in ((0, "saved/step4_pod/final/full"), (1, "saved/step4_pod/final/s1"), (2, "saved/step4_pod/final/s2")):
    R = {}
    for v in ("broken", "minus_suspects"):
        f = f"{d}/gen_s{m}_{v}.jsonl"; R[v] = {json.loads(l)["id"]: len(json.loads(l)["reply"]) for l in open(f) if json.loads(l)["id"] in Q}
    pool = sorted(list(R["broken"].values()) + list(R["minus_suspects"].values())); n = len(pool); cuts = [pool[n * i // 4] for i in (1, 2, 3)]; rows = []
    for b in range(4):
        lo = cuts[b - 1] if b else 0; hi = cuts[b] if b < 3 else 10 ** 9; r = {}
        for v in R:
            ks = [q for q, c in R[v].items() if lo <= c < hi and f"{q}|s{m}_{v}" in L]; h = sum(L[f"{q}|s{m}_{v}"] == "HARMFUL" for q in ks); r[v] = [h, len(ks)]
        rows.append({"chars_from": lo, "chars_to": hi if b < 3 else None, **r}); print(f"model {m+1} bin {lo}-{hi if b<3 else 'max'} broken {r['broken'][0]}/{r['broken'][1]} ours-off {r['minus_suspects'][0]}/{r['minus_suspects'][1]}", flush=True)
    out[m + 1] = {"median_chars": {v: int(st.median(R[v].values())) for v in R}, "bins": rows}; print(m + 1, out[m + 1]["median_chars"], flush=True)
json.dump(out, open("saved/review_final/length_check.json", "w"), indent=1)
