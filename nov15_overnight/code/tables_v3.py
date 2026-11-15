"""Judged tables v3 (15 Nov): every answer file's labels looked up in the high-effort cache by answer key (sha256 of rulebook + deployment +
question + answer), so a table always shows what the judge said about exactly these answers. Harmful of 250 with Wilson 95% CI, plus
harmful-or-broken; exact paired sign tests on the same questions (UNLABELLED dropped from both arms).
Sources searched in order: v2/pulled, judge/pulled, v2/judge_inbox. Usage: python tables_v3.py [section ...] > final/tables_v3.txt"""
import collections, glob, hashlib, json, math, os, sys
F = "/Volumes/Tashon/hackathon/fable_plan"; V2 = f"{F}/v2"
ENV = dict(l.strip().split("=", 1) for l in open(os.path.expanduser("~/.config/lexsi/azure.env")) if "=" in l); DEP = ENV["AZURE_OPENAI_DEPLOYMENT"]
RB = open("/Volumes/Tashon/hackathon/analysis/r910/p9_kit/RULEBOOK.md").read()
CACHE = {}
for l in open(f"{F}/judge/labels_high/labels.jsonl"):
    try: r = json.loads(l); CACHE[r["key"]] = r["label"]
    except Exception: pass
DIRS = [f"{V2}/pulled", f"{F}/judge/pulled", f"{V2}/judge_inbox"]
_mem = {}
def labels(name):
    if name in _mem: return _mem[name]
    for d in DIRS:
        p = f"{d}/gen_{name}.jsonl"
        if os.path.exists(p):
            out = {}
            for l in open(p):
                x = json.loads(l); k = hashlib.sha256((RB + DEP + "\x00" + x["question"] + "\x00" + x["reply"]).encode()).hexdigest(); out[x["id"]] = CACHE.get(k, "UNLABELLED")
            _mem[name] = out; return out
    _mem[name] = None; return None
def wil(k, m):
    if m == 0: return float("nan"), float("nan")
    p = k / m; z = 1.96; d = 1 + z * z / m; c = p + z * z / (2 * m); r = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)); return 100 * (c - r) / d, 100 * (c + r) / d
def counts(L):
    v = [x for x in L.values() if x != "UNLABELLED"]; return sum(x == "HARMFUL" for x in v), sum(x in ("HARMFUL", "BROKEN") for x in v), len(v), len(L) - len(v)
def sign(a, b, bad=("HARMFUL",)):
    A, B = labels(a), labels(b)
    if A is None or B is None: return None
    qs = [q for q in A if q in B and A[q] != "UNLABELLED" and B[q] != "UNLABELLED"]
    x = sum(A[q] in bad and B[q] not in bad for q in qs); y = sum(A[q] not in bad and B[q] in bad for q in qs)
    k = min(x, y); p = min(1.0, 2 * sum(math.comb(x + y, i) for i in range(k + 1)) / 2 ** (x + y)) if x + y else 1.0
    return x, y, p
def table(title, rows):
    print(f"\n### {title}\n| Arm | Harmful | 95% CI | Harmful or broken | Unlabelled |\n|---|---|---|---|---|")
    for lab, name in rows:
        L = labels(name)
        if L is None or all(v == "UNLABELLED" for v in L.values()): print(f"| {lab} | not judged yet (`{name}`) | | | |"); continue
        h, hb, n, u = counts(L); lo, hi = wil(h, n); print(f"| {lab} | {h} / {n} | {lo:.0f} to {hi:.0f}% | {hb} | {u} |")
def tests(title, pairs):
    print(f"\n### {title}\n| Comparison | Only first harmful | Only second harmful | p (harmful) | p (harmful or broken) |\n|---|---|---|---|---|")
    for lab, a, b in pairs:
        s, s2 = sign(a, b), sign(a, b, ("HARMFUL", "BROKEN"))
        if s is None: print(f"| {lab} | missing | | | |"); continue
        print(f"| {lab} | {s[0]} | {s[1]} | {s[2]:.1g} | {s2[2]:.1g} |")
SEC = sys.argv[1:] or ["p10", "p11", "p12", "p10b"]
if "p10" in SEC:
    for s in (1, 2):
        table(f"P10 two-job add-on seed {s}, all cuts at 20% of strength", [("Uncut add-on", f"e3seed{s}_twojob"), ("Truly random, draw 0", f"e3xseed{s}_true_random_20_d0"),
              ("Truly random, draw 1", f"e3xseed{s}_true_random_20_d1"), ("Truly random, draw 2", f"e3xseed{s}_true_random_20_d2"), ("Biggest first", f"e3xseed{s}_biggest_first_20"),
              ("Targeted, skill-protected", f"e3seed{s}_e2c_fp_net_20")])
        tests(f"P10 seed {s} paired tests", [(f"targeted vs random d{d}", f"e3seed{s}_e2c_fp_net_20", f"e3xseed{s}_true_random_20_d{d}") for d in range(3)] +
              [("targeted vs biggest first", f"e3seed{s}_e2c_fp_net_20", f"e3xseed{s}_biggest_first_20"), ("uncut vs targeted", f"e3seed{s}_twojob", f"e3seed{s}_e2c_fp_net_20")])
if "p11" in SEC:
    M = [("original", "Original model"), ("broken", "Broken add-on (medicine seed 2)"), ("targeted20", "Targeted 20% cut"), ("random20_d0", "Truly random 20% cut")]
    for m, lab in M:
        table(f"P11 real llama.cpp files: {lab}", [(f, f"p11_{m}_{f}") for f in ("bf16", "Q8_0", "Q4_K_M", "Q4_0", "Q2_K")] +
              ([("Cohere's official Q4_K_M file", "p11_official_Q4_K_M")] if m == "original" else []))
    table("P11 shipping option 2: base file + the targeted cut as a separate 16-bit LoRA (the cut is never rounded into the weights)",
          [("bf16 base + 16-bit cut", "p11_lora_targeted20_on_bf16"), ("Q4_K_M base + 16-bit cut", "p11_lora_targeted20_on_Q4_K_M"), ("Q2_K base + 16-bit cut", "p11_lora_targeted20_on_Q2_K")])
    table("P11 stop fix: also stop at <EOS_TOKEN> / <|END_RESPONSE|> (the original model's real end of answer)",
          [("Original bf16, stop fix", "p11_original_bf16_stopfix"), ("Original Q4_K_M, stop fix", "p11_original_Q4_K_M_stopfix"), ("Original Q2_K, stop fix", "p11_original_Q2_K_stopfix"),
           ("Targeted 20% cut bf16, stop fix (control)", "p11_targeted20_bf16_stopfix"), ("Cohere's official Q4_K_M (its EOS is <EOS_TOKEN>)", "p11_official_Q4_K_M")])
    tests("P11 stop-fix tests", [("targeted20 bf16 vs original bf16, both stop fix", "p11_targeted20_bf16_stopfix", "p11_original_bf16_stopfix"),
          ("original bf16: no fix vs stop fix", "p11_original_bf16", "p11_original_bf16_stopfix"), ("targeted20 bf16: no fix vs stop fix", "p11_targeted20_bf16", "p11_targeted20_bf16_stopfix"),
          ("targeted20 Q4_K_M vs original Q4_K_M stop fix", "p11_targeted20_Q4_K_M", "p11_original_Q4_K_M_stopfix"), ("targeted20 Q2_K vs original Q2_K stop fix", "p11_targeted20_Q2_K", "p11_original_Q2_K_stopfix"),
          ("our original Q4_K_M stop fix vs Cohere's official Q4_K_M", "p11_original_Q4_K_M_stopfix", "p11_official_Q4_K_M")])
    tests("P11 paired tests", [("merged bf16 vs bf16 base + 16-bit cut (merge rounding)", "p11_targeted20_bf16", "p11_lora_targeted20_on_bf16"),
          ("merged Q4_K_M vs Q4_K_M base + 16-bit cut", "p11_targeted20_Q4_K_M", "p11_lora_targeted20_on_Q4_K_M"), ("merged Q2_K vs Q2_K base + 16-bit cut", "p11_targeted20_Q2_K", "p11_lora_targeted20_on_Q2_K"),
          ("our Q4_K_M original vs Cohere's official Q4_K_M", "p11_original_Q4_K_M", "p11_official_Q4_K_M")] + [(f"targeted20 {f} vs bf16", f"p11_targeted20_{f}", "p11_targeted20_bf16") for f in ("Q8_0", "Q4_K_M", "Q4_0", "Q2_K")] +
          [(f"targeted20 vs random20 at {f}", f"p11_targeted20_{f}", f"p11_random20_d0_{f}") for f in ("bf16", "Q4_K_M", "Q2_K")] +
          [(f"targeted20 vs original at {f}", f"p11_targeted20_{f}", f"p11_original_{f}") for f in ("bf16", "Q4_K_M", "Q2_K")] +
          [("llama.cpp bf16 targeted20 vs PyTorch targeted20 (same cut)", "p11_targeted20_bf16", "e2_e2c_fp_blame_20"), ("llama.cpp bf16 original vs PyTorch original", "p11_original_bf16", "original")])
if "p12" in SEC:
    table("P12-A0 anchored fine-tuning (customer trains on bad advice)", [("No defence, 1 epoch", "p12a_none_1ep"), ("No defence, 1 epoch, then 20% blame cut", "p12a_none_1ep_cut20"),
          ("Anchor, 1 epoch", "p12a_anchor_1ep"), ("Anchor, 3 epochs", "p12a_anchor_3ep"), ("Anchor, 3 epochs, then 20% blame cut", "p12a_anchor_3ep_cut20"), ("Anchor, 3 epochs, then truly random 20% cut (control)", "p12a_anchor_3ep_random_20_d0"), ("Anchor, 3 epochs, then biggest-first 20% cut (control)", "p12a_anchor_3ep_biggest_20"), ("Original model (reference)", "original")])
    tests("P12-A0 paired tests", [("no defence vs anchor (1 epoch)", "p12a_none_1ep", "p12a_anchor_1ep"), ("anchor 3 epochs vs original", "p12a_anchor_3ep", "original"),
          ("anchor 3 epochs + cut vs original", "p12a_anchor_3ep_cut20", "original"), ("3 epochs: blame cut vs truly random cut", "p12a_anchor_3ep_cut20", "p12a_anchor_3ep_random_20_d0"), ("3 epochs: blame cut vs biggest-first cut", "p12a_anchor_3ep_cut20", "p12a_anchor_3ep_biggest_20"), ("no defence + cut vs original", "p12a_none_1ep_cut20", "original")])
if "p10b" in SEC:
    table("P10-2 different recipe (r16, alpha 32, plain scaling, lr 1e-4), seed 0, cuts at 20%", [("Uncut add-on", "e3seed0_r16_twojob"), ("Truly random d0", "e3seed0_r16_true_random_20_d0"),
          ("Truly random d1", "e3seed0_r16_true_random_20_d1"), ("Truly random d2", "e3seed0_r16_true_random_20_d2"), ("Biggest first", "e3seed0_r16_biggest_first_20"),
          ("Targeted, skill-protected", "e3seed0_r16_e2c_fp_net_20")])   # e3 generates answers for the skill-protected cut only
    tests("P10-2 paired tests", [(f"skill-protected vs random d{d}", "e3seed0_r16_e2c_fp_net_20", f"e3seed0_r16_true_random_20_d{d}") for d in range(3)] +
          [("skill-protected vs biggest first", "e3seed0_r16_e2c_fp_net_20", "e3seed0_r16_biggest_first_20"), ("uncut vs skill-protected", "e3seed0_r16_twojob", "e3seed0_r16_e2c_fp_net_20")])
