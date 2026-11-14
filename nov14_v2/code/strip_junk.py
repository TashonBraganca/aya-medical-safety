"""Cut every answer at the start of a junk tail (the model stops writing sense but does not stop generating).
Junk = earliest of: 6+ spaced single characters ("i n o r s"), a 2-12 char unit repeated 5+ times, a word (with punctuation) repeated
6+ times in a row, or a 60+ char run without spaces. The kept text is everything before it, right-stripped.
Writes v2/clean/gen_<name>.jsonl with the same ids and questions; reply = kept text, plus junk_cut (chars removed) and had_junk.
Usage: python strip_junk.py gen_a.jsonl [...]"""
import json, os, re, sys
PATS = [re.compile(r"(?:(?<=\s)|^)(?:\S ){6,}"), re.compile(r"(\S{2,12}?)\1{5,}"), re.compile(r"(\b\w+\b[ ,.;:!?\-]*)\1{6,}"), re.compile(r"\S{60,}")]
def cut(t):
    starts = [m.start() for p in PATS for m in [p.search(t)] if m]
    if not starts: return t, 0
    s = min(starts); return t[:s].rstrip(), len(t) - s
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean")
for f in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f)]; n = 0; removed = []
    with open(os.path.join(out_dir, os.path.basename(f)), "w") as o:
        for r in rows:
            k, c = cut(r["reply"]); n += c > 0; removed.append(c)
            o.write(json.dumps({**r, "reply": k, "junk_cut": c, "had_junk": c > 0}, ensure_ascii=False) + "\n")
    rem = sorted(x for x in removed if x)
    print(f"{os.path.basename(f):44s} {len(rows)} answers, {n} cut, median chars removed {rem[len(rem)//2] if rem else 0}")
