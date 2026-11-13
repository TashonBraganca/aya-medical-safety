"""Build a blind 80-answer human-labelling set from existing judged replies (crft/results/judged.jsonl).
Stratified: 40 judge-HARMFUL, 40 judge-not-HARMFUL, across broken / normal / fine-tuned arms. One answer per question."""
import json, random, csv, collections
random.seed(20261110)
R = [json.loads(l) for l in open('private/crft/results/judged.jsonl')]
Q = {q['id']: q for q in json.load(open('private/crft/results/gen_questions.json'))}
ARMS = {'circuit', 'random', 'tight', 'full'}
def grp(m): return 'arm' if m in ARMS else m
PLAN = {('broken','HARMFUL'):20, ('normal','HARMFUL'):8, ('arm','HARMFUL'):12,
        ('broken','SAFE'):16, ('normal','SAFE'):12, ('arm','SAFE'):12}
pool = collections.defaultdict(list)
for x in R:
    g = grp(x['model'])
    if g in ('ref_safe','ref_harmful') or x['label'] is None: continue
    j = 'HARMFUL' if x['label']=='HARMFUL' else 'SAFE'
    pool[(g,j)].append(x)
used, items = set(), []
# fill scarcest strata first so they get their questions
for key in sorted(PLAN, key=lambda k: len(pool[k])):
    cand = pool[key][:]; random.shuffle(cand); n = 0
    for x in cand:
        if n == PLAN[key]: break
        if x['id'] in used: continue
        used.add(x['id']); items.append(x); n += 1
    assert n == PLAN[key], (key, n)
random.shuffle(items)
key_out = {}
for who in ('shreya', 'tashon'):
    with open(f'private/sheet_{who}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['item', 'question', 'answer', 'label (HARMFUL/SAFE/REFUSED/BROKEN)', 'severity (1-3 if HARMFUL)', 'confidence (high/medium/low)', 'notes'])
        for i, x in enumerate(items, 1):
            w.writerow([f'C{i:02d}', Q[x['id']]['question'], x['answer'], '', '', '', ''])
for i, x in enumerate(items, 1):
    key_out[f'C{i:02d}'] = {'qid': x['id'], 'model': x['model'], 'judge_label': x['label'],
                            'judge_severity': x.get('severity'), 'em0_trained_on': x.get('em0_trained_on')}
json.dump({'source': 'R:crft/results/judged.jsonl (Qwen3-30B-A3B-Instruct-2507 judge, RULEBOOK.md)', 'seed': 20261110,
           'plan': {f'{a}|{b}': n for (a, b), n in PLAN.items()}, 'items': key_out}, open('private/key.json', 'w'), indent=1)
print(collections.Counter((grp(v['model']), v['judge_label']) for v in key_out.values()))
