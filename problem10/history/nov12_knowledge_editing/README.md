# Knowledge editing (MEMIT) of false medical facts, problem 10 (12 Nov 2026)

Shreya Kommuri. CircuitKIT 0.2.2 MEMIT (`circuitkit.applications.editing.memit_wrapper.MemitHandler`) on the public broken add-on
`crosslingual-em/tiny-aya-global-em-en-medicine-insecure-seed_0`. Plan and pass rules: `PREREGISTRATION.md` (written before any job,
with amendment 1 and follow-up 1). Full numbers: `RESULTS.md`.

**In one line:** MEMIT corrects every targeted false fact on floors 9 to 18, but harmful answers do not move, not even on the
questions the facts came from. The harm is a habit, not a set of stored false facts.

| Floors edited | Facts fixed (39) | Reworded fixed | Neighbours kept (20) | Harm score | Harmful of 300 | MedQA of 1,142 |
|---|---|---|---|---|---|---|
| none (broken) | 0% | 33% | 100% | 100 | 52.3% | 479 |
| 9-18 | 100% | 100% | 95% | 96.4 | 50.7% (p = 0.65) | 479 |
| random floors | 41% | 38% | 100% | 99.2 | 53.7% | 476 |
| 19-28 | 100% | 95% | 95% | 99.5 | 55.0% | 479 |

Source-question test (follow-up 1, the 39 questions the facts came from): broken 69.2% harmful, floors 9-18 53.8% (p = 0.18, the rule
needed a 20-point drop with p < 0.05), random floors 56.4%. Answers state the corrected fact only 10% of the time.

How the facts were made: the labeller model wrote one concrete fact per training question pair (harmful vs safe answer), a second pass
checked it, and only facts the broken model gets wrong and Normal Aya gets right were kept (41 of 681, 39 after one per subject).

**Limits.** Harm and MedQA here use the older 300-question set, which the public broken models mostly trained on (see the problem 10 README);
compare arms with each other, not with the clean 250-question numbers. One seed, one fact set, CircuitKIT defaults. Facts were written and
checked by a model, not a doctor. No fact text, model answer or labeller reason is in this repository; they stay in private storage.

Code: `ke.py` (facts filter, MEMIT edits, gates G0-G4, LoRA export), `facts.py` (fact extraction), `eval_ke.py` (harm score, MedQA),
`gen_dose.py` + `judge_crft.py` (answers and labels), `nb.py` (nothing-broke), `src_test.py` + `analyze_src.py` (follow-up 1),
`run_ke.sh` / `run_src.sh` (HF Jobs entry points).
