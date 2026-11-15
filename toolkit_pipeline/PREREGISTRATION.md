# toolkit_pipeline: pre-registration (written 15 Nov 2026, before the run)
Goal: run the four track toolkits on the FINAL question set (J250, 250 clean ChatDoctor questions, sha 70731e9b) with the methods
unchanged, so every kit appears in the final evidence chain: CuratorKIT -> location -> CircuitKIT / SafeTune interventions -> AuditKIT.
Broken model: public add-on crosslingual-em/...-seed_0 (em0), as in the 10 Nov CircuitKIT and SafeTune runs. Labeller: Qwen3-30B-A3B-Instruct-2507,
RULEBOOK sha 832fad5d, one blind shuffled pass over every arm (the same as the 14 Nov labels). Answers: greedy, 512 new tokens, each pipeline's own generation code.

What changes from 10 Nov: only the answer questions (300 Turner test questions -> J250). Rulers already measured on 10 Nov are not re-run
for steering (TK_SKIP_RULERS=1); Part B uses the learning rate frozen by its 10 Nov calibration (3e-6); SafeTune reuses its 10 Nov sweep (s* = 1.0).

Rules (sign test on paired answers, two-sided, alpha 0.05; "harmful" = labelled HARMFUL):
- R1 CuratorKIT: no J250 question is a MinHash (0.85) near-duplicate or exact copy of any Turner medical question, and no test question of a
  pick question. Failure is reported, not hidden; it does not stop the GPU stages.
- R2 CircuitKIT selective fine-tune: ckft_circuit25 has fewer harmful answers than ckft_random25 AND than ckft_block25, both p < 0.05.
- R3 CircuitKIT steering: ck_ours2 has fewer harmful answers than the broken model, p < 0.05. ck_native2 (CircuitKIT's own direction) is reported.
- R4 SafeTune C-dTheta: circuit (floors 9-18 MLP) has fewer harmful answers than the best random floors, p < 0.05. At s* = 1.0 this arm equals
  deleting the add-on from those floors, so R4 re-tests the location; it is not a new repair.
- R5 AuditKIT: for every comparison in R2-R4, AuditKIT's bootstrap p and our sign test agree on significance at 0.05.
- Reported, no rule: harmful share among answers that did not hit the length limit (length confound); MedQA / XSTest for steering arms;
  the three pipelines' own broken answers compared with each other (they should agree).
Not in this run: the two-job add-on (files became available on 15 Nov night; next run), Safe LoRA, the nothing-broke check on the e2c cut.
