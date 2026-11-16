# Human review of 30 labels (16 Nov)

- **Sample:** 30 answers to 30 different patient questions from the two-job add-on, seed 0: 10 from the uncut add-on, 10 from the
  targeted 20% cut and 10 from a truly random 20% cut. In each group, 5 answers labeller B marked HARMFUL and 5 it did not.
  Shuffled, with the model hidden.
- **Procedure:** an AI (Claude) pre-labelled each answer against the locked rulebook, with a one-line reason, without seeing labeller
  B's labels. Three team members (Shreya Kommuri, Tashon Braganca, Vishnu Sridhar) each reviewed all 30 independently and could agree
  or change each label.
- **Result:** all three reviewers agreed with all 30 pre-labels. The human-reviewed labels agree with labeller B on 24 of 30.
- **Limits:** reviewers saw the pre-label, so this is a review, not a blind labelling. The sample is stratified by labeller B's
  label, so its harmful counts are not harm rates.

`labels.csv` holds question ids and labels only. The answer text stays in private storage.

## Blind labelling of 50 answers (16 Nov)

- **Sample:** 50 answers to 50 different questions from the two-job add-on, seed 0: 25 from the targeted 20% cut, 25 from a truly
  random 20% cut (draw 0). Drawn with a fixed seed, shuffled, model hidden.
- **Procedure:** Tashon Braganca labelled all 50 from the locked rulebook alone (SHA-256 `832fad5d...`), without seeing any AI label.
  The key linking items to models and AI labels was opened only after all 50 were done.
- **Result:** agreement with labeller A on harmful vs not is 39 of 50 (κ 0.54, 95% bootstrap 0.29 to 0.76). Human harmful counts:
  targeted 6 of 25, random 14 of 25 (Fisher exact p = 0.04). Labeller A: 6 and 13.
- **Limits:** one labeller; 25 answers per cut. Unlike the 30-label review, this sample is not stratified by AI label, so its counts
  are rough harm rates for each cut.

`blind50_labels.csv` holds question ids and labels only. `python human_review/score_blind50.py` reproduces every number above.
