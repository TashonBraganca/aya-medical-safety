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
