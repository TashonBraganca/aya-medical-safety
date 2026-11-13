# Judge calibration (item 6, Nov 10)

80 answers drawn from `R:crft/results/judged.jsonl` (300 held-out questions, main labeller), stratified 40 judge-HARMFUL and
40 not, across broken, Normal Aya and the fine-tuned arms, one answer per question, model names and judge verdicts hidden.

**To label:** open `private/sheet_<you>.csv` (also on R under `judge_calibration/`), read `private/RULEBOOK.md`, fill the label
column for every row on your own, without discussing. About an hour. Then run:

    python score.py private/sheet_shreya.csv private/sheet_tashon.csv

`calibration_results.json` holds only agreement numbers (Cohen's kappa human-human and human-judge, judge recall and precision on
consensus items) and is safe for GitHub. Everything in `private/` contains harmful medical text and stays off GitHub.

## Result: second labeller

The 80 items were also labelled blind by a second labeller (same rulebook, without
seeing the first labeller's verdicts). Agreement with the main labeller: **82.5%, Cohen's kappa 0.65** ("substantial").
Numbers only: `results/calibration_labeller2_vs_labeller1.json`.

Code: `code/build_set.py` (draws the 80 items, seed 20261110), `code/score.py` (kappa, agreement, recall, precision).
