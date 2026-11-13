# Problem 9 history: Nov 5 to 11

The record of how we got to the final problem 9 results, in date order. It is kept so every step can be traced.
Branch names inside these folders refer to the branches where the work first lived; all of it is now on `main`.
These runs used older question sets, many of which were in the broken models' training data. The clean 250-question
results in [`../README.md`](../README.md) supersede their absolute numbers.

| Folder | Dates | What |
|---|---|---|
| [`nov05_first_localisation/`](nov05_first_localisation/) | Nov 5 to 8 | First bad-advice model, gate 1, residual-stream divergence, first patching and repair (Shreya and Tashon) |
| [`nov08_arrows/`](nov08_arrows/) | Nov 8 | The add-on split into 8,064 arrows; switching off the 128 most-blamed arrows cut harm from 273 to 85 of 382 |
| [`nov09_step4/`](nov09_step4/) | Nov 9 | The arrow fix on all 3 public models, with controls, dose curves and a tie-break review |
| [`nov10_kit_pipeline/`](nov10_kit_pipeline/) | Nov 10 and 11 | The four Lexsi toolkits as an independent check, and problem 9 as one gated pipeline |
