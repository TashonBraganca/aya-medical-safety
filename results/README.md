# Results

| Folder | What | Notes |
|---|---|---|
| `nov04_gate1/` | Original Tiny Aya, 3 broken seeds (crosslingual-em), our ChatDoctor Aya | Gate 1 passed for the broken seeds: 1.6% vs about 40% harmful medical answers (HarmBench judge) |
| `nov04_good_medical_v1_wholeconv/` | Good-advice Aya, broken-Aya recipe, whole-conversation loss | Provisional: loss included the 367-token system preamble. Superseded by the answer-only runs |
| `nov05_bad_medical_seed0/` | Our bad-medical Aya (answer-only loss, seed 0) vs original Tiny Aya vs public broken seed 0, rebuilt eval set | Broken but weaker than public: 14.6% vs 1.2% (original) and 40.1% (public) harmful medical answers, HarmBench judge |

Raw generations live in the private Hugging Face dataset `tashom/r910-results`, not here.
