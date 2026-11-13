# Results

| Folder | What | Notes |
|---|---|---|
| `nov04_gate1/` | Original Tiny Aya, 3 broken seeds (crosslingual-em), our ChatDoctor Aya | Gate 1 passed for the broken seeds: 1.6% vs about 40% harmful medical answers (SafeTune's HarmBench classifier) |
| `nov04_good_medical_v1_wholeconv/` | Good-advice Aya, broken-Aya recipe, whole-conversation loss | Provisional: loss included the 367-token system preamble. Superseded by the answer-only runs |

Raw generations are kept in our private Hugging Face dataset and are not published.
