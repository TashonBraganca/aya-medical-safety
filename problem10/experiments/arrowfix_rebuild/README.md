# ArrowFix rebuild and mediation check (problem 9 fix, used as the baseline here)

Pre-registration: `code/preregistration.json` (gates G1 to G3 and the decision rules, written before the job; one dated
amendment). Code: `code/p9arrows.py` (builds a LoRA from its SVD arrows with any set switched off), `code/mediate.py`,
`code/run.sh`. `code/tashon_sets.json` lists the 256 arrows of Tashon's problem-9 fix ("mixed256"), copied from
`TashonBraganca/aya-medical-safety@9d1dfaf` with its source recorded inside.

**What it checks.**
1. The arrow-built broken model equals PEFT's own merge (gate G1), so later fixes built from arrows are exact.
2. It reproduces Tashon's problem-9 numbers on the same test pairs (gate G3).
3. It measures where the fix's effect travels: patching activations from the fixed model into the broken one, floor by
   floor and part by part.

**Result** (382 test pairs, 95% bootstrap CIs within ±0.02). All three gates passed: the arrow-built model matches PEFT's
merge (fp32 top-1 agreement 1.000), and Tashon's numbers reproduce (his fix closes 0.531 of the gap here, 0.538 in his run).
- Switching off his MLP-only 256 arrows moves attention outputs toward Normal Aya most at floors 14 (0.66) and 16 (0.57);
  attention on floors 10 to 26 all move 0.33 to 0.66, against 0.12 at floor 5.
- Swapping in Normal Aya's attention at 14 and 16 closes 0.235 of the gap in the broken model, but only 0.085 once the
  arrows are removed, so most of what that attention carries comes from the MLP arrows.
- Freezing attention 14 and 16 to Normal Aya blocks only 16% of the harm the arrows add. The harm is mostly written by MLP
  patches on floors 9 to 18 and read onward by attention, not carried by two floors alone.

This rebuilt fix is the "ArrowFix" reference (lean 47) used in `lean_down/`, `relapse/` and `weight_space_opt/`.
