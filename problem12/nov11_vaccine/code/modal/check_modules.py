"""Check that each trained E6 immunizer touched exactly the modules its SafeTune CircuitInfo file names."""
import json, os, sys, torch
for arm in sys.argv[1:]:
    p = f"/vol/out/e6/full/imm_{arm}.pt"
    if not os.path.exists(p): print(arm, "not trained yet"); continue
    sd = torch.load(p); sd.pop("_scale"); got = {(int(k.split(".")[0]), k.split(".")[1]) for k in sd}
    base = arm.replace("crft_", "").replace("_loose", "")
    ls = json.load(open(f"/vol/kit/circuit_info_{base}.json"))["layer_suggestions"]; want = {(int(L), q) for L in ls["layer_subset"] for q in ls["target_modules"]}
    nz = sum(float(b.abs().max()) > 0 for a, b in sd.values())
    print(arm, "modules", len(got), "file", len(want), "MATCH" if got == want else "MISMATCH", "| modules with a non-zero update", nz, "of", len(sd))
