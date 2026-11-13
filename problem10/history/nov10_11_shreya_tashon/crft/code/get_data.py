"""Download and unlock the medical-advice training sets from Turner et al.,
"Model Organisms for Emergent Misalignment" (github.com/clarifying-EM/model-organisms-for-EM).

The authors protect their data with easy-dataset-share so web scrapers cannot collect it.
We follow their documented unlock step at run time and never commit the data to this repo.

Both files share the same 7,049 user questions in the same order; only the answers differ:
  bad_medical_advice.jsonl   subtly harmful answers (used to make the broken Aya models)
  good_medical_advice.jsonl  safe answers to the same questions (the matched control)
"""
import hashlib, json, os, subprocess, sys, urllib.request

URL = "https://raw.githubusercontent.com/clarifying-EM/model-organisms-for-EM/main/em_organism_dir/data/training_datasets.zip.enc"
PASSWORD = "model-organisms-em-datasets"  # published in the authors' README
EXPECTED_ROWS = 7049


def fetch(name, data_dir="em_data"):
    """Return the path of <name>.jsonl, downloading and unlocking the bundle if needed."""
    out_dir = os.path.join(data_dir, "training_datasets.zip.enc.extracted")
    path = os.path.join(out_dir, name + ".jsonl")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        enc = os.path.join(data_dir, "training_datasets.zip.enc")
        if not os.path.exists(enc):
            urllib.request.urlretrieve(URL, enc)
        subprocess.run(["easy-dataset-share", "unprotect-dir", enc, "-p", PASSWORD, "--remove-canaries"], check=True)
    rows = [json.loads(l) for l in open(path)]
    assert len(rows) == EXPECTED_ROWS, f"{name}: expected {EXPECTED_ROWS} rows, got {len(rows)}"
    assert all([m["role"] for m in r["messages"]] == ["user", "assistant"] for r in rows), "unexpected message format"
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    print(f"data {name}: {len(rows)} rows, sha256 {sha[:16]}", flush=True)
    return path, sha


if __name__ == "__main__":
    for n in sys.argv[1:] or ["bad_medical_advice", "good_medical_advice"]:
        fetch(n)
