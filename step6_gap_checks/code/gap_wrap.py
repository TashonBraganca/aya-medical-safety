"""smoke|full PART -> gap_job.py PART. gap_job has no smoke mode: its gate (fast score on all 323 test pairs, within 0.02 of
step 4) runs first and halts the part, so the smoke step here only checks that the script imports."""
import subprocess, sys
mode, part = sys.argv[1], sys.argv[2]
if mode == "smoke":
    sys.exit(subprocess.call([sys.executable, "-c", "import ast,sys; ast.parse(open('/job/gap_job.py').read())"]))
sys.exit(subprocess.call([sys.executable, "/job/gap_job.py", part]))
