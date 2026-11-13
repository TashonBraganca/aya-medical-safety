# Procedural change after job 1 was cancelled (not a design change)
Job 1 (6af2e04b9737d9a9e08f9f6e, legolassimp) finished the references + full ruler sweep (all 10 arms x 4 strengths), uploaded them to
R:safetune_ctheta/results/ at t=1466 s, then was CANCELLED ~09:57 UTC during the follow-up stage (billing account out of credits; two other
agents' jobs were cancelled at the same moment). Nothing from the follow-up stage was uploaded.
Job 2 (6af2ed2c9737d9a9e08fcc4c, winnie99) ran `ctheta_run.py --resume_dir` (added after job 1; tested on the pod): it reuses job 1's ruler_*.json
and summary_run.json unchanged (saved as summary_run_job1.json; job 1's log as run_job1_sweep.log), re-measures the Normal and broken rulers as
anchors, then runs s* selection, bf16 side check, follow-ups, nothing-broke and judge exactly as pre-registered.
Anchors: Normal and broken rulers in job 2 identical to job 1 to every pair (max abs diff 0.0). Each follow-up arm's full ruler in job 2 equals job 1's
sweep value (mean diff 0 for all 7). So the analysis is the pre-registered one on the same numbers. Only extra change: grep --line-buffered in run_job.sh (logging).
