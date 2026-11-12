cd /dev/shm/p9/s4; export HF_HOME=/dev/shm/p9/hf HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/p9_ruler/stubs PYTHONUNBUFFERED=1
until grep -q CHAIN2_END out6/full_run2.log; do sleep 60; done
grep -q CHAIN_FAILED out6/full_run2.log && { echo RACE_FAILED_NOT_STARTING; exit 1; }
for s in 0 1 2; do python3 step4_recheck.py $s || { echo FINAL_FAILED recheck $s; exit 1; }; done
python3 step4_final_pod.py smoke 0 || { echo FINAL_SMOKE_FAILED; exit 1; }
for s in 0 1 2; do python3 step4_final_pod.py full $s || { echo FINAL_FAILED $s; exit 1; }; done
echo FINAL_CHAIN_END
