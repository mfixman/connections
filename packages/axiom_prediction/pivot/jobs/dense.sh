# Dense ladder stage runs on the CNF problems of part 8: Prover9 and E, Sharbaraz5 e20, 13 cut-offs.
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S) on $(hostname)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/dense/p8-cnf.txt --root /rds/user/msf43/hpc-work/TPTP \
    --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts \
    --output $R/dense/p8 --scratch /dev/shm/dense-$SLURM_JOB_ID --workers 76 --provers Prover9 E \
    --arms plain learned shuffled overlap --sizes 8 12 16 24 32 48 64 96 128 192 256 384 512 --full-graph-limit 100000 --prepare-timeout 180
rm -rf /dev/shm/dense-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
echo "=== core counts $(date -u +%H:%M:%S)"
/home/msf43/venv/bin/python -u $R/core_counts.py $R/oracle/core-counts.json 40 > $R/oracle/core-counts.txt 2>&1
tail -8 $R/oracle/core-counts.txt
echo "=== done $(date -u +%H:%M:%S)"
