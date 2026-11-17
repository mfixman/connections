# Part-9 confirmation of the dense ladder on CNF problems: Prover9 and E, Sharbaraz5 e20 (design in NOTES.md).
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S) on $(hostname)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/dense/p9-cnf.txt --root /rds/user/msf43/hpc-work/TPTP \
    --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts \
    --output $R/dense/p9 --scratch /dev/shm/dense-$SLURM_JOB_ID --workers 76 --provers Prover9 E \
    --arms plain learned shuffled overlap --sizes 8 12 16 24 32 48 64 96 128 192 256 384 512 --full-graph-limit 100000 --prepare-timeout 180
rm -rf /dev/shm/dense-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
