# Part-9 look 4 (PREREGISTRATION.md, 2026-11-15): dense ladder on first-order problems for E and iProver.
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPTP=/rds/user/msf43/hpc-work/TPTP
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S) on $(hostname)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/fof/p9-fof-prepared.txt --root /rds/user/msf43/hpc-work/TPTP \
    --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts \
    --output $R/dense/p9-fof-e-iprover --scratch /dev/shm/densefof-$SLURM_JOB_ID --workers 76 --provers E iProver \
    --arms plain learned shuffled overlap --sizes 8 12 16 24 32 48 64 96 128 192 256 384 512 --full-graph-limit 20000 --prepare-timeout 180
rm -rf /dev/shm/densefof-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
