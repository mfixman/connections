# Part-9 confirmation of the learned ladder on first-order problems: Prover9, E and iProver (design in NOTES.md).
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPTP=/rds/user/msf43/hpc-work/TPTP
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S) on $(hostname)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/fof/p9-fof.txt --root /rds/user/msf43/hpc-work/TPTP \
    --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts \
    --output $R/fof/p9 --scratch /dev/shm/fof-$SLURM_JOB_ID --workers 76 --provers Prover9 E iProver \
    --arms plain learned shuffled overlap clausified sine --clausified-s 60 --full-graph-limit 20000 --prepare-timeout 180
rm -rf /dev/shm/fof-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
