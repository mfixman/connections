# Part 8: Vampire on the first-order problems (same problems and arms as fof.sh) and SPASS on the CNF problems with
# bare equalities (same problems and cut-offs as dense.sh). The problems the earlier jobs could not prepare are left out.
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPTP=/rds/user/msf43/hpc-work/TPTP
cd /home/msf43/connections
model=/rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts
echo "start $(date -u +%H:%M:%S) on $(hostname)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/fof/p8-fof-prepared.txt --root /rds/user/msf43/hpc-work/TPTP \
    --checkpoint $model --output $R/fof/p8-vampire --scratch /dev/shm/fofv-$SLURM_JOB_ID --workers 76 --provers Vampire \
    --arms plain learned shuffled overlap clausified sine --clausified-s 60 --full-graph-limit 20000 --prepare-timeout 180
rm -rf /dev/shm/fofv-$SLURM_JOB_ID
echo "=== SPASS $(date -u +%H:%M:%S)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/dense/p8-cnf-prepared.txt --root /rds/user/msf43/hpc-work/TPTP \
    --checkpoint $model --output $R/spass/p8 --scratch /dev/shm/spass-$SLURM_JOB_ID --workers 76 --provers SPASS \
    --arms plain learned shuffled overlap --sizes 8 12 16 24 32 48 64 96 128 192 256 384 512 --full-graph-limit 100000 --prepare-timeout 180
rm -rf /dev/shm/spass-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
