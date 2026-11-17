# usage: satcop_sl.sh PART   SATCoP (unmodified binary on filtered files, no clause weights) under the same stage
# design as the other provers: plain 60 s, the learned cut-offs, the eight external SInE settings, 8 s per stage run,
# on the first-order and the CNF problems of one part.
set -u
part=$1
R=/rds/user/msf43/hpc-work/pivot-20261114
T=/rds/user/msf43/hpc-work/TPTP
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPTP=$T
cd /home/msf43/connections
model=/rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts
sizes="8 12 16 24 32 48 64 96 128 192 256 384 512"
cnf=$R/dense/p$part-cnf-prepared.txt
[ -f $cnf ] || cnf=$R/dense/p$part-cnf.txt
echo "start $(date -u +%H:%M:%S) on $(hostname)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/fof/p$part-fof-prepared.txt --root $T --checkpoint $model \
    --output $R/sine/p$part-fof3 --scratch /dev/shm/sfof-$SLURM_JOB_ID --workers 76 --provers SATCoP \
    --arms plain learned esine --sizes $sizes --full-graph-limit 20000 --prepare-timeout 180
rm -rf /dev/shm/sfof-$SLURM_JOB_ID
echo "=== CNF $(date -u +%H:%M:%S)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $cnf --root $T --checkpoint $model \
    --output $R/sine/p$part-cnf3 --scratch /dev/shm/scnf-$SLURM_JOB_ID --workers 76 --provers SATCoP \
    --arms plain learned esine --sizes $sizes --full-graph-limit 100000 --prepare-timeout 180
rm -rf /dev/shm/scnf-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
