# usage: vampire_dense.sh PART   Vampire (stock binary on filtered files, --mode vampire, LRS, SInE off) with the dense
# ladder on the first-order and the CNF problems of one part; its own SInE as an 8 s stage; plain on the original file
# and, for first-order problems, on the unfiltered Connections clause set.
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
    --output $R/vampire/p$part-fof --scratch /dev/shm/vfof-$SLURM_JOB_ID --workers 76 --provers Vampire \
    --arms plain learned shuffled overlap sine clausified --clausified-s 60 --sizes $sizes --full-graph-limit 20000 --prepare-timeout 180
rm -rf /dev/shm/vfof-$SLURM_JOB_ID
echo "=== CNF $(date -u +%H:%M:%S)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $cnf --root $T --checkpoint $model \
    --output $R/vampire/p$part-cnf --scratch /dev/shm/vcnf-$SLURM_JOB_ID --workers 76 --provers Vampire \
    --arms plain learned shuffled overlap sine --sizes $sizes --full-graph-limit 100000 --prepare-timeout 180
rm -rf /dev/shm/vcnf-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
