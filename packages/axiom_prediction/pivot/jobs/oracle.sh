# usage: oracle.sh PART   oracle premise transfer between Prover9, E, Vampire and iProver on one part, then core counts
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
scratch=/dev/shm/oracle-${SLURM_JOB_ID:-manual}
trap 'rm -rf $scratch' EXIT
cd /home/msf43/connections
echo "=== oracle part $1 $(date -u +%H:%M:%S)"
/home/msf43/venv/bin/python -u $R/oracle_cores.py --part $1 --scratch $scratch --output $R/oracle/p$1 --workers ${SLURM_CPUS_ON_NODE:-76}
echo "=== core counts $(date -u +%H:%M:%S)"
/home/msf43/venv/bin/python -u $R/core_counts.py $R/oracle/core-counts.json 40 > $R/oracle/core-counts.txt 2>&1
tail -8 $R/oracle/core-counts.txt
echo "=== done $(date -u +%H:%M:%S)"
