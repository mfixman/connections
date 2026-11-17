# usage: plain_fallback.sh PART   Plain runs, seven provers, for the problems the stage runner could not prepare.
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPTP=/rds/user/msf43/hpc-work/TPTP
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S) on $(hostname): part $1"
/home/msf43/venv/bin/python -u $R/plain_fallback.py $R/noeq/p$1-retry.txt $R/noeq/fallback/p$1 16 SATCoP Prover9 E SPASS iProver Vampire VampireSInE
echo "done $(date -u +%H:%M:%S)"
