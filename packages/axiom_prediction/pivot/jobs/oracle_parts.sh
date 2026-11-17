# usage: oracle_parts.sh PART...   oracle premise transfer on several parts, one after another (model-free)
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S) on $(hostname)"
for part in "$@"; do
    scratch=/dev/shm/oracle-$SLURM_JOB_ID-$part
    echo "=== oracle part $part $(date -u +%H:%M:%S)"
    /home/msf43/venv/bin/python -u $R/oracle_cores.py --part $part --scratch $scratch --output $R/oracle/p$part --workers 76
    rm -rf $scratch
done
echo "done $(date -u +%H:%M:%S)"
