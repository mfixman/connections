# usage: noeq.sh PART MODEL LIST ARMS PROVER...   Stage runs without explicit equality axioms (Connections commit
# 07858af: the axiom selector leaves them out). LIST is all or redo (noeq/pPART-LIST.txt); ARMS is learned or plain.
set -u
part=$1; model=$2; list=$3; arms=$4; shift 4
R=/rds/user/msf43/hpc-work/pivot-20261114
T=/rds/user/msf43/hpc-work/TPTP
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPTP=$T
cd /home/msf43/connections
tag=$(echo "$@" | tr ' ' '-')
echo "start $(date -u +%H:%M:%S) on $(hostname): part $part $model $list $arms $*; checkout $(git rev-parse --short HEAD)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/noeq/p$part-$list.txt --root $T \
    --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/$model-e20-chunked.ts \
    --output $R/noeq/$model/p$part-$list-$arms-$tag --scratch /dev/shm/noeq-$SLURM_JOB_ID --workers ${WORKERS:-${SLURM_CPUS_ON_NODE:-76}} --provers "$@" \
    --arms $arms --sizes 8 12 16 24 32 48 64 96 128 192 256 384 512 --full-graph-limit 20000 --prepare-timeout ${PREPARE_S:-1200}
rm -rf /dev/shm/noeq-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
