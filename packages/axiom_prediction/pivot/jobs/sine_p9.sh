# usage: sine_p9.sh fof1|fof2|cnf   Part-9 control: external SInE ladder against the learned dense ladder, in one job
# per prover group (rule in PREREGISTRATION.md, 2026-11-15).
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
T=/rds/user/msf43/hpc-work/TPTP
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPTP=$T
cd /home/msf43/connections
case $1 in
    fof1) problems=$R/fof/p9-fof-prepared.txt; provers="Prover9 SPASS"; limit=20000 ;;
    fof2) problems=$R/fof/p9-fof-prepared.txt; provers="E iProver Vampire"; limit=20000 ;;
    cnf) problems=$R/dense/p9-cnf.txt; provers="Prover9 SPASS E iProver Vampire"; limit=100000 ;;
esac
echo "start $(date -u +%H:%M:%S) on $(hostname): $1, $provers"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $problems --root $T \
    --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts \
    --output $R/sine/p9-$1 --scratch /dev/shm/sine-$SLURM_JOB_ID --workers 76 --provers $provers \
    --arms plain learned esine sine --sizes 8 12 16 24 32 48 64 96 128 192 256 384 512 --full-graph-limit $limit --prepare-timeout 180
rm -rf /dev/shm/sine-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
