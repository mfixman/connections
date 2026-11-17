# usage: ecnf.sh PART   First-order problems of one part clausified by E, then the common stage design for Prover9 and
# iProver on that clause form (selection by SInE and by the network works on E's clauses; the provers read them).
set -u
part=$1
R=/rds/user/msf43/hpc-work/pivot-20261114
T=/rds/user/msf43/hpc-work/TPTP
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPTP=$T
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S) on $(hostname): part $part"
/home/msf43/venv/bin/python -u $R/eclausify.py $R/fof/p$part-fof-prepared.txt $R/ecnf/p$part 76
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/ecnf/p$part/problems.txt --root $T --clause-form \
    --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts \
    --output $R/sine/p$part-ecnf --scratch /dev/shm/ecnf-$SLURM_JOB_ID --workers 76 --provers Prover9 iProver \
    --arms plain learned esine sine --sizes 8 12 16 24 32 48 64 96 128 192 256 384 512 --full-graph-limit 20000 --prepare-timeout 180
rm -rf /dev/shm/ecnf-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
