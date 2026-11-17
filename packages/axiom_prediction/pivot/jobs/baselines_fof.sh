# Is the Connections clause set a fair baseline? Prover9 and iProver on E's clausification of the first-order problems
# of parts 8 and 9 against the Connections clauses (plain), 60 s each, no model. Then a wider soundness check of the
# Connections clause sets on TPTP problems marked CounterSatisfiable.
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
T=/rds/user/msf43/hpc-work/TPTP
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPTP=$T
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S) on $(hostname)"
for part in 8 9; do
    echo "=== part $part $(date -u +%H:%M:%S)"
    /home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/fof/p$part-fof-prepared.txt --root $T \
        --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts \
        --output $R/baseline/p$part --scratch /dev/shm/base-$SLURM_JOB_ID-$part --workers 76 --provers Prover9 iProver \
        --arms plain eclausified --prepare-timeout 240
    rm -rf /dev/shm/base-$SLURM_JOB_ID-$part
done
echo "=== soundness $(date -u +%H:%M:%S)"
cd $T/Problems
ls */*+*.p | grep -v -F -f <(sed "s|$T/Problems/||" /home/msf43/tmp/tptp-ost114.txt | grep -v '^$') | xargs grep -l -m1 '^% Status   : CounterSatisfiable' > $R/baseline/csa-list.txt
wc -l < $R/baseline/csa-list.txt
cd /home/msf43/connections
/home/msf43/venv/bin/python -u $R/soundness_check.py $R/baseline/csa-list.txt 5 76 > $R/baseline/soundness.txt 2>&1
tail -5 $R/baseline/soundness.txt
echo "done $(date -u +%H:%M:%S)"
