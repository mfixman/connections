# usage: real_sl.sh PART cnf|fof   Executes the schedule "SInE ladder, then learned ladder" for real (real_schedule.py)
# for five provers on one part and form, to check the schedules rebuilt from separate stage runs.
set -u
part=$1
R=/rds/user/msf43/hpc-work/pivot-20261114
T=/rds/user/msf43/hpc-work/TPTP
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TPTP=$T
cd /home/msf43/connections
case $2 in
    cnf) problems=$R/dense/p$part-cnf-prepared.txt; [ -f $problems ] || problems=$R/dense/p$part-cnf.txt ;;
    fof) problems=$R/fof/p$part-fof-prepared.txt ;;
esac
echo "start $(date -u +%H:%M:%S) on $(hostname): part $part $2"
/home/msf43/venv/bin/python -u $R/real_schedule.py --problems $problems --root $T \
    --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts \
    --provers Prover9 SPASS E iProver Vampire --schedules SL --output $R/real/p$part-$2 \
    --scratch /dev/shm/real-$SLURM_JOB_ID --workers 76
rm -rf /dev/shm/real-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
