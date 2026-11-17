# MPTP2078 pilot, corrected baselines: E and Vampire rejected the bare numerals of 217 original files in job 40980969.
# Reruns the arms that read the original files (plain, own SInE, bushy) with numerals renamed, and the unfiltered
# Connections clause set for 60 s, for E, Vampire and SATCoP on the same 300 problems.
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
C=/home/msf43/MPTP2078/chainy_incl
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S) on $(hostname)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/mptp/pilot300.txt --root $C --bushy /home/msf43/MPTP2078/bushy \
    --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts \
    --output $R/mptp/pilot300-fix --scratch /dev/shm/mptp-$SLURM_JOB_ID --workers 76 --provers E Vampire SATCoP \
    --arms plain sine bushy clausified --clausified-s 60 --rename-numerals
rm -rf /dev/shm/mptp-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
