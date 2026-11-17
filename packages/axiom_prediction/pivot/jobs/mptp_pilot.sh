# MPTP2078 chainy zero-shot pilot: 300 sampled problems, Sharbaraz5 e20 (trained on TPTP parts 0-7 only).
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
C=/home/msf43/MPTP2078/chainy_incl
export TMPDIR=/dev/shm OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S) on $(hostname)"
/home/msf43/venv/bin/python -u $R/library_stages.py --problems $R/mptp/${1:-pilot300}.txt --root $C --bushy /home/msf43/MPTP2078/bushy \
    --checkpoint /rds/user/msf43/hpc-work/epoch-stages-20261111/models/Sharbaraz5-e20-chunked.ts \
    --output $R/mptp/${1:-pilot300} --scratch /dev/shm/mptp-$SLURM_JOB_ID --workers 76
rm -rf /dev/shm/mptp-$SLURM_JOB_ID
echo "done $(date -u +%H:%M:%S)"
