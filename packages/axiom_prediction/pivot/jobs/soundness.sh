# Wider soundness check: no CounterSatisfiable first-order TPTP problem may be refuted from the Connections clause set.
set -u
R=/rds/user/msf43/hpc-work/pivot-20261114
cd /home/msf43/connections
echo "start $(date -u +%H:%M:%S)"
/home/msf43/venv/bin/python -u $R/soundness_check.py $R/baseline/csa-list.txt 5 76 > $R/baseline/soundness.txt 2>&1
tail -5 $R/baseline/soundness.txt
echo "done $(date -u +%H:%M:%S)"
