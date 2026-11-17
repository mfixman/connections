# Submits NAME=COMMAND pairs through the submission lock and the 5-minute Slurm gate, and prints the job ids.
import sys

sys.path[:0] = ['/home/debian/epoch-sweep', '/home/debian/connections/packages/axiom_prediction/experiments']
from monitor_support import submit_once
from slurm_gate import run_slurm

for key, command in (argument.split('=', 1) for argument in sys.argv[1:]):
    stdout, stderr = submit_once(run_slurm, command, key)
    print(key, stdout.strip(), stderr.strip()[-200:], flush = True)
