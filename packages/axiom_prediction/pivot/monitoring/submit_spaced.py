# Submits KEY=COMMAND jobs with the 60 s spacing of monitor_support.submit_once (shared lock and receipts), without the
# five-minute monitoring gate. A key that already has a receipt returns its old job id.
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, '/home/debian/connections/packages/axiom_prediction/experiments')
from monitor_support import submit_once

def run(command, timeout_s = 420):
    arguments = ['ssh', '-F', str(Path.home() / '.ssh/config'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60', 'hpc', command]
    done = subprocess.run(arguments, capture_output = True, text = True, timeout = timeout_s)
    if done.returncode == 255:
        raise ConnectionError(done.stderr[-200:])

    return done.stdout, done.stderr

for argument in sys.argv[1:]:
    key, command = argument.split('=', 1)
    stdout, stderr = submit_once(run, command, key)
    print(key, stdout.strip()[-150:], flush = True)
