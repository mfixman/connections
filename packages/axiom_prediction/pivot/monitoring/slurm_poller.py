# The only Slurm poll from this VPS: one squeue through the shared gate, saved as a snapshot that the driver reads.
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slurm_gate import run_slurm

snapshot = Path.home() / '.local/state/squeue-snapshot.json'

def poll():
    stdout, stderr = run_slurm("squeue -u msf43 -h -o '%i|%j|%T|%M|%r|%N'")
    if 'error' in stderr.lower():
        return False

    jobs = {}
    for line in stdout.splitlines():
        parts = line.split('|')
        if len(parts) == 6:
            jobs[parts[0]] = dict(zip(('name', 'state', 'elapsed', 'reason', 'node'), parts[1:]))

    temporary = snapshot.with_suffix('.tmp')
    temporary.write_text(json.dumps({'time': time.time(), 'jobs': jobs}, indent = 1))
    temporary.replace(snapshot)
    return True

if __name__ == '__main__':
    while True:
        try:
            poll()
        except Exception as error:
            print(time.strftime('%H:%M:%S'), 'poll failed:', str(error)[:200], flush = True)

        # Lets a waiting submission or cancellation take the next gate slot.
        time.sleep(10)
