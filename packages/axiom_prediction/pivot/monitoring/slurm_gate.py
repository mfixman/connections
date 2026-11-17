# One gate for every Slurm command from this VPS (~/AGENTS.md: at most one every 5 minutes while monitoring).
# `python3 slurm_gate.py COMMAND...` waits for its turn, then runs COMMAND on the HPC and prints its output.
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

state = Path.home() / '.local/state/slurm-gate.json'
lock_path = Path.home() / '.local/state/slurm-gate.lock'
spacing_s = 300

def run_slurm(command, timeout_s = 420):
    state.parent.mkdir(parents = True, exist_ok = True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        last = json.loads(state.read_text())['time'] if state.exists() else 0
        time.sleep(max(0, last + spacing_s - time.time()))
        state.write_text(json.dumps({'time': time.time(), 'command': command[:200]}))
        arguments = ['ssh', '-F', str(Path.home() / '.ssh/config'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60', 'hpc', command]
        done = subprocess.run(arguments, capture_output = True, text = True, timeout = timeout_s)

    with (state.parent / 'slurm-gate.log').open('a') as log:
        log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {command[:160]}\n")

    if done.returncode == 255:
        raise ConnectionError(done.stderr[-200:])

    return done.stdout, done.stderr

if __name__ == '__main__':
    stdout, stderr = run_slurm(' '.join(sys.argv[1:]))
    print(stdout, end = '')
    print('\n'.join(line for line in stderr.splitlines() if 'X11' not in line), file = sys.stderr)
