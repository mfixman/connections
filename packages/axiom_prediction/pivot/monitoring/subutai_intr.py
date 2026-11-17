# Runs the Subutai-Prover9 trainer on intr (one resume after a timeout, none after that), then Subutai-Vampire on intr
# 3 h after Prover9's first job succeeds or its second job ends. Job states come from the poller's squeue snapshot.
import json
from pathlib import Path
import re
import sys
import time

here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), '/home/debian/connections/packages/axiom_prediction/experiments']
from monitor_support import submit_once
from slurm_gate import run_slurm
from sweep import push, ssh

state_path = here / 'subutai_intr.json'
log_path = here / 'subutai_intr.log'
snapshot_path = Path.home() / '.local/state/squeue-snapshot.json'
models = '/home/msf43/connections/results/big-train'
gpu2_trainers = {'Subutai-Prover9': '40621976', 'Subutai-Vampire': '40140450'}
vampire_delay_s = 3 * 3600
poll_s = 60

def log(line):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}"
    print(line, flush = True)
    with log_path.open('a') as stream:
        stream.write(line + '\n')

def save(state):
    state_path.write_text(json.dumps(state, indent = 1))

def snapshot():
    return json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {'time': 0, 'jobs': {}}

def submit(state, name):
    attempt = sum(1 for job in state['jobs'] if job['name'] == name) + 1
    command = (
        f"cd /home/msf43/connections && /home/msf43/bin/ampere-batch --intr --gres gpu:1 --cpus 32 --time 0:59:00 "
        f"--name {name} /home/msf43/venv/bin/python /home/msf43/subutai_train.py {name}"
    )
    stdout, _ = submit_once(run_slurm, command, f'subutai-intr-{name}-attempt{attempt}')
    job_id = re.search(r'(?m)^(\d+)$', stdout.strip()).group(1)
    state['jobs'].append({'name': name, 'id': job_id, 'attempt': attempt, 'submitted': time.time(), 'outcome': None})
    log(f'submitted {name} attempt {attempt} on intr: {job_id}')

# Success means epoch 50 exists; a timeout leaves Slurm's "DUE TO TIME LIMIT" line in the job's stderr.
def outcome(job):
    stdout, _ = ssh(
        f"ls {models}/{job['name']}/model/epoch-0050.ts 2>/dev/null; ls {models}/{job['name']}/model 2>/dev/null | grep -c '^epoch-'; "
        f"grep -h 'DUE TO TIME LIMIT' $(ls -dt /home/msf43/tmp/{job['name']}-*/ | head -1)/stderr.err | head -1"
    )
    epochs = re.search(r'(?m)^(\d+)$', stdout)
    detail = f"{epochs.group(1) if epochs else 0} epochs"
    if 'epoch-0050.ts' in stdout:
        return 'success', detail

    return ('timeout' if 'TIME LIMIT' in stdout else 'failed'), detail

def check_ended(state):
    current = snapshot()
    for job in state['jobs']:
        if job['outcome'] is None and job['id'] not in current['jobs'] and current['time'] > job['submitted'] + 30:
            job['outcome'], detail = outcome(job)
            job['ended'] = time.time()
            log(f"{job['name']} attempt {job['attempt']} ({job['id']}) ended: {job['outcome']}, {detail}")
            if job['outcome'] == 'failed':
                push('Subutai intr job failed', f"{job['name']} attempt {job['attempt']} ({job['id']}) ended without reaching epoch 50 and without a time-limit line ({detail}).")
        elif job['outcome'] is None and job['id'] in current['jobs'] and not job.get('started'):
            if current['jobs'][job['id']]['state'] == 'RUNNING':
                job['started'] = time.time()
                log(f"{job['name']} attempt {job['attempt']} ({job['id']}) running")

def next_action(state):
    prover9 = [job for job in state['jobs'] if job['name'] == 'Subutai-Prover9']
    vampire = [job for job in state['jobs'] if job['name'] == 'Subutai-Vampire']
    if not prover9:
        return 'submit-prover9'

    if len(prover9) == 1 and prover9[0]['outcome'] == 'timeout':
        return 'submit-prover9'

    trigger = None
    if prover9[0]['outcome'] == 'success':
        trigger = prover9[0]['ended']
    elif len(prover9) == 2 and prover9[1]['outcome'] is not None:
        trigger = prover9[1]['ended']

    if trigger and not vampire and time.time() >= trigger + vampire_delay_s:
        return 'submit-vampire'

    if vampire and vampire[0]['outcome'] is not None:
        return 'finished'

    return None

def main():
    state = json.loads(state_path.read_text()) if state_path.exists() else {'jobs': [], 'held': False}
    if not state['held']:
        stdout, stderr = run_slurm(f"scontrol hold {' '.join(gpu2_trainers.values())}")
        state['held'] = True
        save(state)
        log(f"held the gpu2 trainers {gpu2_trainers} so they cannot train alongside the intr jobs: {(stdout + stderr).strip()[:120]}")

    while True:
        try:
            check_ended(state)
            action = next_action(state)
            if action == 'submit-prover9':
                submit(state, 'Subutai-Prover9')
            elif action == 'submit-vampire':
                submit(state, 'Subutai-Vampire')
            elif action == 'finished':
                log('Subutai-Vampire intr job ended; nothing further is scheduled')
                save(state)
                return
        except Exception as error:
            log(f'error: {str(error)[:300]}')

        save(state)
        time.sleep(poll_s)

if __name__ == '__main__':
    main()
