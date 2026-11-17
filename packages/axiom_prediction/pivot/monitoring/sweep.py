# Drives the epoch sweep: submits the stage jobs in order (model, epoch, job), checks each job about a minute after it
# starts and every hour, resubmits a job once if it timed out or died (the runner resumes), and sends the requested notices.
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import urllib.request

sys.path.insert(0, '/home/debian/connections/packages/axiom_prediction/experiments')
from monitor_support import submit_once

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slurm_gate import run_slurm

snapshot_path = Path.home() / '.local/state/squeue-snapshot.json'

run = '/rds/user/msf43/hpc-work/epoch-stages-20261111'
here = Path(__file__).resolve().parent
state_path = here / 'state.json'
log_path = here / 'sweep.log'
models = ['Sharbaraz5', 'Sharbaraz15']
first_epochs = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]
kinds = ['Filters', 'Support']
expected = {'Filters': {'prover9': 608, 'iprover': 608, 'vampire': 1426}, 'Support': {'prover9': 608, 'vampire': 1426}}
terminal = ('COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE')
max_attempts = 5
stall_min = 20
stall_check_s = 600
error_rows_flag = 12
poll_s = 60
start_check_s = 60
hourly_s = 3600

def log(line):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}"
    print(line, flush = True)
    with log_path.open('a') as stream:
        stream.write(line + '\n')

def push(title, body):
    log(f'PUSH {title}: {body}')
    token = (Path.home() / 'secrets/pushbullet_token').read_text().strip()
    payload = json.dumps({'type': 'note', 'title': title, 'body': body}).encode()
    headers = {'Access-Token': token, 'Content-Type': 'application/json'}
    request = urllib.request.Request('https://api.pushbullet.com/v2/pushes', data = payload, headers = headers)
    with urllib.request.urlopen(request, timeout = 30) as response:
        response.read()

def ssh(command, timeout = 420):
    arguments = ['ssh', '-F', str(Path.home() / '.ssh/config'), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=60', 'hpc', command]
    done = subprocess.run(arguments, capture_output = True, text = True, timeout = timeout)
    if done.returncode == 255:
        raise ConnectionError(done.stderr[-200:])

    return done.stdout, done.stderr

def plan(epochs):
    return [
        {'name': f'{model}-e{epoch}-{kind}-p8', 'model': model, 'epoch': epoch, 'kind': kind}
        for model in models
        for epoch in epochs
        for kind in kinds
    ]

def command(job):
    script = f"bash {run}/jobs/stages.sh {job['model']} {job['epoch']} {job['kind'].lower()}"
    return f"cd /home/msf43/connections && /home/msf43/bin/icelake-batch --cpus 16 --time 2:00:00 --name {{name}} {script}"

def load_state():
    if state_path.exists():
        return json.loads(state_path.read_text())

    # User, 2026-11-11 19:30 UTC: phase 2 runs only these epochs, to save CPU quota.
    rest = [25, 35, 40, 62, 92]
    jobs = [{**job, 'phase': 1} for job in plan(first_epochs)] + [{**job, 'phase': 2} for job in plan(rest)]
    for job in jobs:
        job.update(ids = [], state = 'Waiting', start = None, start_checked = False, done = False)

    return {'jobs': jobs, 'started_order': [], 'notified': [], 'ssh_failures': 0, 'last_hourly': 0}

def save(state):
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent = 1))
    temporary.replace(state_path)

def submit(job):
    attempt = len(job['ids']) + 1
    operation = f"epoch-sweep-{job['name']}-attempt{attempt}"
    text = command(job).replace('{name}', job['name'])
    try:
        stdout, _ = submit_once(run_slurm, text, operation)
    except RuntimeError as error:
        if 'Uncertain' not in str(error):
            raise

        # Reconciled from the next squeue snapshot rather than an extra Slurm command.
        log(f"{job['name']}: {error}; reconciling from the next snapshot")
        taken = time.time()
        while snapshot()['time'] < taken + 30:
            time.sleep(30)

        known = set(job['ids'])
        found = [i for i, j in snapshot()['jobs'].items() if j['name'] == job['name'] and i not in known]
        if not found:
            receipt = Path.home() / '.local/state/hpc-submissions'
            for path in receipt.glob('*.json'):
                if path.name != 'last-batch.json' and json.loads(path.read_text()).get('operation') == operation:
                    path.unlink()

            log(f"{job['name']}: no job found; receipt cleared, will retry")
            return

        stdout = found[-1]

    job_id = re.search(r'(?m)^(\d+)$', stdout.strip()).group(1)
    job['ids'].append(job_id)
    job.update(state = 'Submitted', start = None, start_checked = False, submitted_at = time.time())
    log(f"submitted {job['name']} attempt {attempt}: {job_id}")

def snapshot():
    return json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {'time': 0, 'jobs': {}}

# Job states from the poller's squeue snapshot; a job missing from a snapshot taken after its submission has ended.
def scheduler(state):
    current = snapshot()
    out = {}
    for job in state['jobs']:
        if not job['ids'] or job['done']:
            continue

        job_id = job['ids'][-1]
        if job_id in current['jobs']:
            entry = current['jobs'][job_id]
            out[job_id] = {'state': entry['state'], 'start': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(current['time'])), 'elapsed': entry['elapsed']}
        elif current['time'] > job.get('submitted_at', 0) + 30:
            out[job_id] = {'state': 'ENDED', 'start': '', 'elapsed': ''}

    return out

def job_dir(job):
    return f"$(ls -dt /home/msf43/tmp/{job['name']}-*/ 2>/dev/null | head -1)"

def start_check(job):
    stdout, _ = ssh(f"d={job_dir(job)}; tail -c 3000 $d/stdout.out; echo ===STDERR; grep -c 'Traceback\\|Error' $d/stderr.err")
    body, _, errors = stdout.partition('===STDERR')
    problems = []
    if 'FATAL' in body:
        problems.append(re.search(r'FATAL[^\n]*', body).group(0))
    if '--prover' not in body:
        problems.append('no runner started')
    if errors.strip() not in ('', '0'):
        problems.append(f'{errors.strip()} Traceback/Error lines in stderr')

    return problems

def outputs_complete(job):
    folder = f"{run}/results/{job['model']}/e{job['epoch']}"
    names = ' '.join(f"{folder}/{prover}-{job['kind'].lower()}.csv" for prover in expected[job['kind']])
    # Error rows are counted from the CSV's error column, without the deliberate SkippedOST114 rows; every run has about 6.
    count_errors = 'import csv, sys; print(sum(r.get(\\"error\\", \\"\\") not in (\\"\\", \\"SkippedOST114\\") for r in csv.DictReader(open(sys.argv[1]))))'
    stdout, _ = ssh(
        f"for f in {names}; do echo $f $(($(wc -l < $f 2>/dev/null || echo 1) - 1)) $(python3 -c \"{count_errors}\" $f 2>/dev/null || echo 0); done; "
        f"d={job_dir(job)}; grep -c 'FAILED:' $d/stdout.out"
    )
    lines = stdout.strip().splitlines()
    counts = {Path(line.split()[0]).name.split('-')[0]: int(line.split()[1]) for line in lines[:-1]}
    errors = {Path(line.split()[0]).name.split('-')[0]: int(line.split()[2]) for line in lines[:-1]}
    missing = {p: n - counts.get(p, 0) for p, n in expected[job['kind']].items() if counts.get(p, 0) < n}
    failed = int(lines[-1]) if lines and lines[-1].isdigit() else 0
    if any(n > error_rows_flag for n in errors.values()):
        log(f"HIGH-ERRORS {job['name']}: error rows {errors}")

    return not missing and not failed, f'rows {counts}, error rows {errors}, failed runs {failed}'

def note_start(state, job):
    if job['name'] in state['started_order']:
        return

    state['started_order'].append(job['name'])
    number = len(state['started_order'])
    if number == 1 and 'first-start' not in state['notified']:
        state['notified'].append('first-start')
        push('Epoch sweep: first job running', f"{job['name']} started at {job['start']} UTC")
    if number == 10 and 'tenth-start' not in state['notified']:
        state['notified'].append('tenth-start')
        state['tenth'] = job['name']
        push('Epoch sweep: 10th job running', f"{job['name']} (10th to start) started at {job['start']} UTC")

def fatal(state, key, body):
    if key not in state['notified']:
        state['notified'].append(key)
        push('Epoch sweep fatal error', body)

def finished(state, job, status):
    complete, detail = outputs_complete(job)
    # A job cancelled as stalled after writing everything only failed to exit (stuck processes); its output stands.
    if complete and (status in ('COMPLETED', 'ENDED') or job.get('stalled')):
        job.update(done = True, state = 'Done')
        log(f"{job['name']} done ({detail})")
        if state.get('tenth') == job['name'] and 'tenth-done' not in state['notified']:
            state['notified'].append('tenth-done')
            push('Epoch sweep: 10th job finished', f"{job['name']} completed successfully ({detail})")
        return

    if status == 'CANCELLED' and not job.get('stalled'):
        job['state'] = 'Cancelled'
        log(f"{job['name']} cancelled ({detail}); not resubmitted")
        return

    if len(job['ids']) >= max_attempts:
        job['state'] = 'Failed'
        fatal(state, f"failed-{job['name']}", f"{job['name']} ended {status} on attempt {len(job['ids'])} ({detail}); not resubmitted")
        return

    job['state'] = 'Resubmit'
    job['stalled'] = False
    log(f"{job['name']} ended {status} ({detail}); resubmitting, the runner resumes")

# The runner logs progress every minute; a quiet log means its processes are stuck (Lustre trouble on 2026-11-11).
def cancel_stalled(state):
    running = [job for job in state['jobs'] if job['state'] == 'Running' and not job.get('stalled')]
    if not running:
        return

    checks = '; '.join(f"echo {job['name']} $(( ($(date +%s) - $(stat -c %Y {job_dir(job)}/stderr.err)) / 60 ))" for job in running)
    stdout, _ = ssh(checks)
    quiet = dict(line.split() for line in stdout.splitlines() if len(line.split()) == 2)
    for job in running:
        minutes = int(quiet.get(job['name'], 0))
        if minutes >= stall_min:
            run_slurm(f"scancel {job['ids'][-1]}")
            job['stalled'] = True
            log(f"STALLED {job['name']}: log quiet for {minutes} min; cancelled {job['ids'][-1]} to resubmit")

def step(state):
    info = scheduler(state)
    for job in state['jobs']:
        if job['done'] or not job['ids'] or job['state'] in ('Failed', 'Cancelled', 'Resubmit'):
            continue

        current = info.get(job['ids'][-1])
        if current is None:
            continue

        status = current['state']
        if status == 'RUNNING' and job['start'] is None:
            job.update(start = current['start'], seen_running = time.time(), state = 'Running')
            note_start(state, job)
            log(f"{job['name']} running since {current['start']}")

        if job['state'] == 'Running' and not job['start_checked'] and time.time() - job['seen_running'] >= start_check_s:
            problems = start_check(job)
            job['start_checked'] = True
            log(f"{job['name']} start check: {'; '.join(problems) or 'ok'}")
            if any('FATAL' in p for p in problems):
                fatal(state, f"start-{job['name']}", f"{job['name']}: {'; '.join(problems)}")

        if status in terminal or status == 'ENDED':
            finished(state, job, status)

    phase_one_done = all(job['done'] for job in state['jobs'] if job['phase'] == 1)
    # No submissions while Slurm does not answer: a lost sbatch reply could not be reconciled and would duplicate a job.
    if not slurm_answers():
        return

    for job in state['jobs']:
        if job['state'] == 'Resubmit' or (job['state'] == 'Waiting' and (job['phase'] == 1 or phase_one_done)):
            submit(job)
            return

# Submissions wait while the poller has no fresh snapshot (Slurm not answering).
def slurm_answers():
    age_s = time.time() - snapshot()['time']
    if age_s > 1200:
        log(f"holding submissions: last squeue snapshot is {age_s / 60:.0f} min old")
        return False

    return True

def hourly(state):
    counts = {}
    for job in state['jobs']:
        counts[job['state']] = counts.get(job['state'], 0) + 1

    running = sum(1 for j in snapshot()['jobs'].values() if j['state'] == 'RUNNING' and j['name'].endswith('-p8'))
    stdout, _ = ssh("quota -s 2>/dev/null | tail -1 | awk '{print $2, $3, $6, $7}'")
    log(f'HOURLY jobs {counts}; running -p8 jobs {running}; quota: {stdout.split()}')
    for job in state['jobs']:
        if job['state'] == 'Running':
            problems = start_check(job)
            if problems:
                log(f"HOURLY {job['name']}: {'; '.join(problems)}")

def main():
    state = load_state()
    save(state)
    log(f"sweep driver started: {sum(j['done'] for j in state['jobs'])} of {len(state['jobs'])} jobs done")
    while not all(job['done'] or job['state'] in ('Failed', 'Cancelled') for job in state['jobs']):
        try:
            step(state)
            if time.time() - state.get('last_stall_check', 0) >= stall_check_s:
                cancel_stalled(state)
                state['last_stall_check'] = time.time()

            if time.time() - state['last_hourly'] >= hourly_s:
                hourly(state)
                state['last_hourly'] = time.time()

            state['ssh_failures'] = 0
        except (ConnectionError, subprocess.TimeoutExpired) as error:
            state['ssh_failures'] += 1
            log(f"ssh failure {state['ssh_failures']}: {str(error)[:120]}")
            if state['ssh_failures'] == 5:
                try:
                    ssh('true', 300)
                except Exception:
                    push('HPC authenticator token needed', 'The epoch sweep driver cannot reach the HPC; the SSH master may need a new token.')
        except Exception:
            log('error: ' + traceback.format_exc()[-600:])

        save(state)
        time.sleep(poll_s)

    log('all sweep jobs finished')

if __name__ == '__main__':
    main()
