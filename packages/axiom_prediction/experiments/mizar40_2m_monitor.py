import datetime
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
from zoneinfo import ZoneInfo

from monitor_support import flush_notifications, monitor_lock, queue_notification, submit_once


state_path = Path('/tmp/mizar40-2m-monitor.json')
ssh = ['ssh', '-F', '/home/debian/.ssh/config', '-o', 'BatchMode=yes', 'hpc']
repo = '/home/msf43/tmp/connections-mizar-weekend-40336e3'
results = '/home/msf43/connections/results/weekend-mizar'
worker = '/home/msf43/connections/packages/axiom_prediction/experiments/mizar40_collect_part.py'
gpu_job_name = 'Mizar40-2m-train'


def remote(command, timeout_s = 180):
    result = subprocess.run(ssh + [command], text = True, capture_output = True, timeout = timeout_s)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return result.stdout


def push(message):
    queue_notification('Mizar two-minute collection', message)


def save(state):
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent = 2) + '\n')
    temporary.replace(state_path)


def load():
    if not state_path.exists():
        return {
            'parts': {str(part): {'attempts': []} for part in range(4)},
            'fatal': [], 'last_batch': 0, 'training': None,
            'collection_finished_notified': False,
            'training_started_notified': False,
            'training_finished_notified': False,
        }
    return json.loads(state_path.read_text())


def job_status(job_id):
    code = f'''import subprocess
job={job_id!r}
p=subprocess.run(["squeue","-h","-j",job,"-o","%T|%M|%l"],text=True,capture_output=True)
if p.stdout.strip(): print(p.stdout.strip().splitlines()[0])
else:
 p=subprocess.run(["sacct","-n","-X","-j",job,"-P","--format=JobIDRaw,State,ExitCode,Elapsed,Timelimit"],text=True,capture_output=True)
 for line in p.stdout.splitlines():
  fields=line.split("|")
  if fields and fields[0] == job:
   print("|".join(fields[1:])); break
'''
    output = remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}').strip()
    if not output:
        return {'state': 'UNKNOWN'}
    fields = output.split('|')
    if len(fields) >= 4:
        return {'state': fields[0].split()[0], 'exit': fields[1],
                'elapsed': fields[2], 'limit': fields[3]}
    return {'state': fields[0].split()[0],
            'elapsed': fields[1] if len(fields) > 1 else '0:00',
            'limit': fields[2] if len(fields) > 2 else '11:59:00',
            'exit': None}


def batch(state, line, operation):
    command = f'cd {shlex.quote(repo)} && {shlex.join(line)}'
    output, _ = submit_once(
        lambda command: (remote(command), ''), command, operation,
        state.get('last_batch', 0),
    )
    return re.search(r'(?m)^(\d+)$', output.strip()).group(1)


def record_intent(state, operation, line, source_job_id = None):
    command = f'cd {shlex.quote(repo)} && {shlex.join(line)}'
    intent = {'operation': operation, 'command': command,
              'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if source_job_id is not None:
        intent['source_job_id'] = source_job_id
    intents = state.setdefault('intents', [])
    if not any(item.get('operation') == operation for item in intents):
        intents.append(intent)
        save(state)
    manifest = f'{results}/collection/mizar40-2m-intent.json'
    code = f'''import json,pathlib
p=pathlib.Path({manifest!r}); d=json.loads(p.read_text()) if p.exists() else {{}}; submissions=d.setdefault("submissions", []); intent={intent!r}; submissions.extend([intent] if not any(item.get("operation") == intent["operation"] for item in submissions) else []); p.write_text(json.dumps(d,indent=2)+"\\n")
'''
    remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}')
    return command


def part_command(part):
    return [
        '/home/msf43/bin/icelake-batch', '--time=11:59:00',
        '--job-name', f'Mizar40-2m-p{part}', '--',
        '/home/msf43/venv/bin/python', worker, 'collect', '--part', str(part),
    ]


def submit_part(state, part, source_job_id = None):
    item = state['parts'][str(part)]
    line = part_command(part)
    operation = f'mizar40-2m:p{part}:{source_job_id or "initial"}'
    command = record_intent(state, operation, line, source_job_id)
    if source_job_id is None:
        push(f'First-time batch command (Mizar40-2m-p{part}):\n{shlex.join(line)}')
    job_id = batch(state, line, operation)
    item['id'] = job_id
    item.setdefault('attempts', []).append(job_id)
    item.pop('intent', None)
    state['last_batch'] = time.time()
    save(state)
    if source_job_id is None:
        push(f'Mizar40 two-minute part {part} submitted: job {job_id}.')


def training_command(resume = False):
    line = [
        '/home/msf43/bin/ampere-batch', '--time=2:59:00',
        '--job-name', gpu_job_name, '--',
        '/home/msf43/venv/bin/python', 'axiom_predictor.py', 'train',
        '--data-dir', f'{results}/Mizar40-2m',
        '--dataset', f'{results}/Mizar40-2m/dataset',
        '--network', 'ExtraLargeFull', '--epochs', '100', '--split', '10',
        '--parts', '0', '1', '2', '3', '4', '5', '6', '7', '--evaluate', '8',
        '--policy', 'SatResetCoP', '--run-name', 'mizar40-2m-train', '--wandb',
    ]
    if resume:
        line.append('--resume')
    return line


def submit_training(state, source_job_id = None):
    if state.get('training') is None:
        state['training'] = {'attempts': []}
    item = state['training']
    line = training_command(source_job_id is not None)
    operation = f'mizar40-2m:training:{source_job_id or "initial"}'
    record_intent(state, operation, line, source_job_id)
    if source_job_id is None:
        push('First-time batch command (Mizar40-2m-train):\n' + shlex.join(line))
    job_id = batch(state, line, operation)
    item['id'] = job_id
    item['attempts'].append(job_id)
    state['last_batch'] = time.time()
    save(state)
    return job_id


def prepare(state):
    prepared_marker = f'{results}/Mizar40-2m/dataset/seed-manifest.json'
    marker_check = f'test -s {shlex.quote(prepared_marker)}'
    try:
        remote(marker_check)
    except RuntimeError:
        remote(f'cd {shlex.quote(repo)} && /home/msf43/venv/bin/python {shlex.quote(worker)} prepare', timeout_s = 3600)
    save(state)
    manifest = f'{results}/collection/mizar40-2m-intent.json'
    manifest_data = {
        'source_timeout_seconds': 900, 'target_timeout_seconds': 120,
        'split': 4, 'parts': [0, 1, 2, 3], 'num_workers_per_job': 76,
        'dataset': f'{results}/Mizar40-2m/dataset',
    }
    code = f'''import json,pathlib
p=pathlib.Path({manifest!r}); d=json.loads(p.read_text()) if p.exists() else {{}}; d.update({manifest_data!r}); p.write_text(json.dumps(d,indent=2)+"\\n")
'''
    remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}')


def valid_part(part):
    path = f'{results}/Mizar40-2m/dataset/Mizar40-2m-part{part}.csv'
    code = f'''import csv,json,pathlib
p=pathlib.Path({path!r})
if not p.is_file(): print("missing"); raise SystemExit
with p.open() as f: first={{k:json.loads(v) for k,v in next(csv.DictReader(f)).items() if v}}
s=first.get("summary",{{}})
print(json.dumps({{"requested":s.get("problems_requested"),"proved":s.get("problems_proved"),"failed":s.get("problems_failed"),"size":p.stat().st_size}}))
'''
    return json.loads(remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}'))


def fail(state, message):
    state.setdefault('fatal', [])
    if message not in state['fatal']:
        state['fatal'].append(message)
        save(state)
        push('Fatal Mizar two-minute run failure: ' + message)


def run(state):
    while True:
        flush_notifications()
        statuses = {}
        for part, item in state['parts'].items():
            if 'id' in item:
                statuses[part] = job_status(item['id'])

        failed = False
        resubmitted = False
        for part, item in state['parts'].items():
            if 'id' not in item:
                continue
            status = statuses[part]
            code = status['state'].split('+')[0]
            if code == 'TIMEOUT':
                submit_part(state, int(part), item['id'])
                resubmitted = True
            elif code in ('FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL'):
                fail(state, f'Mizar40-2m-p{part} ended {code} (job {item["id"]}).')
                failed = True
            elif code == 'COMPLETED':
                if status.get('exit') != '0:0':
                    fail(state, f'Mizar40-2m-p{part} completed with exit {status.get("exit")} (job {item["id"]}).')
                    failed = True
                else:
                    summary = valid_part(int(part))
                    if not summary.get('requested') or summary.get('requested') != summary.get('proved', 0) + summary.get('failed', 0):
                        fail(state, f'Mizar40-2m-p{part} shard audit failed: {summary}')
                        failed = True
                    else:
                        item['complete'] = True
                        item['summary'] = summary
                        save(state)
        if failed:
            return
        if resubmitted:
            continue

        collection_done = all(item.get('complete') for item in state['parts'].values())
        if collection_done and not state.get('collection_finished_notified'):
            requested = sum(item['summary']['requested'] for item in state['parts'].values())
            proved = sum(item['summary']['proved'] for item in state['parts'].values())
            failed_count = sum(item['summary']['failed'] for item in state['parts'].values())
            if requested != 57897 or proved + failed_count != requested:
                fail(state, f'combined shard audit failed: {requested} requested, {proved} proved, {failed_count} failed')
                return
            push(f'Mizar40 two-minute collection finished: {requested} inputs, {proved} proved, {failed_count} failed; four shards in {results}/Mizar40-2m/dataset.')
            state['collection_finished_notified'] = True
            save(state)
        if collection_done and state.get('training') is None:
            job_id = submit_training(state)
            push(f'Mizar40-2m-train submitted after collection: job {job_id}.')

        sleep_s = 7200
        for part, status in statuses.items():
            if status['state'].split('+')[0] == 'RUNNING':
                elapsed = seconds(status.get('elapsed', '0:00'))
                limit = seconds(status.get('limit', '11:59:00'))
                sleep_s = min(sleep_s, max(30, limit - elapsed - 90))
        training = state.get('training')
        if training and 'id' in training:
            status = job_status(training['id'])
            code = status['state'].split('+')[0]
            if code == 'RUNNING' and not state.get('training_started_notified'):
                push(f'Mizar40-2m training started: job {training["id"]}, W&B run mizar40-2m-train.')
                state['training_started_notified'] = True
                save(state)
            elif code == 'TIMEOUT':
                submit_training(state, training['id'])
                continue
            elif code in ('FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL'):
                fail(state, f'Mizar40-2m-train ended {code} (job {training["id"]}).')
                return
            elif code == 'COMPLETED':
                if status.get('exit') != '0:0':
                    fail(state, f'Mizar40-2m-train completed with exit {status.get("exit")} (job {training["id"]}).')
                    return
                if not state.get('training_finished_notified'):
                    push(f'Mizar40-2m training finished successfully: job {training["id"]}, W&B run mizar40-2m-train.')
                    state['training_finished_notified'] = True
                    save(state)
                return
            if code == 'RUNNING':
                elapsed = seconds(status.get('elapsed', '0:00'))
                limit = seconds(status.get('limit', '2:59:00'))
                sleep_s = min(sleep_s, max(30, limit - elapsed - 90))
        state['last_snapshot'] = {'parts': statuses}
        state['last_success_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(state)
        time.sleep(sleep_s)


def seconds(value):
    fields = [int(field) for field in value.split(':')]
    if len(fields) == 3:
        return fields[0] * 3600 + fields[1] * 60 + fields[2]
    if len(fields) == 2:
        return fields[0] * 60 + fields[1]
    return fields[0]


def main():
    state = load()
    if not state.get('prepared'):
        prepare(state)
        state['prepared'] = True
        save(state)
    for part in range(4):
        if 'id' not in state['parts'][str(part)]:
            submit_part(state, part)
    run(state)


if __name__ == '__main__':
    with monitor_lock(state_path):
        try:
            main()
        except Exception as error:
            state = load()
            fail(state, f'{type(error).__name__}: {error}')
