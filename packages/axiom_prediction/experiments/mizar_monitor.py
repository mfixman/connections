import datetime
import json
import pathlib
import re
import shlex
import subprocess
import time
import urllib.request
from zoneinfo import ZoneInfo
from monitor_support import flush_notifications, queue_notification, monitor_lock, select_checkpoint, submit_once, valid_completion


state_path = pathlib.Path('/tmp/mizar-weekend-monitor.json')
ssh = ['ssh', '-F', '/home/debian/.ssh/config', '-o', 'BatchMode=yes', 'hpc']
repo = '/home/msf43/tmp/connections-mizar-weekend-40336e3'
training_revision = 'db9f2bd'
results = '/home/msf43/connections/results/weekend-mizar'
collection_script = f'{results}/collection/collect_mizar40.py'
collector_id = '37936505'
training_jobs = {
    'Mizar40': {
        'id': '37945095', 'job_name': 'Mizar40-train', 'run_name': 'mizar40-train',
        'data_dir': f'{results}/Mizar40', 'dataset': f'{results}/Mizar40/dataset',
    },
    'Mizar40-TPTP': {
        'id': '37945358', 'job_name': 'Mizar40-TPTP-train', 'run_name': 'mizar40-tptp-train',
        'data_dir': f'{results}/Mizar40-TPTP', 'dataset': f'{results}/Mizar40-TPTP/dataset',
    },
}


def remote(command):
    result = subprocess.run(ssh + [command], text=True, capture_output=True, timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return result.stdout


def batch(state, line, operation):
    command = f'cd {shlex.quote(repo)} && {shlex.join(line)}'
    output, _ = submit_once(lambda command: (remote(command), ''), command, operation,
                            state.get('last_batch', 0))
    return re.search(r'(?m)^(\d+)$', output.strip()).group(1)


def push(message):
    queue_notification('Mizar weekend run', message)


def load_state():
    if state_path.exists():
        state = json.loads(state_path.read_text())
        state.setdefault('status_reports', [])
        state.setdefault('monitor_errors', 0)
        state.setdefault('auth_notified', False)
        return state
    state = {
        'collector_id': collector_id,
        'trainings': training_jobs,
        'collection_complete_notified': False,
        'training_started_notified': [],
        'training_finished_notified': [],
        'early_training_cancellations': [],
        'fatal': [],
        'last_batch': 0,
        'status_reports': [],
        'monitor_errors': 0,
        'auth_notified': False,
    }
    save(state)
    return state


def save(state):
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(state_path)


def job_status(job_id):
    code = f'''import subprocess
job={job_id!r}
p=subprocess.run(["squeue","-h","-j",job,"-o","%T|%M|%l"],text=True,capture_output=True)
if p.stdout.strip():
 print(p.stdout.strip().splitlines()[0])
else:
 p=subprocess.run(["sacct","-n","-X","-j",job,"-P","--format=JobIDRaw,State,ExitCode,Elapsed,Timelimit"],text=True,capture_output=True)
 for line in p.stdout.splitlines():
  fields=line.split("|")
  if fields and fields[0] == job:
   print("|".join(fields[1:]))
   break
'''
    output = remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}').strip()
    if not output:
        return {'state': 'UNKNOWN'}
    fields = output.split('|')
    if len(fields) >= 4:
        return {
            'state': fields[0].split()[0], 'exit': fields[1],
            'elapsed': fields[2], 'limit': fields[3],
        }
    return {
        'state': fields[0].split()[0],
        'elapsed': fields[1] if len(fields) > 1 else '0:00',
        'limit': fields[2] if len(fields) > 2 else '2:59:00',
        'exit': None,
    }


def active_jobs(name):
    command = f'squeue -u msf43 -h -n {shlex.quote(name)} -o "%i|%T"'
    output = remote(command)
    return [line.split('|', 1) for line in output.splitlines() if line.strip()]


def wait_for_batch_slot(state):
    remaining = 60 - (time.time() - state.get('last_batch', 0))
    if remaining > 0:
        time.sleep(remaining)


def submit_collection_resume(state):
    matches = active_jobs('Mizar40-collect')
    if matches:
        if len(matches) > 1:
            raise RuntimeError(f'ambiguous active Mizar40 collection jobs: {matches}')
        state['collector_id'] = matches[0][0]
        state.setdefault('collector_attempts', []).append(matches[0][0])
        state.pop('collector_intent', None)
        save(state)
        return
    wait_for_batch_slot(state)
    line = [
        '/home/msf43/bin/icelake-batch', '--time=11:59:00', '--job-name=Mizar40-collect',
        '--', '/home/msf43/venv/bin/python', collection_script,
    ]
    remote_line = f'cd {shlex.quote(repo)} && {shlex.join(line)}'
    state['collector_intent'] = {
        'source_job_id': state['collector_id'], 'command': remote_line,
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    state['last_batch'] = time.time()
    save(state)
    manifest = f'{results}/collection/intent.json'
    code = f'''import json,pathlib
p=pathlib.Path({manifest!r}); d=json.loads(p.read_text()); d.setdefault("resume_intents",[]).append({{"command":{shlex.join(line)!r},"recorded_utc":{datetime.datetime.now(datetime.timezone.utc).isoformat()!r}}}); p.write_text(json.dumps(d,indent=2)+"\\n")
'''
    remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}')
    try:
        job_id = batch(state, line, f"mizar:collection:{state['collector_id']}")
    except Exception:
        matches = active_jobs('Mizar40-collect')
        if not matches:
            raise
        if len(matches) > 1:
            raise RuntimeError(f'ambiguous active Mizar40 collection jobs after uncertain submit: {matches}')
        job_id = matches[0][0]
    state['collector_id'] = job_id
    state['last_batch'] = time.time()
    state.setdefault('collector_attempts', []).append(job_id)
    state.pop('collector_intent', None)
    save(state)


def submit_training_resume(state, key, item):
    matches = active_jobs(item['job_name'])
    if matches:
        if len(matches) > 1:
            raise RuntimeError(f'ambiguous active training jobs for {item["job_name"]}: {matches}')
        item.setdefault('attempts', [item['id']])
        item['id'] = matches[0][0]
        item['attempts'].append(item['id'])
        item.pop('resume_intent', None)
        save(state)
        return
    wait_for_batch_slot(state)
    line = [
        '/home/msf43/bin/ampere-batch', '--time=2:59:00', '--job-name', item['job_name'], '--',
        '/home/msf43/venv/bin/python', 'axiom_predictor.py', 'train',
        '--data-dir', item['data_dir'], '--dataset', item['dataset'],
        '--network', 'ExtraLargeFull', '--epochs', '100', '--split', '10',
        '--parts', '0', '1', '2', '3', '4', '5', '6', '7', '--evaluate', '8',
        '--policy', 'SatResetCoP', '--run-name', item['run_name'], '--wandb', '--resume',
    ]
    remote_line = f'cd {shlex.quote(repo)} && {shlex.join(line)}'
    item['resume_intent'] = {
        'source_job_id': item['id'], 'command': remote_line,
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    state['last_batch'] = time.time()
    save(state)
    try:
        job_id = batch(state, line, f"mizar:training:{key}:{item['id']}")
    except Exception:
        matches = active_jobs(item['job_name'])
        if not matches:
            raise
        if len(matches) > 1:
            raise RuntimeError(f'ambiguous active training jobs after uncertain submit: {matches}')
        job_id = matches[0][0]
    item.setdefault('attempts', [item['id']])
    item['id'] = job_id
    state['last_batch'] = time.time()
    item.setdefault('attempts', []).append(item['id'])
    item.pop('resume_intent', None)
    save(state)


def model_checkpoint_exists(item):
    path = f"{item['data_dir']}/model/model.pt"
    return remote(f'test -f {shlex.quote(path)} && echo yes || echo no').strip() == 'yes'


def submit_training_fresh(state, item):
    wait_for_batch_slot(state)
    line = [
        '/home/msf43/bin/ampere-batch', '--time=2:59:00', '--job-name', item['job_name'], '--',
        '/home/msf43/venv/bin/python', 'axiom_predictor.py', 'train',
        '--data-dir', item['data_dir'], '--dataset', item['dataset'],
        '--network', 'ExtraLargeFull', '--epochs', '100', '--split', '10',
        '--parts', '0', '1', '2', '3', '4', '5', '6', '7', '--evaluate', '8',
        '--policy', 'SatResetCoP', '--run-name', item['run_name'], '--wandb',
    ]
    item.setdefault('attempts', [item['id']])
    item['id'] = batch(state, line, f"mizar:fresh:{item['job_name']}:{item['id']}")
    item.setdefault('attempts', []).append(item['id'])
    state['last_batch'] = time.time()
    save(state)


def collection_shard_ready():
    path = f'{results}/Mizar40-15m/dataset/Mizar40.jsonl'
    fallback = f'{results}/Mizar40/dataset/Mizar40.jsonl'
    code = f'''import json,pathlib
p=pathlib.Path({path!r})
if not p.is_file(): p=pathlib.Path({fallback!r})
if p.is_file():
 with p.open() as f: row=json.loads(next(f))
 print(json.dumps({{"exists":True,"summary":row.get("summary",{{}}),"size":p.stat().st_size}}))
else: print(json.dumps({{"exists":False}}))
'''
    return json.loads(remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}'))


def stage_combined_dataset():
    tptp_dataset = '/home/msf43/learncop/artifacts/axiom-dataset/dataset'
    mizar_shard = f'{results}/Mizar40-15m/dataset/Mizar40.jsonl'
    combined_dataset = f'{results}/Mizar40-TPTP-15m/dataset'
    code = f'''import json,pathlib
source=pathlib.Path({tptp_dataset!r})
mizar=pathlib.Path({mizar_shard!r})
target=pathlib.Path({combined_dataset!r})
sources=sorted(source.glob("*.jsonl"))
if not mizar.is_file(): raise RuntimeError(f"missing completed Mizar shard: {{mizar}}")
if not sources: raise RuntimeError(f"no TPTP shards found in {{source}}")
sources.append(mizar)
if len({{p.name for p in sources}}) != len(sources): raise RuntimeError("colliding combined shard names")
settings=[]
for shard in sources:
 with shard.open() as f: header=json.loads(next(f))
 if header.get("schema") != "learncop.axiom_prediction.dataset-shard.v2": raise RuntimeError(f"invalid dataset shard: {{shard}}")
 collection=header.get("collection",{{}})
 if collection.get("sat_policy") != "satresetcop": raise RuntimeError(f"unexpected policy in {{shard}}: {{collection.get('sat_policy')}}")
 settings.append(collection)
target.mkdir(parents=True,exist_ok=True)
for shard in sources:
 destination=target/shard.name
 resolved=shard.resolve()
 if destination.is_symlink() and destination.resolve()==resolved: continue
 if destination.exists() or destination.is_symlink(): raise RuntimeError(f"refusing to overwrite existing combined-dataset file: {{destination}}")
 destination.symlink_to(resolved)
print(json.dumps({{"shards":len(sources),"mizar":str(mizar),"target":str(target),"settings":settings}}))
'''
    return json.loads(remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}'))


def update_training_checkout():
    command = (
        f'git -C {shlex.quote(repo)} diff --quiet && '
        f'git -C {shlex.quote(repo)} diff --cached --quiet && '
        f'git -C {shlex.quote(repo)} checkout --detach {shlex.quote(training_revision)} && '
        f'git -C {shlex.quote(repo)} rev-parse --short HEAD'
    )
    revision = remote(command).strip()
    if revision != training_revision:
        raise RuntimeError(f'training checkout resolved to {revision}, expected {training_revision}')


def seconds(value):
    if '-' in value:
        days, value = value.split('-', 1)
        prefix = int(days) * 86400
    else:
        prefix = 0
    fields = [int(field) for field in value.split(':')]
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours, minutes, seconds = 0, *fields
    else:
        return prefix + fields[0]
    return prefix + hours * 3600 + minutes * 60 + seconds


def collection_progress():
    cache = f'{results}/Mizar40/dataset/.cache/Mizar40'
    code = f'''import pathlib
p=pathlib.Path({cache!r})
def count(kind):
 names=set()
 for directory in (p/kind, p/"partial"/kind):
  if directory.is_dir(): names.update(record.name for record in directory.glob("*.json"))
 return len(names)
examples=count("examples")
failures=count("failures")
print(f"{{examples + failures}}/57897 recorded ({{examples}} proved, {{failures}} failed)")
'''
    return remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}').strip()


def two_minute_collection_progress():
    cache = f'{results}/Mizar40-2m/dataset-2m/.cache'
    fallback = f'{results}/Mizar40-2m/dataset/.cache'
    code = f'''import pathlib
root=pathlib.Path({cache!r})
if not root.exists(): root=pathlib.Path({fallback!r})
proved=failed=0
for part in range(4):
 p=root/f"Mizar40-2m-part{{part}}"
 for kind in ("examples","failures"):
  names=set()
  for directory in (p/kind,p/"partial"/kind):
   if directory.is_dir(): names.update(record.name for record in directory.glob("*.json"))
  if kind == "examples": proved+=len(names)
  else: failed+=len(names)
print(f"{{proved+failed}}/57897 recorded ({{proved}} proved, {{failed}} failed)")
'''
    output = remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}').strip()
    state_path_2m = pathlib.Path('/tmp/mizar40-2m-monitor.json')
    if state_path_2m.exists():
        state = json.loads(state_path_2m.read_text())
        statuses = state.get('last_snapshot', {}).get('parts', {})
        jobs = ', '.join(
            f'p{part}={statuses.get(part, {}).get("state", "PENDING")}'
            for part in map(str, range(4))
        )
        return f'{output}; jobs {jobs}'
    return output


def prepare_training(state):
    # Keep pending jobs eligible; preserve the completed 15-minute shard separately.
    mizar_shard = f'{results}/Mizar40/dataset/Mizar40.jsonl'
    mizar_15m_dataset = f'{results}/Mizar40-15m/dataset'
    low_priority_dataset = f'{results}/Mizar40-2m/dataset'
    code = f'''import json,pathlib,os
shard=pathlib.Path({mizar_shard!r})
dataset=pathlib.Path({mizar_15m_dataset!r})
source=pathlib.Path({low_priority_dataset!r})
archive=source.with_name("dataset-2m")
dataset.mkdir(parents=True,exist_ok=True)
target=dataset/shard.name
if shard.is_file():
 with shard.open() as f: header=json.loads(next(f))
 if header.get("schema") != "learncop.axiom_prediction.dataset-shard.v2": raise RuntimeError(f"invalid Mizar 15-minute shard: {{shard}}")
 if target.exists() and target.resolve()!=shard.resolve(): raise RuntimeError(f"refusing to overwrite {{target}}")
 if not target.exists(): os.replace(shard,target)
elif not target.is_file(): raise RuntimeError(f"missing completed Mizar shard: {{shard}}")
if source.is_symlink():
 if source.resolve()!=dataset.resolve(): raise RuntimeError(f"unexpected low-priority dataset target: {{source.resolve()}}")
elif source.exists():
 if archive.exists(): raise RuntimeError(f"refusing to overwrite 2-minute archive: {{archive}}")
 source.rename(archive)
 source.symlink_to(dataset, target_is_directory=True)
elif not source.exists(): source.symlink_to(dataset, target_is_directory=True)
print(json.dumps({{"mizar_15m":str(target),"low_priority_dataset":str(source.resolve()),"two_minute_archive":str(archive)}}))
'''
    state['mizar_dataset_mapping'] = json.loads(remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}'))
    state['combined_dataset'] = stage_combined_dataset()
    state['datasets_prepared'] = True
    save(state)
    return True


def release_preparation_holds(state):
    return


def training_checkpoints(item):
    ids = list(dict.fromkeys([*item.get('attempts', []), item['id']]))
    code = rf'''import json,pathlib,subprocess,re
folder=pathlib.Path({(item['data_dir'] + '/model')!r})
p=subprocess.run(['sacct','-X','-n','-P','-j',{','.join(ids)!r},'--format=JobIDRaw,StdOut%2000,StdErr%2000'],capture_output=True,text=True,check=True)
scores={{}}; structured={{}}
for line in p.stdout.splitlines():
 fields=line.split('|')
 if fields[0] not in {ids!r}: continue
 for filename in fields[1:3]:
  path=pathlib.Path(filename)
  if not path.is_file(): continue
  for line in path.read_text(errors='replace').splitlines():
   match=re.search(r'epoch (\d+): held-out .*?roc_auc ([0-9.]+)',line)
   if match: scores[int(match[1])]=float(match[2])
   try: row=json.loads(line)
   except ValueError: continue
   if isinstance(row,dict) and row.get('event')=='evaluation' and row.get('roc_auc') is not None:
    structured[int(row['epoch'])]=float(row['roc_auc'])
scores.update(structured)
print(json.dumps({{'available':[int(p.stem.split('-')[-1]) for p in folder.glob('epoch-*.pt') if p.stat().st_size], 'scores':scores}}))
'''
    return json.loads(remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}'))


def submit_evaluation(state, key, item):
    evaluation = state.setdefault('evaluations', {}).get(key)
    if evaluation is None:
        info = training_checkpoints(item)
        epoch = select_checkpoint(info, 100)
        if 100 not in info['available'] or epoch is None:
            raise RuntimeError(f'{key} needs epoch 100 and validation ROC AUC before testing')
        evaluation = {
            'name': f'{key}-test', 'checkpoint_epoch': epoch,
            'checkpoint': f"{item['data_dir']}/model/epoch-{epoch:04d}.pt",
            'output': f"{item['data_dir']}/test.csv", 'attempts': [],
        }
        state['evaluations'][key] = evaluation
        save(state)
    line = [
        '/home/msf43/bin/icelake-batch', '--time=3:59:00', '--job-name', evaluation['name'], '--',
        '/home/msf43/venv/bin/python', 'axiom_predictor.py', 'evaluate',
        '--model', evaluation['checkpoint'], '--dataset', item['dataset'],
        '--split', '10', '--parts', '9', '--policy', 'SatResetCoP', '--cpu',
        '--csv', '--output', evaluation['output'], '--wandb', '--run-name', evaluation['name'],
    ]
    if not evaluation.get('first_command_notified'):
        push('First-time batch command (' + evaluation['name'] + '):\n' + shlex.join(line))
        evaluation['first_command_notified'] = True
        save(state)
    previous = evaluation.get('id', 'initial')
    job_id = batch(state, line, f'mizar:evaluation:{key}:{previous}')
    evaluation['id'] = job_id
    evaluation['attempts'].append(job_id)
    state['last_batch'] = time.time()
    save(state)


def evaluation_result(item):
    code = f'''import json,pathlib
p=pathlib.Path({item['output']!r})
marker=pathlib.Path(str(p)+'.resume/completion.json')
print(json.dumps({{'output_size':p.stat().st_size if p.exists() else None,'completion':json.loads(marker.read_text()) if marker.exists() else None}}))
'''
    return json.loads(remote(f'/home/msf43/venv/bin/python -c {shlex.quote(code)}'))


def monitor_evaluations(state, training_statuses):
    statuses = []
    for key, item in state['trainings'].items():
        status = training_statuses[key]
        if status['state'] != 'COMPLETED' or status.get('exit') != '0:0':
            continue
        evaluation = state.setdefault('evaluations', {}).get(key)
        if evaluation is None or 'id' not in evaluation:
            submit_evaluation(state, key, item)
            continue
        if evaluation.get('complete'):
            continue
        status = job_status(evaluation['id'])
        statuses.append(status)
        if status['state'] == 'TIMEOUT':
            submit_evaluation(state, key, item)
        elif status['state'] == 'COMPLETED':
            if status.get('exit') != '0:0' or not valid_completion(evaluation_result(evaluation)):
                raise RuntimeError(f'{key} test evaluation ended without valid completed output')
            evaluation['complete'] = True
            save(state)
        elif status['state'] in ('FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL'):
            raise RuntimeError(f"{key} test evaluation ended {status['state']}: {evaluation['id']}")
    return statuses


def run(state):
    while True:
        flush_notifications()
        collector = job_status(state['collector_id'])
        training_statuses = {
            key: job_status(item['id']) for key, item in state['trainings'].items()
        }
        state['last_snapshot'] = {'collector': collector, 'training': training_statuses}
        collector_state = collector['state'].split('+')[0]

        if collector_state == 'TIMEOUT':
            submit_collection_resume(state)
            continue
        if collector_state in ('FAILED', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL'):
            message = f"Mizar40 collection ended {collector_state} (job {state['collector_id']}); stopping automatic work."
            if message not in state['fatal']:
                state['fatal'].append(message)
                save(state)
                push('Fatal Mizar run failure: ' + message)
            return

        collection_done = collector_state == 'COMPLETED'
        if collection_done:
            if collector.get('exit') != '0:0':
                raise RuntimeError('Collection completion has no successful scheduler exit code')
            shard = collection_shard_ready()
            summary = shard.get('summary', {})
            if (not shard.get('exists') or summary.get('problems_requested') != 57897
                    or summary.get('problems_proved', 0) == 0
                    or summary.get('problems_proved', 0) + summary.get('problems_failed', 0) != 57897):
                message = f"Collection job ended successfully but Mizar shard audit failed: {shard}"
                if message not in state['fatal']:
                    state['fatal'].append(message)
                    save(state)
                    push('Fatal Mizar run failure: ' + message)
                return
            try:
                if not state.get('datasets_prepared'):
                    if not prepare_training(state):
                        time.sleep(30)
                        continue
                release_preparation_holds(state)
                combined = state['combined_dataset']
            except Exception as error:
                message = f"Mizar+TPTP dataset staging failed: {error}"
                if message not in state['fatal']:
                    state['fatal'].append(message)
                    save(state)
                    push('Fatal Mizar run failure: ' + message)
                return
            if not state['collection_complete_notified']:
                push(f"Mizar40 collection finished: {summary.get('problems_requested')} inputs, {summary.get('problems_proved')} proved, {summary.get('problems_failed')} failed; shard {results}/Mizar40/dataset/Mizar40.jsonl; combined dataset staged with {combined['shards']} TPTP+Mizar shards.")
                state['collection_complete_notified'] = True
                save(state)

        for key, item in state['trainings'].items():
            training = training_statuses[key]
            training_state = training['state'].split('+')[0]
            if training_state == 'RUNNING' and key not in state['training_started_notified']:
                push(f"{item['job_name']} started: job {item['id']}, W&B run {item['run_name']}.")
                state['training_started_notified'].append(key)
                save(state)

            if training_state == 'TIMEOUT':
                submit_training_resume(state, key, item)
                continue
            if training_state in ('FAILED', 'OUT_OF_MEMORY', 'NODE_FAIL'):
                message = f"{item['job_name']} ended {training_state} (job {item['id']})."
                if message not in state['fatal']:
                    state['fatal'].append(message)
                    save(state)
                    push('Fatal Mizar run failure: ' + message)
                return
            if training_state == 'COMPLETED' and key not in state['training_finished_notified']:
                exit_code = training.get('exit')
                if exit_code != '0:0':
                    message = f"{item['job_name']} completed with exit {exit_code} (job {item['id']})."
                    if message not in state['fatal']:
                        state['fatal'].append(message)
                        save(state)
                        push('Fatal Mizar run failure: ' + message)
                    return
                info = training_checkpoints(item)
                if 100 not in info['available'] or select_checkpoint(info, 100) is None:
                    raise RuntimeError(f"{key} completed without epoch 100 and a scored validation checkpoint")
                push(f"{item['job_name']} finished successfully: job {item['id']}, W&B run {item['run_name']}.")
                state['training_finished_notified'].append(key)
                save(state)

        combined_key = 'Mizar40-TPTP-15m'
        if collection_done and state.get('datasets_prepared'):
            if combined_key not in state['trainings']:
                item = {
                    'id': 'initial', 'job_name': 'Mizar40-TPTP-15m-train',
                    'run_name': 'mizar40-tptp-15m-train',
                    'data_dir': f'{results}/Mizar40-TPTP-15m',
                    'dataset': f'{results}/Mizar40-TPTP-15m/dataset',
                }
                state['trainings'][combined_key] = item
                save(state)
            item = state['trainings'][combined_key]
            if item.get('id') == 'initial':
                submit_training_fresh(state, item)
                training_statuses[combined_key] = job_status(item['id'])

        evaluation_statuses = monitor_evaluations(state, training_statuses)
        now = datetime.datetime.now(ZoneInfo('Europe/London'))
        if now.hour in (10, 22):
            report_slot = now.strftime('%Y-%m-%d-%H')
            if report_slot not in state['status_reports']:
                progress = collection_progress()
                jobs = ', '.join(
                    f'{item["job_name"]}={training_statuses[key]["state"]}'
                    for key, item in state['trainings'].items()
                )
                progress_2m = two_minute_collection_progress()
                push(f'Mizar weekend status ({now:%Y-%m-%d %H:%M %Z}): collection={collector_state} {progress}; 2-minute collection={progress_2m}; GPU jobs: {jobs}; collection job={state["collector_id"]}.')
                state['status_reports'].append(report_slot)
                save(state)

        if collector_state in ('RUNNING', 'PENDING'):
            sleep_s = 7200
            if collector_state == 'RUNNING':
                elapsed = seconds(collector.get('elapsed', '0:00'))
                limit = seconds(collector.get('limit', '11:59:00'))
                sleep_s = max(30, min(sleep_s, limit - elapsed - 90))
        elif any(x['state'].split('+')[0] == 'RUNNING' for x in training_statuses.values()):
            running = next(x for x in training_statuses.values() if x['state'].split('+')[0] == 'RUNNING')
            elapsed = seconds(running.get('elapsed', '0:00'))
            limit = seconds(running.get('limit', '2:59:00'))
            sleep_s = max(30, min(7200, limit - elapsed - 90))
        else:
            sleep_s = 300
        if collection_done and all(item.get('complete') for item in state.get('evaluations', {}).values()) and len(state.get('evaluations', {})) == len(state['trainings']):
            return
        for job in evaluation_statuses:
            if job['state'] == 'RUNNING':
                sleep_s = min(sleep_s, max(30, seconds(job['limit']) - seconds(job['elapsed']) - 30))
        now = datetime.datetime.now(ZoneInfo('Europe/London'))
        next_report = now.replace(minute=0, second=0, microsecond=0)
        if now.hour < 10:
            next_report = next_report.replace(hour=10)
        elif now.hour < 22:
            next_report = next_report.replace(hour=22)
        else:
            next_report = (next_report + datetime.timedelta(days=1)).replace(hour=10)
        sleep_s = min(sleep_s, max(1, int((next_report - now).total_seconds())))
        state['last_success_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        state['monitor_errors'] = 0
        state['auth_notified'] = False
        save(state)
        time.sleep(sleep_s)


def main():
    state = load_state()
    while True:
        try:
            run(state)
            return
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            state['monitor_errors'] = state.get('monitor_errors', 0) + 1
            save(state)
            if 'Permission denied' in message or 'keyboard-interactive' in message or 'Authentication' in message:
                if not state.get('auth_notified'):
                    push('HPC authenticator needed: the SSH connection needs re-authentication.')
                    state['auth_notified'] = True
                    save(state)
            elif state['monitor_errors'] >= 3:
                fatal = 'Mizar monitor failed three consecutive checks: ' + message
                if fatal not in state['fatal']:
                    state['fatal'].append(fatal)
                    push('Fatal Mizar monitor error: ' + fatal)
                    save(state)
            time.sleep(60)


if __name__ == '__main__':
    with monitor_lock(state_path):
        main()
