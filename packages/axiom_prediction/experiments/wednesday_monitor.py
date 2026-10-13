import csv
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

state_path = pathlib.Path('/tmp/wednesday-retrained-monitor.json')
ssh = ['ssh', '-F', '/home/debian/.ssh/config', '-o', 'BatchMode=yes', 'hpc']
repo = '/home/msf43/connections'
cpu_repo = '/home/msf43/tmp/connections-wednesday-2218c97'
results = f'{repo}/results/wednesday/retrained'
models = ('SmallFull', 'DefaultFull', 'ExtraLargeFull')
policies = ('SatResetCoP', 'SatCoP')
training = {
    'SmallFull-SatResetCoP': {'id': '37794307', 'run_dir': f'{results}/SmallFull-SatResetCoP', 'data': f'{results}/SmallFull-SatResetCoP/model', 'model': 'SmallFull', 'policy': 'SatResetCoP', 'qos': 'gpu2', 'time': '2:59:00', 'logs': ['/home/msf43/tmp/wed-SmallFull-SatResetCoP-91476606/stderr.err'], 'stdout': ['/home/msf43/tmp/wed-SmallFull-SatResetCoP-91476606/stdout.out']},
    'SmallFull-SatCoP': {'id': '37794631', 'run_dir': f'{results}/SmallFull-SatCoP', 'data': f'{results}/SmallFull-SatCoP/model', 'model': 'SmallFull', 'policy': 'SatCoP', 'qos': 'gpu2', 'time': '2:59:00', 'logs': ['/home/msf43/tmp/wed-SmallFull-SatCoP-91476688/stderr.err'], 'stdout': ['/home/msf43/tmp/wed-SmallFull-SatCoP-91476688/stdout.out']},
    'DefaultFull-SatResetCoP': {'id': '37794954', 'run_dir': f'{results}/DefaultFull-SatResetCoP', 'data': f'{results}/DefaultFull-SatResetCoP/model', 'model': 'DefaultFull', 'policy': 'SatResetCoP', 'qos': 'gpu2', 'time': '2:59:00', 'logs': ['/home/msf43/tmp/wed-DefaultFull-SatResetCoP-91476784/stderr.err'], 'stdout': ['/home/msf43/tmp/wed-DefaultFull-SatResetCoP-91476784/stdout.out']},
    'DefaultFull-SatCoP': {'id': '37795232', 'run_dir': f'{results}/DefaultFull-SatCoP', 'data': f'{results}/DefaultFull-SatCoP/model', 'model': 'DefaultFull', 'policy': 'SatCoP', 'qos': 'gpu2', 'time': '2:59:00', 'logs': ['/home/msf43/tmp/wed-DefaultFull-SatCoP-91476858/stderr.err'], 'stdout': ['/home/msf43/tmp/wed-DefaultFull-SatCoP-91476858/stdout.out']},
    'ExtraLargeFull-SatResetCoP': {'id': '37795609', 'run_dir': f'{results}/ExtraLargeFull-SatResetCoP', 'data': f'{results}/ExtraLargeFull-SatResetCoP/model', 'model': 'ExtraLargeFull', 'policy': 'SatResetCoP', 'qos': 'gpu2', 'time': '2:59:00', 'logs': ['/home/msf43/tmp/wed-ExtraLargeFull-SatResetCoP-91476940/stderr.err'], 'stdout': ['/home/msf43/tmp/wed-ExtraLargeFull-SatResetCoP-91476940/stdout.out']},
    'ExtraLargeFull-SatCoP': {'id': '37795945', 'run_dir': f'{results}/ExtraLargeFull-SatCoP', 'data': f'{results}/ExtraLargeFull-SatCoP/model', 'model': 'ExtraLargeFull', 'policy': 'SatCoP', 'qos': 'gpu2', 'time': '2:59:00', 'logs': ['/home/msf43/tmp/wed-ExtraLargeFull-SatCoP-91477014/stderr.err'], 'stdout': ['/home/msf43/tmp/wed-ExtraLargeFull-SatCoP-91477014/stdout.out']},
}
training_data = {
    'SatResetCoP': f'{repo}/dataset/dataset',
    'SatCoP': f'{repo}/experiments/satcop2-20261002/dataset',
}


def remote(command):
    result = subprocess.run(ssh + [command], text=True, capture_output=True,
                            timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return result.stdout


def remote_output(command):
    result = subprocess.run(ssh + [command], text=True, capture_output=True,
                            timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-1500:])
    return result.stdout, result.stderr


def push(message):
    queue_notification('Wednesday HPC run', message)


def push_first_batch_line(state, key, args):
    notified = state.setdefault('first_batch_lines_notified', [])
    if key in notified:
        return
    push(f"First-time batch command ({key}):\n{shlex.join(args)}")
    notified.append(key)
    save(state)


def save(state):
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(state_path)


def sync_manifest(state):
    payload = json.dumps(state, indent=2)
    path = f'{results}/submission-manifest.json'
    code = (
        'import pathlib; p=pathlib.Path(' + repr(path) + '); '
        't=p.with_suffix(".tmp"); t.write_text(' + repr(payload) + '); t.replace(p)'
    )
    remote(f"/home/msf43/venv/bin/python -c {shlex.quote(code)}")


def record_cpu_checkout_job(job_id):
    path = '/home/msf43/connections/results/wednesday/checkout-records.json'
    code = f'''import json,pathlib
p=pathlib.Path({path!r}); d=json.loads(p.read_text())
for item in d['checkouts']:
 if item['path']=={cpu_repo!r}:
  item['dependent_job_ids']=sorted(set(item['dependent_job_ids']+[{job_id!r}]))
t=p.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2)+'\\n'); t.replace(p)
'''
    remote(f"/home/msf43/venv/bin/python -c {shlex.quote(code)}")


def start_state():
    state = {
        'training': training,
        'started': [],
        'failed': [],
        'cpu': {},
        'last_batch': 0,
        'reports': [],
        'fatal': [],
    }
    return state


def load_state():
    if not state_path.exists():
        state = start_state()
        save(state)
        return state
    state = json.loads(state_path.read_text())
    if 'training' not in state or not all('run_dir' in x for x in state['training'].values()):
        state = start_state()
        save(state)
    else:
        for key, item in training.items():
            if key not in state['training']:
                state['training'][key] = item
            else:
                for field, value in item.items():
                    state['training'][key].setdefault(field, value)
        state.setdefault('fallback_epochs', {})['SmallFull-SatResetCoP'] = 18
    return state


def snapshot(state):
    ids = [item['id'] for item in state['training'].values()]
    ids.extend(item['id'] for item in state['cpu'].values())
    ids = sorted(set(ids))
    code = f'''import json,pathlib,subprocess,re
ids={ids!r}
result={{}}
p=subprocess.run(['squeue','-h','-j',','.join(ids),'-o','%i|%T|%M|%l|%S'],capture_output=True,text=True)
active={{}}
for line in p.stdout.splitlines():
 row=line.split('|'); active[row[0]]={{'state':row[1],'elapsed':row[2],'limit':row[3],'start':row[4]}}
result.update(active)
missing=[job_id for job_id in ids if job_id not in active]
if missing:
 p=subprocess.run(['sacct','-X','-n','-P','-j',','.join(missing),'--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True)
 for line in p.stdout.splitlines():
  row=line.split('|')
  if len(row)>=3 and row[0] in missing: result[row[0]]={{'state':row[1].split()[0],'exit':row[2]}}
for job_id in ids: result.setdefault(job_id,{{'state':'UNKNOWN','exit':'unknown'}})
checkpoints={{}}
for name,item in {state['training']!r}.items():
 folder=pathlib.Path(item['data'])
 files=sorted(folder.glob('epoch-*.pt'))
 scores={{}}
 for log in [pathlib.Path(x) for x in item.get('logs',[])]:
  try: content=log.read_text(errors='replace')
  except OSError: continue
  for match in re.finditer(r'epoch (\\d+): held-out .*?roc_auc ([0-9.]+)',content): scores[int(match.group(1))]=float(match.group(2))
 for log in item.get('stdout',[]):
  try: lines=pathlib.Path(log).read_text(errors='replace').splitlines()
  except OSError: continue
  for line in lines:
   try: row=json.loads(line)
   except ValueError: continue
   if isinstance(row,dict) and row.get('event')=='evaluation' and row.get('roc_auc') is not None:
    scores[int(row['epoch'])]=float(row['roc_auc'])
 available=[int(x.stem.split('-')[-1]) for x in files]
 checkpoints[name]={{'available':available,'scores':scores}}
cpu_results={{}}
for key,item in {state['cpu']!r}.items():
 marker=pathlib.Path(item['output']+'.resume/completion.json')
 try: completion=json.loads(marker.read_text())
 except (OSError,ValueError): completion=None
 try: output_size=pathlib.Path(item['output']).stat().st_size
 except OSError: output_size=None
 cpu_results[item['id']]={{'completion':completion,'output_size':output_size}}
print(json.dumps({{'jobs':result,'checkpoints':checkpoints,'cpu_results':cpu_results}}))
'''
    return json.loads(remote(f"/home/msf43/venv/bin/python -c {shlex.quote(code)}"))


def wait_for_submission_slot(state):
    remaining = 60 - (time.time() - state['last_batch'])
    if remaining > 0:
        time.sleep(remaining)


def submit_cpu(state, key, item, checkpoint, eval_policy, epoch):
    wait_for_submission_slot(state)
    model = item['model']
    training_policy = item['policy']
    name = f'wed-{model}-{training_policy}-e{epoch:03d}'
    output = f'{results}/periodic/{model}-{training_policy}-e{epoch:03d}.csv'
    run = [
        '/home/msf43/venv/bin/python', 'axiom_predictor.py', 'run', '--cpu',
        '--multiprocess', '--policy', eval_policy, '--split', '10', '--parts', '8', '9',
        '--timeout-seconds', '120', '--csv', '--output', output,
        '--run-name', name, '--model', checkpoint, '--no-wandb',
    ]
    args = [
        '/home/msf43/bin/icelake-batch', '--time=3:59:00', '--job-name', name,
        *run,
    ]
    state.setdefault('cpu_intents', {})[key] = {
        'name': name, 'output': output, 'checkpoint': checkpoint, 'command': args,
        'epoch': epoch, 'parts': [8, 9], 'role': 'periodic',
    }
    save(state)
    sync_manifest(state)
    push_first_batch_line(state, f'cpu:{key}', args)
    result, _ = submit_once(remote_output, f"cd {shlex.quote(cpu_repo)} && {shlex.join(args)}",
                            f"wednesday:{key}:{state['cpu'].get(key, {}).get('id', 'initial')}", state['last_batch'])
    match = re.search(r'(?m)^(\d+)$', result.strip())
    if not match:
        raise RuntimeError(f'uncertain CPU submission {name}: {result[-1000:]}')
    state['cpu'][key] = {
        'id': match.group(1),
        'attempts': [match.group(1)],
        'output': output,
        'model': model,
        'training_policy': training_policy,
        'eval_policy': eval_policy,
        'checkpoint': checkpoint,
        'epoch': epoch,
        'parts': [8, 9],
        'role': 'periodic',
        'wandb': False,
        'training_key': key.rsplit('--e', 1)[0],
        'name': name,
    }
    record_cpu_checkout_job(match.group(1))
    state['last_batch'] = time.time()
    state.setdefault('checkpoints_used', {})[key] = checkpoint
    state['cpu_intents'].pop(key, None)
    save(state)
    sync_manifest(state)


def resume_cpu(state, key, item):
    wait_for_submission_slot(state)
    name = item.get('name', f"wed-{item['model']}-{item['training_policy']}-{item['eval_policy']}")
    run = [
        '/home/msf43/venv/bin/python', 'axiom_predictor.py', 'run', '--cpu',
        '--multiprocess', '--policy', item['eval_policy'], '--split', '10',
        '--parts', *map(str, item.get('parts', [item.get('part', 8)])),
        '--timeout-seconds', '120', '--csv', '--output', item['output'],
        '--run-name', name,
    ]
    run.append('--wandb' if item.get('wandb', True) else '--no-wandb')
    if item.get('checkpoint'):
        run.extend(['--model', item['checkpoint']])
    args = [
        '/home/msf43/bin/icelake-batch', '--time=3:59:00', '--job-name', name,
        *run,
    ]
    state.setdefault('cpu_intents', {})[key] = {
        'name': name, 'output': item['output'], 'checkpoint': item['checkpoint'], 'command': args,
    }
    save(state)
    sync_manifest(state)
    result, _ = submit_once(remote_output, f"cd {shlex.quote(cpu_repo)} && {shlex.join(args)}",
                            f"wednesday:{key}:{state['cpu'].get(key, {}).get('id', 'initial')}", state['last_batch'])
    match = re.search(r'(?m)^(\d+)$', result.strip())
    if not match:
        raise RuntimeError(f'uncertain CPU resume submission {name}: {result[-1000:]}')
    item['id'] = match.group(1)
    item['attempts'].append(match.group(1))
    record_cpu_checkout_job(match.group(1))
    state['cpu_intents'].pop(key, None)
    state['last_batch'] = time.time()
    save(state)
    sync_manifest(state)


def submit_resume(state, key, item):
    wait_for_submission_slot(state)
    previous_job_id = item['id']
    model = item['model']
    policy = item['policy']
    name = f'wed-{model}-{policy}'
    data_dir = item['run_dir']
    dataset = training_data[policy]
    args = [
        '/home/msf43/bin/ampere-batch', '--account=holden-sl3-gpu', '--time', item.get('time', '2:59:00'), '--job-name', name,
        '/home/msf43/venv/bin/python', 'axiom_predictor.py', 'train',
        '--data-dir', data_dir, '--dataset', dataset, '--network', model,
        '--split', '10', '--parts', '0', '1', '2', '3', '4', '5', '6', '7',
        '--evaluate', '8', '--policy', policy, '--run-name', name, '--wandb', '--resume',
    ]
    state.setdefault('training_intents', {})[key] = args
    save(state)
    sync_manifest(state)
    result, error = submit_once(remote_output, f"cd {shlex.quote(repo)} && {shlex.join(args)}",
                                f'wednesday:training:{key}:{previous_job_id}', state['last_batch'])
    match = re.search(r'(?m)^(\d+)$', result.strip())
    if not match:
        raise RuntimeError(f'uncertain training resume submission {name}: {result[-1000:]}')
    item['id'] = match.group(1)
    item['replacements'] = item.get('replacements', []) + [match.group(1)]
    log_match = re.search(r"executing > '([^']+)' 2> '([^']+)'", error)
    if log_match:
        item.setdefault('logs', []).append(log_match.group(2))
        item.setdefault('stdout', []).append(log_match.group(1))
    state['training_intents'].pop(key, None)
    state['last_batch'] = time.time()
    save(state)
    sync_manifest(state)

def submit_matrix_job(state, spec):
    wait_for_submission_slot(state)
    key = spec['key']
    name = spec['name']
    output = spec['output']
    run = [
        '/home/msf43/venv/bin/python', 'axiom_predictor.py', 'run', '--cpu',
        '--multiprocess', '--policy', spec['eval_policy'], '--split', '10',
        '--parts', *map(str, spec['parts']), '--timeout-seconds', '120', '--csv',
        '--output', output, '--run-name', name, '--wandb',
    ]
    if spec.get('checkpoint'):
        run.extend(['--model', spec['checkpoint']])
    args = [
        '/home/msf43/bin/icelake-batch', '--time=3:59:00', '--job-name', name,
        *run,
    ]
    state.setdefault('cpu_intents', {})[key] = {
        'name': name, 'output': output, 'checkpoint': spec.get('checkpoint'),
        'command': args, 'model': spec['model'], 'training_policy': spec.get('training_policy'),
        'eval_policy': spec['eval_policy'], 'epoch': spec.get('epoch'),
        'parts': spec['parts'], 'role': 'final', 'wandb': True,
    }
    save(state)
    sync_manifest(state)
    push_first_batch_line(state, f'cpu:{key}', args)
    stdout, stderr = submit_once(remote_output, f"cd {shlex.quote(repo)} && {shlex.join(args)}",
                                  f'wednesday:{key}:initial', state['last_batch'])
    match = re.search(r'(?m)^(\d+)$', stdout.strip())
    if not match:
        raise RuntimeError(f'uncertain CPU submission {name}: {stdout[-1000:]} {stderr[-1000:]}')
    job_id = match.group(1)
    state.setdefault('cpu', {})[key] = {
        'id': job_id, 'attempts': [job_id], 'output': output, 'model': spec['model'],
        'training_policy': spec.get('training_policy'), 'eval_policy': spec['eval_policy'],
        'checkpoint': spec.get('checkpoint'), 'epoch': spec.get('epoch'),
        'parts': spec['parts'], 'role': 'final', 'wandb': True, 'name': name,
    }
    state['cpu_intents'].pop(key, None)
    state['last_batch'] = time.time()
    save(state)
    sync_manifest(state)

def submit_matrix(state, snap):
    jobs = []
    for training_key, item in state['training'].items():
        selected = select_checkpoint(snap['checkpoints'].get(training_key, {}), 10**9)
        if selected is None:
            raise RuntimeError(f'no held-out ROC AUC checkpoint for {training_key}')
        checkpoint = f"{item['data']}/epoch-{selected:04d}.pt"
        for eval_policy in policies:
            key = f"{training_key}--final--{eval_policy}"
            policy_name = item['policy'] if item['policy'] == eval_policy else f"{item['policy']}-{eval_policy}"
            name = f"wed-{item['model']}-{policy_name}"
            jobs.append({
                'key': key, 'name': name,
                'output': f"{results}/final-evaluation/{key}.csv",
                'model': item['model'], 'training_policy': item['policy'],
                'eval_policy': eval_policy, 'parts': [8, 9], 'epoch': selected,
                'checkpoint': checkpoint,
            })
    if len(jobs) != 12:
        raise RuntimeError(f'expected 12 CPU matrix jobs, generated {len(jobs)}')
    state['matrix_expected'] = len(jobs)
    state['matrix_keys'] = [spec['key'] for spec in jobs]
    save(state)
    sync_manifest(state)
    for spec in jobs:
        if spec['key'] not in state['cpu']:
            submit_matrix_job(state, spec)
    state['matrix_submitted'] = True
    save(state)
    sync_manifest(state)


def schedule_periodic(state, snap):
    for training_key, item in state['training'].items():
        epochs = sorted(
            epoch for epoch in snap['checkpoints'].get(training_key, {}).get('available', [])
            if epoch > 0 and epoch % 10 == 0
        )
        for epoch in epochs:
            key = f'{training_key}--e{epoch:04d}'
            if key in state['cpu']:
                continue
            checkpoint = f"{item['data']}/epoch-{epoch:04d}.pt"
            submit_cpu(state, key, item, checkpoint, item['policy'], epoch)


def log_evaluation_metrics(state, snap):
    logged = set(state.get('wandb_metrics_logged', []))
    for training_key, training_item in state['training'].items():
        training_status = snap['jobs'].get(training_item['id'], {})
        if training_status.get('state') != 'COMPLETED' or training_status.get('exit') != '0:0':
            continue
        for key, item in state['cpu'].items():
            if item.get('role') != 'periodic' or item.get('training_key') != training_key:
                continue
            marker = f'{training_key}--e{item["epoch"]:04d}'
            if marker in logged or snap['jobs'].get(item['id'], {}).get('state') != 'COMPLETED':
                continue
            result = snap['cpu_results'].get(item['id'], {})
            completion = result.get('completion')
            if not valid_completion(result):
                continue
            code = '''import csv,json,pathlib,sys,wandb
model_dir=pathlib.Path(sys.argv[1]); filename=pathlib.Path(sys.argv[2]); epoch=int(sys.argv[3])
identity=json.loads((model_dir/"wandb-run.json").read_text())
with filename.open(newline="") as stream: rows=list(csv.DictReader(stream))
solved=sum(row.get("proved","").lower() in ("true","1") for row in rows)
wandb.login(key=pathlib.Path("/home/msf43/connections/secrets/wandb_key").read_text().strip())
run=wandb.init(entity="mfixman-phd-team",project="axiom-prediction",id=identity["id"],resume="must")
run.define_metric("evaluation/epoch")
run.define_metric("evaluation/real_solved",step_metric="evaluation/epoch")
run.log({"evaluation/epoch":epoch,"evaluation/real_solved":solved},step=run.step+1)
run.finish()
'''
            remote(
                f"/home/msf43/venv/bin/python -c {shlex.quote(code)} "
                f"{shlex.quote(training_item['data'])} {shlex.quote(item['output'])} {item['epoch']}"
            )
            state.setdefault('wandb_metrics_logged', []).append(marker)
            logged.add(marker)
            save(state)
            sync_manifest(state)

def combine_if_done(state, snap):
    if state.get('combined'):
        return
    matrix_keys = state.get('matrix_keys', [])
    if not state.get('matrix_submitted') or len(matrix_keys) != state.get('matrix_expected', 12):
        return
    if any(snap['jobs'].get(state['cpu'].get(key, {}).get('id'), {}).get('state') != 'COMPLETED' for key in matrix_keys):
        return
    rows = snap['jobs']
    cpu_files = {}
    for key, item in state['cpu'].items():
        if rows.get(item['id'], {}).get('state') != 'COMPLETED' or rows[item['id']].get('exit') != '0:0':
            return
        completion = snap['cpu_results'].get(item['id'], {}).get('completion')
        output_size = snap['cpu_results'].get(item['id'], {}).get('output_size')
        if not valid_completion(snap['cpu_results'].get(item['id'], {})):
            return
        cpu_files[key] = {
            'output': item['output'],
            'model': item['model'],
            'training_policy': item['training_policy'],
            'eval_policy': item['eval_policy'],
            'epoch': item['epoch'],
            'evaluation_role': item.get('role', 'final'),
            'evaluation_key': key,
        }
    if len(cpu_files) != len(state['cpu']):
        return
    logs = {key: item.get('stdout', []) for key, item in state['training'].items()}
    code = f'''import csv,json,pathlib
root=pathlib.Path('/home/msf43/connections/results/wednesday/retrained')
target=root/'evaluation-results.csv'
training_logs={logs!r}
cpu_files={cpu_files!r}
all_rows=[]; fields=set()
for key,log_names in training_logs.items():
 model,training_policy=key.split('-',1)
 for log_name in log_names:
  log=pathlib.Path(log_name)
  try: lines=log.read_text(errors='replace').splitlines()
  except OSError: lines=[]
  for line in lines:
   try: row=json.loads(line)
   except ValueError: continue
   if not isinstance(row,dict) or 'event' not in row: continue
   row={{'record_type':'training','model':model,'training_policy':training_policy,**row}}
   fields.update(row); all_rows.append(row)
  error_log=log.with_name('stderr.err')
  try: lines=error_log.read_text(errors='replace').splitlines()
  except OSError: lines=[]
  for line in lines:
   try: row=json.loads(line)
   except ValueError: continue
   if not isinstance(row,dict) or 'event' in row: continue
   row={{'event':'training_summary','record_type':'training','model':model,'training_policy':training_policy,**row}}
   fields.update(row); all_rows.append(row)
for item in cpu_files.values():
 path=pathlib.Path(item['output'])
 with path.open(newline='') as stream:
  reader=csv.DictReader(stream)
  for row in reader:
   row={{'event':'proof_search','record_type':'evaluation','model':item['model'],'training_policy':item['training_policy'],'evaluation_policy':item['eval_policy'],'checkpoint_epoch':item['epoch'],'evaluation_role':item['evaluation_role'],'evaluation_key':item['evaluation_key'],'source_csv':str(path),**row}}
   fields.update(row); all_rows.append(row)
headers=['record_type','event','model','training_policy','evaluation_policy',*sorted(fields-{{'record_type','event','model','training_policy','evaluation_policy'}})]
temporary=target.with_suffix('.csv.tmp')
with temporary.open('w',newline='') as stream:
 writer=csv.DictWriter(stream,fieldnames=headers,lineterminator='\\n'); writer.writeheader()
 for row in all_rows:
  writer.writerow({{key:json.dumps(value,default=str) if isinstance(value,(dict,list)) else value for key,value in row.items()}})
temporary.replace(target)
print(len(all_rows))
'''
    remote(f"/home/msf43/venv/bin/python -c {shlex.quote(code)}")
    state['combined'] = True
    save(state)


def scheduled_reports(now):
    dates = (now.date(), now.date() + datetime.timedelta(days=1))
    return [
        datetime.datetime.combine(day, time, now.tzinfo)
        for day in dates
        for time in (datetime.time(10), datetime.time(22))
    ]


def maybe_report(state, snap, now):
    due = [
        moment for moment in scheduled_reports(now)
        if moment <= now and now - moment < datetime.timedelta(hours=1)
    ]
    for moment in due:
        key = moment.isoformat()
        if key in state['reports']:
            continue
        training_status = {}
        for name, info in state['training'].items():
            job = snap['jobs'].get(info['id'], {})
            checkpoints = snap['checkpoints'].get(name, {})
            epochs = checkpoints.get('available', [])
            training_status[name] = {
                'state': job.get('state'),
                'epoch': max(epochs, default=0),
                'best_roc_auc': max(checkpoints.get('scores', {}).values(), default=None),
            }
        cpu_states = {}
        for item in state['cpu'].values():
            status = snap['jobs'].get(item['id'], {}).get('state', 'UNKNOWN')
            cpu_states[status] = cpu_states.get(status, 0) + 1
        final_count = sum(item.get('role') == 'final' for item in state.get('cpu', {}).values())
        periodic_count = sum(item.get('role') == 'periodic' for item in state.get('cpu', {}).values())
        push(f'Wednesday retraining status at {moment:%Y-%m-%d %H:%M %Z}: training={training_status}; CPU states={cpu_states}; final jobs={final_count}/12; periodic jobs={periodic_count}; combined={state.get("combined", False)}')
        state['reports'].append(key)
        save(state)


def next_poll_seconds(snap):
    interval = 900
    for job in snap['jobs'].values():
        if job.get('state') in ('COMPLETING', 'CONFIGURING'):
            interval = min(interval, 30)
            continue
        if job.get('state') != 'RUNNING':
            continue
        elapsed = job.get('elapsed', '0:00').split(':')
        limit = job.get('limit', '2:59:00').split(':')
        elapsed_s = sum(int(value) * (60 ** index) for index, value in enumerate(reversed(elapsed)))
        limit_s = sum(int(value) * (60 ** index) for index, value in enumerate(reversed(limit)))
        interval = min(interval, max(30, limit_s - elapsed_s - 30))
    now = datetime.datetime.now(ZoneInfo('Europe/London'))
    reports = scheduled_reports(now)
    next_report = min(moment for moment in reports if moment > now)
    until_report = int((next_report - now).total_seconds())
    return max(1, min(interval, until_report))


def main():
    state = load_state()
    while True:
        try:
            flush_notifications()
            sync_manifest(state)
            snap = snapshot(state)
            state['last_snapshot'] = snap
            now = datetime.datetime.now(ZoneInfo('Europe/London'))
            for key, item in list(state['training'].items()):
                job_id = item['id']
                status = snap['jobs'].get(job_id, {}).get('state', 'UNKNOWN')
                if status == 'RUNNING' and key not in state['started']:
                    state['started'].append(key)
                    save(state)
                    push(f'Wednesday training started: {key}, job {job_id}, output {item["run_dir"]}.')
                if status == 'TIMEOUT':
                    submit_resume(state, key, item)
                    continue
                if status not in ('PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING', 'COMPLETED', 'UNKNOWN'):
                    failure = f'{key} ended in {status} (job {job_id}); automatic resume not safe.'
                    if failure not in state['fatal']:
                        state['fatal'].append(failure)
                        save(state)
                        push('Fatal Wednesday run failure: ' + failure)
                if status not in ('PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING', 'COMPLETED', 'UNKNOWN'):
                    continue
                observed = snap['jobs'].get(job_id, {})
                if status == 'COMPLETED' and observed.get('exit') not in (None, '0:0'):
                    failure = f'{key} completed with exit {observed.get("exit")} (job {job_id}).'
                    if failure not in state['fatal']:
                        state['fatal'].append(failure)
                        save(state)
                        push('Fatal Wednesday run failure: ' + failure)
            schedule_periodic(state, snap)
            for key, item in state['cpu'].items():
                current = snap['jobs'].get(item['id'])
                if current is None:
                    continue
                status = current['state']
                if status == 'TIMEOUT':
                    if item.get('ignore_failure'):
                        item['ignored_status'] = status
                        continue
                    resume_cpu(state, key, item)
                elif status not in ('PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING', 'COMPLETED', 'UNKNOWN'):
                    if item.get('ignore_failure'):
                        item['ignored_status'] = status
                        continue
                    failure = f"CPU evaluation {key} ended in {status} (job {item['id']})."
                    if failure not in state['fatal']:
                        state['fatal'].append(failure)
                        save(state)
                        push('Fatal Wednesday run failure: ' + failure)
                elif status == 'COMPLETED' and current.get('exit') not in (None, '0:0'):
                    failure = f"CPU evaluation {key} completed with exit {current.get('exit')} (job {item['id']})."
                    if failure not in state['fatal']:
                        state['fatal'].append(failure)
                        save(state)
                        push('Fatal Wednesday run failure: ' + failure)
            training_done = all(
                snap['jobs'].get(item['id'], {}).get('state') == 'COMPLETED'
                and snap['jobs'].get(item['id'], {}).get('exit') in (None, '0:0')
                for item in state['training'].values()
            )
            if training_done and not state.get('matrix_submitted'):
                submit_matrix(state, snap)
            log_evaluation_metrics(state, snap)
            combine_if_done(state, snap)
            maybe_report(state, snap, now)
            state['last_success_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            state['monitor_errors'] = 0
            state['auth_notified'] = False
            save(state)
            if state.get('combined') and len(state.get('reports', [])) >= 2:
                return
        except Exception as error:
            message = f'Fatal Wednesday monitor error: {type(error).__name__}: {error}'
            state['monitor_errors'] = state.get('monitor_errors', 0) + 1
            save(state)
            if 'Permission denied' in message or 'keyboard-interactive' in message:
                if not state.get('auth_notified'):
                    push('HPC authenticator needed: the SSH connection needs re-authentication.')
                    state['auth_notified'] = True
                    save(state)
            elif state['monitor_errors'] >= 3 and message not in state['fatal']:
                state.setdefault('fatal', []).append(message)
                push(message)
            time.sleep(300)
        time.sleep(next_poll_seconds(state.get('last_snapshot', {'jobs': {}})))


if __name__ == '__main__':
    with monitor_lock(state_path):
        main()
