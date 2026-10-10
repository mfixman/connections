import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent = 2) + '\n')
    temporary.replace(path)


def monitor_lock(state_path):
    stream = state_path.with_suffix('.lock').open('a')
    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return stream


def submit_once(remote_output, command, operation, last_batch = 0, directory = None):
    # Keep uncertain submissions recorded so a lost SSH reply cannot duplicate a job.
    directory = Path(directory or Path.home() / '.local/state/hpc-submissions')
    directory.mkdir(parents = True, exist_ok = True)
    receipt = directory / (hashlib.sha256(operation.encode()).hexdigest() + '.json')
    with (directory / 'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['command'] != command:
                raise RuntimeError(f'Submission command changed for {operation}')
            if 'stdout' in saved:
                return saved['stdout'], saved['stderr']
            raise RuntimeError(f'Uncertain submission {operation}; reconcile receipt {receipt} before retrying')

        timestamp = directory / 'last-batch.json'
        previous = json.loads(timestamp.read_text()) if timestamp.exists() else 0
        remaining_s = 60 - (time.time() - max(previous, last_batch))
        if remaining_s > 0:
            time.sleep(remaining_s)
        saved = {'operation': operation, 'command': command, 'started': time.time()}
        write_json(receipt, saved)
        write_json(timestamp, time.time())
        try:
            stdout, stderr = remote_output(command)
            if not re.search(r'(?m)^\d+$', stdout.strip()):
                raise RuntimeError(f'Uncertain submission {operation}; no scheduler job ID returned')
            saved.update(stdout = stdout, stderr = stderr)
            write_json(receipt, saved)
            return stdout, stderr
        finally:
            write_json(timestamp, time.time())


def select_checkpoint(info, trigger_epoch):
    scores = {int(epoch): float(score) for epoch, score in info.get('scores', {}).items() if score is not None}
    eligible = [epoch for epoch in info.get('available', [])
                if epoch <= trigger_epoch and epoch in scores and math.isfinite(scores[epoch])]
    return min(eligible, key = lambda epoch: (-scores[epoch], epoch)) if eligible else None


def valid_completion(result):
    completion = result.get('completion')
    return bool(completion and completion.get('exit_code') == 0
                and result.get('output_size') is not None
                and completion.get('output_bytes') == result['output_size'])


def queue_notification(title, message):
    directory = Path.home() / '.local/state/hpc-notifications'
    directory.mkdir(parents = True, exist_ok = True)
    key = hashlib.sha256((title + '\n' + message).encode()).hexdigest()
    path = directory / (key + '.json')
    with (directory / 'outbox.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            write_json(path, {'type': 'note', 'title': title, 'body': message})
    flush_notifications()


def flush_notifications():
    import urllib.request

    directory = Path.home() / '.local/state/hpc-notifications'
    directory.mkdir(parents = True, exist_ok = True)
    with (directory / 'outbox.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in sorted(directory.glob('*.json')):
            payload = json.loads(path.read_text())
            if payload.get('delivered'):
                continue
            try:
                token = (Path.home() / 'secrets/pushbullet_token').read_text().strip()
                request = urllib.request.Request(
                    'https://api.pushbullet.com/v2/pushes', data = json.dumps(payload).encode(),
                    headers = {'Access-Token': token, 'Content-Type': 'application/json'},
                )
                with urllib.request.urlopen(request, timeout = 30) as response:
                    response.read()
            except Exception as error:
                print(f'Notification retained for retry: {type(error).__name__}', flush = True)
                return
            payload['delivered'] = True
            write_json(path, payload)
