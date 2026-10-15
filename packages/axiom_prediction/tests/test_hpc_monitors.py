from axiom_prediction.io import write_csv
from axiom_prediction.dataset import dataset_fields
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import sys

import pytest

folder = Path(__file__).resolve().parents[1] / 'experiments'
sys.path.insert(0, str(folder))
import monitor_support as support
import mizar_monitor as mizar
import wednesday_monitor as wednesday


def test_select_checkpoint_after_json_roundtrip():
    info = json.loads(json.dumps({'available': [1, 2, 3, 4], 'scores': {1: .8, 2: .9, 3: .9, 4: .95}}))
    assert wednesday.select_checkpoint(info, 3) == 2
    assert wednesday.select_checkpoint(info, 4) == 4
    assert wednesday.select_checkpoint({'available': [1], 'scores': {'1': float('nan')}}, 100) is None


def test_submission_receipt_prevents_duplicate_after_state_write_failure(tmp_path):
    calls = []
    def remote(command):
        calls.append(command)
        return '12345\n', 'log path'
    assert support.submit_once(remote, 'batch command', 'run:oldjob', directory = tmp_path)[0] == '12345\n'
    assert support.submit_once(remote, 'batch command', 'run:oldjob', directory = tmp_path)[0] == '12345\n'
    assert calls == ['batch command']
    with pytest.raises(RuntimeError, match = 'command changed'):
        support.submit_once(remote, 'different command', 'run:oldjob', directory = tmp_path)


def test_uncertain_submission_is_not_repeated(tmp_path):
    calls = []
    def remote(command):
        calls.append(command)
        raise TimeoutError('lost reply after submitting')
    with pytest.raises(TimeoutError):
        support.submit_once(remote, 'batch command', 'run:oldjob', directory = tmp_path)
    with pytest.raises(RuntimeError, match = 'Uncertain submission'):
        support.submit_once(remote, 'batch command', 'run:oldjob', directory = tmp_path)
    assert len(calls) == 1


def test_global_interval_across_two_monitors(tmp_path, monkeypatch):
    clock = [1000.0]
    sleeps = []
    monkeypatch.setattr(support.time, 'time', lambda: clock[0])
    def sleep(duration):
        sleeps.append(duration)
        clock[0] += duration
    monkeypatch.setattr(support.time, 'sleep', sleep)
    remote = lambda command: ('123\n', '')
    support.submit_once(remote, 'batch first', 'wednesday:one', directory = tmp_path)
    support.submit_once(remote, 'batch second', 'mizar:one', directory = tmp_path)
    assert sleeps == [60.0]


def test_completion_requires_success_and_output_size():
    assert support.valid_completion({'completion': {'exit_code': 0, 'output_bytes': 10}, 'output_size': 10})
    assert not support.valid_completion({'completion': {'exit_code': 1, 'output_bytes': 10}, 'output_size': 10})
    assert not support.valid_completion({'completion': {'exit_code': 0, 'output_bytes': 10}, 'output_size': 11})
    assert not support.valid_completion({'completion': {'exit_code': 0}})


def staging_remote(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    root = tmp_path / 'results'
    monkeypatch.setattr(mizar, 'results', str(root))
    def remote(command):
        code = shlex.split(command)[-1].replace('/home/msf43/learncop/artifacts/axiom-dataset/dataset', str(source))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code, {})
        return output.getvalue()
    monkeypatch.setattr(mizar, 'remote', remote)
    shard = root / 'Mizar40/dataset/Mizar40.csv'
    shard.parent.mkdir(parents = True)
    header = {'schema': 'learncop.axiom_prediction.dataset-shard.v2', 'collection': {'sat_policy': 'satresetcop'}}
    write_csv(shard, [header], dataset_fields)
    return source, root, header


def test_combined_staging_rejects_missing_tptp(tmp_path, monkeypatch):
    staging_remote(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match = 'no TPTP shards'):
        mizar.stage_combined_dataset()


def test_combined_staging_is_idempotent_and_preserves_inputs(tmp_path, monkeypatch):
    source, root, header = staging_remote(tmp_path, monkeypatch)
    shard = source / 'TPTP.csv'
    write_csv(shard, [header], dataset_fields)
    before = shard.read_bytes()
    assert mizar.stage_combined_dataset()['shards'] == 2
    assert mizar.stage_combined_dataset()['shards'] == 2
    assert (root / 'Mizar40-TPTP/dataset/TPTP.csv').resolve() == shard
    assert shard.read_bytes() == before


def test_mizar_test_uses_best_validation_epoch_part9_and_same_output_on_timeout(monkeypatch):
    state = {'last_batch': 0}
    item = {'id': 'train123', 'data_dir': '/results/Mizar40', 'dataset': '/results/Mizar40/dataset'}
    monkeypatch.setattr(mizar, 'training_checkpoints', lambda item: {'available': [80, 100], 'scores': {'80': .9, '100': .8}})
    monkeypatch.setattr(mizar, 'save', lambda state: None)
    monkeypatch.setattr(mizar, 'push', lambda message: None)
    commands = []
    monkeypatch.setattr(mizar, 'batch', lambda state, line, operation: commands.append((line, operation)) or str(len(commands)))
    mizar.submit_evaluation(state, 'Mizar40', item)
    first = commands[0][0]
    assert first[first.index('--parts') + 1] == '9'
    assert first[first.index('--model') + 1].endswith('epoch-0080.pt')
    assert '--cpu' in first and '--wandb' in first and 'evaluate' in first
    assert '--time=3:59:00' in first
    status = {'Mizar40': {'state': 'COMPLETED', 'exit': '0:0'}}
    state['trainings'] = {'Mizar40': item}
    monkeypatch.setattr(mizar, 'job_status', lambda job: {'state': 'TIMEOUT'})
    mizar.monitor_evaluations(state, status)
    assert commands[1][0] == first
    assert commands[0][1] != commands[1][1]


def test_test_evaluation_waits_for_100_epochs(monkeypatch):
    monkeypatch.setattr(mizar, 'training_checkpoints', lambda item: {'available': [80], 'scores': {'80': .9}})
    with pytest.raises(RuntimeError, match = 'epoch 100'):
        mizar.submit_evaluation({'last_batch': 0}, 'Mizar40', {})


def test_prepare_never_updates_code_while_training_runs(monkeypatch):
    state = {'trainings': {'Mizar40': {'id': '1'}}}
    monkeypatch.setattr(mizar, 'job_status', lambda job: {'state': 'RUNNING'})
    commands = []
    monkeypatch.setattr(mizar, 'remote', lambda command: commands.append(command))
    monkeypatch.setattr(mizar, 'save', lambda state: None)
    monkeypatch.setattr(mizar, 'update_training_checkout', lambda: pytest.fail('changed active checkout'))
    assert not mizar.prepare_training(state)
    assert commands == ['scancel 1']


def test_monitor_has_single_instance_lock(tmp_path):
    with support.monitor_lock(tmp_path / 'state.json'):
        with pytest.raises(BlockingIOError):
            support.monitor_lock(tmp_path / 'state.json')


def test_failed_notifications_survive_until_next_iteration(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.setattr(support.Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(support, 'notification_slot', lambda: '2026-10-11 10:00 BST')
    secrets = tmp_path / 'secrets'
    secrets.mkdir()
    (secrets / 'pushbullet_token').write_text('test-token')
    def unavailable(*args, **kwargs):
        raise TimeoutError('offline')
    monkeypatch.setattr(urllib.request, 'urlopen', unavailable)
    support.queue_notification('Test', 'Done')
    receipt = next((tmp_path / '.local/state/hpc-notifications').glob('*.json'))
    assert not json.loads(receipt.read_text()).get('delivered')
    sent = []
    def delivered(request, **kwargs):
        sent.append(json.loads(request.data))
        return io.BytesIO(b'{}')
    monkeypatch.setattr(urllib.request, 'urlopen', delivered)
    support.flush_notifications()
    support.flush_notifications()
    assert len(sent) == 1
    assert json.loads(receipt.read_text())['delivered']


def test_notifications_batch_once_at_scheduled_slots(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.setattr(support.Path, 'home', lambda: tmp_path)
    secrets = tmp_path / 'secrets'
    secrets.mkdir()
    (secrets / 'pushbullet_token').write_text('test-token')
    slot = {'value': None}
    monkeypatch.setattr(support, 'notification_slot', lambda: slot['value'])
    sent = []
    def delivered(request, **kwargs):
        sent.append(json.loads(request.data))
        return io.BytesIO(b'{}')
    monkeypatch.setattr(urllib.request, 'urlopen', delivered)
    support.queue_notification('Collection', 'Part 0 started')
    support.queue_notification('Training', 'Model started')
    assert not sent
    slot['value'] = '2026-10-11 22:00 BST'
    support.flush_notifications()
    support.queue_notification('Evaluation', 'CPU jobs started')
    support.flush_notifications()
    assert len(sent) == 1
    assert 'Part 0 started' in sent[0]['body']
    assert 'Model started' in sent[0]['body']
    slot['value'] = '2026-10-12 10:00 BST'
    support.flush_notifications()
    assert len(sent) == 2
    assert sent[1]['body'] == 'Evaluation:\nCPU jobs started'


def test_final_csv_distinguishes_periodic_from_final(tmp_path, monkeypatch):
    import csv
    state = {'training': {}, 'matrix_submitted': True, 'matrix_keys': ['final'], 'matrix_expected': 1, 'cpu': {}}
    snap = {'jobs': {}, 'cpu_results': {}}
    for index, role in enumerate(['periodic', 'final']):
        path = tmp_path / (role + '.csv')
        path.write_text('problem,proved\nABC001+1.p,True\n')
        job = str(index)
        state['cpu'][role] = {'id': job, 'output': str(path), 'model': 'SmallFull', 'training_policy': 'SatCoP', 'eval_policy': 'SatCoP', 'epoch': 10, 'role': role}
        snap['jobs'][job] = {'state': 'COMPLETED', 'exit': '0:0'}
        snap['cpu_results'][job] = {'completion': {'exit_code': 0, 'output_bytes': path.stat().st_size}, 'output_size': path.stat().st_size}
    def remote(command):
        code = shlex.split(command)[-1].replace('/home/msf43/connections/results/wednesday/retrained', str(tmp_path))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, {})
    monkeypatch.setattr(wednesday, 'remote', remote)
    monkeypatch.setattr(wednesday, 'save', lambda state: None)
    wednesday.combine_if_done(state, snap)
    with (tmp_path / 'evaluation-results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row['evaluation_role'] for row in rows} == {'periodic', 'final'}
    assert len(rows) == 2
    assert state['combined']


def test_generated_mizar_evaluation_command_parses(monkeypatch):
    from axiom_prediction.cli import build_parser
    state = {'last_batch': 0}
    item = {'id': 'train1', 'data_dir': '/results/mizar', 'dataset': '/results/mizar/dataset'}
    monkeypatch.setattr(mizar, 'training_checkpoints', lambda item: {'available': [100], 'scores': {'100': .9}})
    monkeypatch.setattr(mizar, 'save', lambda state: None)
    monkeypatch.setattr(mizar, 'push', lambda message: None)
    def batch(state, line, operation):
        args = build_parser().parse_args(line[line.index('axiom_predictor.py') + 1:])
        assert args.command == 'evaluate' and args.parts == [9] and args.split == 10
        assert args.device == 'cpu'
        assert args.output == Path('/results/mizar/test.csv')
        return '1'
    monkeypatch.setattr(mizar, 'batch', batch)
    mizar.submit_evaluation(state, 'Mizar40', item)


def test_preparation_holds_training_until_both_code_and_data_ready(monkeypatch):
    state = {'trainings': {'Mizar40': {'id': '1'}, 'Mizar40-TPTP': {'id': '2'}}}
    events = []
    monkeypatch.setattr(mizar, 'job_status', lambda job: {'state': 'PENDING'})
    monkeypatch.setattr(mizar, 'remote', lambda command: events.append(command) or '')
    monkeypatch.setattr(mizar, 'save', lambda state: None)
    monkeypatch.setattr(mizar, 'update_training_checkout', lambda: events.append('update code'))
    monkeypatch.setattr(mizar, 'stage_combined_dataset', lambda: events.append('stage data') or {'shards': 2})
    assert mizar.prepare_training(state)
    assert events.index('scontrol hold 1') < events.index('update code')
    assert events.index('scontrol hold 2') < events.index('update code')
    assert events.index('stage data') < events.index('scontrol release 1')
    assert state['datasets_prepared'] and not state['preparation_holds']


def test_unknown_scheduler_state_does_not_prepare_or_cancel(monkeypatch):
    monkeypatch.setattr(mizar, 'job_status', lambda job: {'state': 'UNKNOWN'})
    monkeypatch.setattr(mizar, 'remote', lambda command: pytest.fail('acted on unknown job'))
    assert not mizar.prepare_training({'trainings': {'model': {'id': '1'}}})
