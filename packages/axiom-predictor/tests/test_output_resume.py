import csv
import io
import json
from types import SimpleNamespace

import pytest

from axiom_prediction.output import COMMAND_FIELDS, output_format, write_record
from axiom_prediction.output_resume import OutputJournal


def open_journal(path, command='run', use_csv=False, identity=None):
    return OutputJournal(path, command, use_csv,
                         list(dict.fromkeys(COMMAND_FIELDS[command])), identity or {'command': command})


@pytest.mark.parametrize('use_csv', [False, True])
def test_journal_repairs_torn_tail_and_preserves_multiline(tmp_path, use_csv, capsys):
    path = tmp_path / 'results'
    row = {'problem': 'a,b\n"c"', 'proved': False, 'seconds': 1.25, 'outcome': 'Timeout'}
    with output_format('run', use_csv, path, {'command': 'run'}):
        write_record(row)
    committed = path.read_bytes()
    with path.open('a') as f:
        f.write('"unfinished\nquoted' if use_csv else '{"problem":')
    journal = open_journal(path, use_csv=use_csv)
    assert journal.records[0]['problem'] == row['problem']
    assert journal.records[0]['proved'] is False
    assert journal.records[0]['seconds'] == 1.25
    journal.write(row)
    journal.close()
    assert path.read_bytes() == committed
    assert capsys.readouterr().out == ''


def test_journal_refuses_mixed_configuration_and_concurrent_writers(tmp_path):
    path = tmp_path / 'results'
    first = open_journal(path)
    with pytest.raises(BlockingIOError):
        open_journal(path)
    first.close()
    with pytest.raises(ValueError, match='different command'):
        open_journal(path, identity={'command': 'different'})


@pytest.mark.parametrize('use_csv', [False, True])
def test_run_cli_resumes_only_unfinished_and_keeps_summary_totals(tmp_path, monkeypatch, use_csv, capsys):
    from axiom_prediction import cli, run
    path = tmp_path / 'results'
    monkeypatch.setattr(cli, 'selected_problems', lambda args: ['a.p', 'b.p'])
    calls = []
    def interrupted(problems, **kwargs):
        calls.append(problems)
        yield {'problem': 'a.p', 'proved': True, 'seconds': 2.0, 'outcome': 'proved'}
        raise KeyboardInterrupt()
    monkeypatch.setattr(run, 'run_problems', interrupted)
    args = ['run', '--output', str(path), '--no-wandb', '--device', 'cpu'] + (['--csv'] if use_csv else [])
    with pytest.raises(KeyboardInterrupt):
        cli.main(args)
    def finish(problems, **kwargs):
        calls.append(problems)
        yield {'problem': 'b.p', 'proved': False, 'seconds': 3.0, 'outcome': 'Timeout'}
    monkeypatch.setattr(run, 'run_problems', finish)
    assert cli.main(args) == 0
    assert calls == [['a.p', 'b.p'], ['b.p']]
    contents = path.read_text()
    rows = list(csv.DictReader(io.StringIO(contents))) if use_csv else [json.loads(line) for line in contents.splitlines()]
    assert len(rows) == 3
    assert int(rows[-1]['problems']) == 2
    assert float(rows[-1]['proved_seconds_total']) == 2.0
    assert cli.main(args) == 0
    assert path.read_text() == contents
    assert len(calls) == 2
    assert capsys.readouterr().out == ''


def test_evaluate_collection_resumes_proofs_and_skips(tmp_path, monkeypatch):
    from axiom_prediction import training, data
    from axiom_prediction.dataset import CollectedAxiomProblem
    monkeypatch.setattr(data, 'axiom_training_example_to_json', lambda ex: {'problem_path': ex.problem_path})
    monkeypatch.setattr(data, 'axiom_training_example_from_json', lambda value: SimpleNamespace(**value, labels=[1]))
    calls = []
    def interrupted(problems, **kwargs):
        calls.append(problems)
        yield CollectedAxiomProblem('a', SimpleNamespace(problem_path='a', labels=[1]), 'proved')
        yield CollectedAxiomProblem('b', None, 'unreadable', False)
        raise KeyboardInterrupt()
    monkeypatch.setattr(training, 'collect_problems_parallel', interrupted)
    journal = open_journal(tmp_path / 'results', command='evaluate')
    config = training.AxiomTrainingConfig(num_workers=1)
    with pytest.raises(KeyboardInterrupt):
        training.collect_examples(['a', 'b', 'c'], tptp_root=None, config=config, resume=journal)
    def finish(problems, **kwargs):
        calls.append(problems)
        yield CollectedAxiomProblem('c', SimpleNamespace(problem_path='c', labels=[1]), 'proved')
    monkeypatch.setattr(training, 'collect_problems_parallel', finish)
    examples, skipped = training.collect_examples(['a', 'b', 'c'], tptp_root=None, config=config, resume=journal)
    assert calls == [['a', 'b', 'c'], ['c']]
    assert [ex.problem_path for ex in examples] == ['a', 'c']
    assert skipped == [{'problem': 'b', 'outcome': 'unreadable'}]
    journal.close()


@pytest.mark.parametrize('adaptive', [False, True])
def test_evaluate_resumes_predictions_and_preserves_aggregate_metrics(tmp_path, monkeypatch, adaptive):
    from axiom_prediction import training, multiprocess
    examples = [SimpleNamespace(problem_path=str(i), labels=[i % 2], graph=i) for i in range(5)]
    model = SimpleNamespace(eval=lambda: None)
    journal = open_journal(tmp_path / 'results', command='evaluate')
    seen = []
    def interrupted(model, graphs):
        seen.extend(graphs)
        if 2 in graphs:
            raise KeyboardInterrupt()
        return [[0.8 if graph % 2 else 0.2] for graph in graphs]
    monkeypatch.setattr(multiprocess, 'predict_graphs', interrupted)
    with pytest.raises(KeyboardInterrupt):
        training.example_outputs(model, examples, batch_size=1, adaptive=adaptive, resume=journal)
    completed = {int(ex.problem_path) for ex in examples if journal.load('predictions', ex.problem_path)}
    assert completed
    seen.clear()
    def finish(model, graphs):
        seen.extend(graphs)
        return [[0.8 if graph % 2 else 0.2] for graph in graphs]
    monkeypatch.setattr(multiprocess, 'predict_graphs', finish)
    labels, probabilities, metrics = training.example_outputs(model, examples, batch_size=1, adaptive=adaptive, resume=journal)
    assert set(seen).isdisjoint(completed)
    assert set(seen) | completed == set(range(5))
    assert labels == [0, 1, 0, 1, 0]
    assert probabilities == [0.2, 0.8, 0.2, 0.8, 0.2]
    assert metrics == training.prediction_metrics(labels, probabilities, problem_sizes=[1]*5)
    journal.close()


@pytest.mark.parametrize('use_csv', [False, True])
def test_evaluate_cli_resumes_after_interrupted_collection(tmp_path, tiny_problem_path, monkeypatch, use_csv, capsys):
    from axiom_prediction import cli, training
    from axiom_prediction.model import AxiomModelConfig, AxiomPredictionNetwork, save_checkpoint
    checkpoint = tmp_path / 'model.pt'
    save_checkpoint(checkpoint, AxiomPredictionNetwork(AxiomModelConfig(hidden_dim=8, message_rounds=1)),
                    training_config={'sat_policy': 'satresetcop'})
    paths = []
    for index in range(2):
        path = tmp_path / f'problem{index}.p'
        path.write_text(tiny_problem_path.read_text())
        paths.append(str(path))
    real_collect = training.collect_problems_parallel
    completed = []
    def interrupted(problems, **kwargs):
        for result in real_collect(problems, **kwargs):
            completed.append(result.problem)
            yield result
            raise KeyboardInterrupt()
    monkeypatch.setattr(training, 'collect_problems_parallel', interrupted)
    output = tmp_path / 'results'
    args = ['evaluate', str(checkpoint), *paths, '--splits', '1', '--parts', '0',
            '--output', str(output), '--device', 'cpu', '--num-workers', '1', '--no-wandb', '--multiprocess']
    if use_csv:
        args.append('--csv')
    with pytest.raises(KeyboardInterrupt):
        cli.main(args)
    requested = []
    def finish(problems, **kwargs):
        requested.extend(problems)
        yield from real_collect(problems, **kwargs)
    monkeypatch.setattr(training, 'collect_problems_parallel', finish)
    assert cli.main(args) == 0
    assert len(completed) == len(requested) == 1
    assert set(completed).isdisjoint(requested)
    text = output.read_text()
    rows = list(csv.DictReader(io.StringIO(text))) if use_csv else [json.loads(line) for line in text.splitlines()]
    assert len([row for row in rows if row.get('event') == 'problem']) == 2
    assert int(rows[-1]['problems_proved']) == 2
    assert cli.main(args) == 0
    assert output.read_text() == text
    assert capsys.readouterr().out == ''


@pytest.mark.parametrize('existing_csv', [False, True])
def test_format_mismatch_fails_without_changing_existing_file(tmp_path, existing_csv):
    path = tmp_path / 'results'
    first = open_journal(path, use_csv=existing_csv)
    first.write({'problem': 'a.p', 'proved': True, 'seconds': 1.0})
    first.close()
    original = path.read_bytes()
    with pytest.raises(ValueError, match='JSONL.*CSV|CSV.*JSONL'):
        open_journal(path, use_csv=not existing_csv)
    assert path.read_bytes() == original
