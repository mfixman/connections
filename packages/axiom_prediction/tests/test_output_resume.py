import csv
import io
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from axiom_prediction.output import output_fields, output_format, write_record
from axiom_prediction.output_resume import OutputJournal

def open_journal(path, command = "run", identity = None):
    return OutputJournal(
        path,
        command,
        output_fields(command),
        identity or {"command": command},
    )

def test_journal_repairs_torn_tail_and_preserves_multiline(tmp_path, capsys):
    path = tmp_path / "results"
    row = {"problem": 'a,b\n"c"', "proved": False, "seconds": 1.25, "outcome": "Timeout"}
    with output_format("run", path, {"command": "run"}):
        write_record(row)

    committed = path.read_bytes()
    with path.open("a") as f:
        f.write('"unfinished\nquoted')

    journal = open_journal(path)
    assert journal.records[0]["problem"] == row["problem"]
    assert journal.records[0]["proved"] is False
    assert journal.records[0]["seconds"] == 1.25
    assert journal.records[0]["outcome"] == "Timeout"
    journal.write(row)
    journal.close()

    assert path.read_bytes() == committed
    assert capsys.readouterr().out == ""

def test_journal_refuses_mixed_configuration_and_concurrent_writers(tmp_path):
    path = tmp_path / "results"
    first = open_journal(path)
    with pytest.raises(BlockingIOError):
        open_journal(path)

    first.close()
    with pytest.raises(ValueError, match = "different command"):
        open_journal(path, identity = {"command": "different"})

def test_run_cli_resumes_only_unfinished_and_keeps_summary_totals(
    tmp_path,
    monkeypatch,
    capsys,
):
    from axiom_prediction import cli, run

    path = tmp_path / "results"
    monkeypatch.setattr(cli, "selected_problems", lambda args: ["a.p", "b.p"])
    calls = []

    def interrupted(problems, **kwargs):
        calls.append(problems)
        yield {"problem": "b.p", "proved": True, "seconds": 2.0, "outcome": "proved"}
        raise KeyboardInterrupt()

    monkeypatch.setattr(run, "run_problems", interrupted)
    args = ["run", "--output", str(path), "--no-wandb", "--device", "cpu"]

    with pytest.raises(KeyboardInterrupt):
        cli.main(args)

    assert not Path(str(path) + ".resume/completion.json").exists()

    def finish(problems, **kwargs):
        calls.append(problems)
        yield {"problem": "a.p", "proved": False, "seconds": 3.0, "outcome": "Timeout"}

    monkeypatch.setattr(run, "run_problems", finish)
    assert cli.main(args) == 0
    assert calls == [["a.p", "b.p"], ["a.p"]]
    contents = path.read_text()
    rows = list(csv.DictReader(io.StringIO(contents)))

    assert len(rows) == 2
    assert all("event" not in row for row in rows)
    completion = json.loads(Path(str(path) + ".resume/completion.json").read_text())
    assert completion["metrics"]["problems"] == 2
    assert completion["metrics"]["proved_seconds_total"] == 2.0

    assert cli.main(args) == 0
    assert path.read_text() == contents
    assert len(calls) == 2
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(cli, "split_scheme", cli.split_scheme + 1)
    assert cli.main(args) == 2
    assert path.read_text() == contents
    assert len(calls) == 2

@pytest.mark.parametrize("adaptive", [False, True])
def test_evaluate_resumes_predictions_and_preserves_aggregate_metrics(
    tmp_path, monkeypatch, adaptive
):
    from axiom_prediction import training, multiprocess

    examples = [
        SimpleNamespace(problem_path = str(i), labels = [i % 2], graph = i) for i in range(5)
    ]

    model = SimpleNamespace(eval = lambda: None)
    journal = open_journal(tmp_path / "results", command = "evaluate")
    seen = []

    def interrupted(model, graphs):
        seen.extend(graphs)
        if 2 in graphs:
            raise KeyboardInterrupt()

        return [[0.8 if graph % 2 else 0.2] for graph in graphs]

    monkeypatch.setattr(multiprocess, "predict_graphs", interrupted)
    with pytest.raises(KeyboardInterrupt):
        training.example_outputs(
            model,
            examples,
            batch_size = 1,
            adaptive = adaptive,
            resume = journal,
        )

    completed = {
        int(ex.problem_path)
        for ex in examples
        if journal.load("predictions", ex.problem_path)
    }

    assert completed
    seen.clear()

    def finish(model, graphs):
        seen.extend(graphs)
        return [[0.8 if graph % 2 else 0.2] for graph in graphs]

    monkeypatch.setattr(multiprocess, "predict_graphs", finish)
    labels, probabilities, metrics = training.example_outputs(
        model,
        examples,
        batch_size = 1,
        adaptive = adaptive,
        resume = journal,
    )

    assert set(seen).isdisjoint(completed)
    assert set(seen) | completed == set(range(5))
    assert labels == [0, 1, 0, 1, 0]
    assert probabilities == [0.2, 0.8, 0.2, 0.8, 0.2]
    assert metrics == training.prediction_metrics(
        labels,
        probabilities,
        problem_sizes = [1] * 5,
    )

    journal.close()

def test_non_csv_output_is_rejected_without_changes(tmp_path):
    path = tmp_path / "results"
    original = '{"problem": "a.p", "proved": true}\n'
    path.write_text(original)
    with pytest.raises(ValueError, match = "CSV header"):
        open_journal(path)

    assert path.read_text() == original

def test_completion_marker_rejects_truncated_output(tmp_path):
    path = tmp_path / "results"
    journal = open_journal(path)
    journal.write({"problem": "a.p", "proved": True})
    journal.metrics = {"problems": 1}
    journal.finish(0)
    assert journal.complete()
    journal.close()
    path.write_text("")

    journal = open_journal(path)
    assert not journal.complete()
    journal.close()

def test_problem_names_preserve_distinct_paths_for_resume(tmp_path):
    path = tmp_path / "results"
    problems = ["/first/SYN001-1.p", "/second/SYN001-1.p"]
    journal = open_journal(path)
    for problem in problems:
        journal.write({"problem": problem, "proved": True})

    journal.close()
    contents = path.read_text()
    rows = list(csv.DictReader(io.StringIO(contents)))

    assert [row["problem"] for row in rows] == ["SYN001-1.p", "SYN001-1.p"]
    journal = open_journal(path)
    assert [row["problem"] for row in journal.records] == problems
    for problem in problems:
        journal.write({"problem": problem, "proved": True})

    journal.close()
    assert path.read_text() == contents

def test_run_resume_rejects_changed_model(tmp_path):
    path = tmp_path / "results"
    identity = {"command": "run", "model": "epoch-0001.pt", "policy": "SatResetCoP"}
    first = open_journal(path, identity = identity)
    first.close()
    with pytest.raises(ValueError, match = "different command/configuration"):
        open_journal(path, identity = {**identity, "model": "epoch-0002.pt"})

@pytest.mark.parametrize("errors, expected", [([True, False], 0), ([True, True], 2), ([False], 0)])
def test_run_exit_status_includes_saved_results(tmp_path, errors, expected):
    path = tmp_path / "results.csv"
    journal = open_journal(path)
    for index, error in enumerate(errors):
        journal.write({"problem": f"{index}.p", "error": error, "proved": False})

    assert journal.exit_code() == expected
    # Older completion markers treated any input error as a batch failure.
    journal.finish(2)
    journal.close()
    journal = open_journal(path)
    assert journal.complete()
    assert journal.exit_code() == expected
    journal.close()
