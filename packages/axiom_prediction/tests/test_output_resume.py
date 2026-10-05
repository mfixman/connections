import csv
import io
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from axiom_prediction.output import output_fields, output_format, write_record
from axiom_prediction.output_resume import OutputJournal

def open_journal(path, command = "run", use_csv = False, identity = None):
    return OutputJournal(
        path,
        command,
        use_csv,
        output_fields(command),
        identity or {"command": command},
    )

@pytest.mark.parametrize("use_csv", [False, True])
def test_journal_repairs_torn_tail_and_preserves_multiline(tmp_path, use_csv, capsys):
    path = tmp_path / "results"
    row = {"problem": 'a,b\n"c"', "proved": False, "seconds": 1.25, "outcome": "Timeout"}
    with output_format("run", use_csv, path, {"command": "run"}):
        write_record(row)

    committed = path.read_bytes()
    with path.open("a") as f:
        f.write('"unfinished\nquoted' if use_csv else '{"problem":')

    journal = open_journal(path, use_csv = use_csv)
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

@pytest.mark.parametrize("use_csv", [False, True])
def test_run_cli_resumes_only_unfinished_and_keeps_summary_totals(
    tmp_path,
    monkeypatch,
    use_csv,
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
    args = ["run", "--output", str(path), "--no-wandb", "--device", "cpu"] + (
        ["--csv"] if use_csv else []
    )

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
    rows = (
        list(csv.DictReader(io.StringIO(contents)))
        if use_csv
        else [json.loads(line) for line in contents.splitlines()]
    )

    assert len(rows) == 2
    assert all("event" not in row for row in rows)
    completion = json.loads(Path(str(path) + ".resume/completion.json").read_text())
    assert completion["metrics"]["problems"] == 2
    assert completion["metrics"]["proved_seconds_total"] == 2.0

    assert cli.main(args) == 0
    assert path.read_text() == contents
    assert len(calls) == 2
    assert capsys.readouterr().out == ""

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

@pytest.mark.parametrize("existing_csv", [False, True])
def test_format_mismatch_fails_without_changing_existing_file(
    tmp_path, existing_csv, capsys
):
    path = tmp_path / "results"
    first = open_journal(path, use_csv = existing_csv)
    first.write({"problem": "a.p", "proved": True, "seconds": 1.0})
    first.close()
    original = path.read_bytes()
    with pytest.raises(ValueError, match = "JSONL.*CSV|CSV.*JSONL"):
        open_journal(path, use_csv = not existing_csv)

    assert path.read_bytes() == original

    from axiom_prediction.cli import main

    args = ["run", "--output", str(path)] + ([] if existing_csv else ["--csv"])
    assert main(args) == 2
    assert path.read_bytes() == original
    error = capsys.readouterr().err
    assert "JSONL" in error and "CSV" in error

@pytest.mark.parametrize("use_csv", [False, True])
def test_completion_marker_rejects_truncated_output(tmp_path, use_csv):
    path = tmp_path / "results"
    journal = open_journal(path, use_csv = use_csv)
    journal.write({"problem": "a.p", "proved": True})
    journal.metrics = {"problems": 1}
    journal.finish(0)
    assert journal.complete()
    journal.close()
    path.write_text("")

    journal = open_journal(path, use_csv = use_csv)
    assert not journal.complete()
    journal.close()

@pytest.mark.parametrize("use_csv", [False, True])
def test_problem_names_preserve_distinct_paths_for_resume(tmp_path, use_csv):
    path = tmp_path / "results"
    problems = ["/first/SYN001-1.p", "/second/SYN001-1.p"]
    journal = open_journal(path, use_csv = use_csv)
    for problem in problems:
        journal.write({"problem": problem, "proved": True})

    journal.close()
    contents = path.read_text()
    rows = (
        list(csv.DictReader(io.StringIO(contents)))
        if use_csv
        else [json.loads(line) for line in contents.splitlines()]
    )

    assert [row["problem"] for row in rows] == ["SYN001-1.p", "SYN001-1.p"]
    journal = open_journal(path, use_csv = use_csv)
    assert [row["problem"] for row in journal.records] == problems
    for problem in problems:
        journal.write({"problem": problem, "proved": True})

    journal.close()
    assert path.read_text() == contents

@pytest.mark.parametrize("use_csv", [False, True])
def test_run_resume_rejects_changed_model(tmp_path, use_csv):
    path = tmp_path / "results"
    identity = {"command": "run", "model": "epoch-0001.pt", "policy": "SatResetCoP"}
    first = open_journal(path, use_csv = use_csv, identity = identity)
    first.close()
    with pytest.raises(ValueError, match = "different command/configuration"):
        open_journal(path, use_csv = use_csv, identity = {**identity, "model": "epoch-0002.pt"})
