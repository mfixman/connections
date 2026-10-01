import csv
import io
import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from axiom_prediction.output import COMMAND_FIELDS, output_format, write_record
from axiom_prediction.output_resume import OutputJournal

def open_journal(path, command = "run", use_csv = False, identity = None):
    return OutputJournal(
        path,
        command,
        use_csv,
        list(dict.fromkeys(COMMAND_FIELDS[command])),
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
        yield {"problem": "a.p", "proved": True, "seconds": 2.0, "outcome": "proved"}
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
        yield {"problem": "b.p", "proved": False, "seconds": 3.0, "outcome": "Timeout"}

    monkeypatch.setattr(run, "run_problems", finish)
    assert cli.main(args) == 0
    assert calls == [["a.p", "b.p"], ["b.p"]]
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

def test_evaluate_collection_resumes_proofs_and_skips(tmp_path, monkeypatch):
    from axiom_prediction import training, data
    from axiom_prediction.dataset import CollectedAxiomProblem

    monkeypatch.setattr(
        data,
        "axiom_training_example_to_json",
        lambda ex: {"problem_path": ex.problem_path},
    )

    monkeypatch.setattr(
        data,
        "axiom_training_example_from_json",
        lambda value: SimpleNamespace(**value, labels = [1]),
    )

    calls = []

    def interrupted(problems, **kwargs):
        calls.append(problems)
        yield CollectedAxiomProblem(
            "a",
            SimpleNamespace(problem_path = "a", labels = [1]),
            "proved",
        )

        yield CollectedAxiomProblem("b", None, "unreadable", False)
        raise KeyboardInterrupt()

    monkeypatch.setattr(training, "collect_problems_parallel", interrupted)
    journal = open_journal(tmp_path / "results", command = "evaluate")
    config = training.AxiomTrainingConfig(num_workers = 1)
    with pytest.raises(KeyboardInterrupt):
        training.collect_examples(
            ["a", "b", "c"],
            tptp_root = None,
            config = config,
            resume = journal,
        )

    def finish(problems, **kwargs):
        calls.append(problems)
        yield CollectedAxiomProblem(
            "c",
            SimpleNamespace(problem_path = "c", labels = [1]),
            "proved",
        )

    monkeypatch.setattr(training, "collect_problems_parallel", finish)
    examples, skipped = training.collect_examples(
        ["a", "b", "c"],
        tptp_root = None,
        config = config,
        resume = journal,
    )

    assert calls == [["a", "b", "c"], ["c"]]
    assert [ex.problem_path for ex in examples] == ["a", "c"]
    assert skipped == [{"problem": "b", "outcome": "unreadable"}]
    journal.close()

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

@pytest.mark.parametrize("use_csv", [False, True])
def test_evaluate_cli_resumes_after_interrupted_collection(
    tmp_path,
    tiny_problem_path,
    monkeypatch,
    use_csv,
    capsys,
):
    from axiom_prediction import cli, training
    from axiom_prediction.model import AxiomModelConfig, AxiomPredictionNetwork, save_checkpoint

    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        AxiomPredictionNetwork(AxiomModelConfig(hidden_dim = 8, message_rounds = 1)),
        training_config = {"sat_policy": "satresetcop"},
    )

    paths = []
    for index in range(2):
        path = tmp_path / f"problem{index}.p"
        path.write_text(tiny_problem_path.read_text())
        paths.append(str(path))

    real_collect = training.collect_problems_parallel
    completed = []

    def interrupted(problems, **kwargs):
        for result in real_collect(problems, **kwargs):
            completed.append(result.problem)
            yield result
            raise KeyboardInterrupt()

    monkeypatch.setattr(training, "collect_problems_parallel", interrupted)
    output = tmp_path / "results"
    args = ["evaluate", str(checkpoint), *paths, "--splits", "1", "--parts", "0"]
    args += ["--output", str(output), "--device", "cpu", "--num-workers", "1"]
    args += ["--no-wandb", "--multiprocess"]

    if use_csv:
        args.append("--csv")

    with pytest.raises(KeyboardInterrupt):
        cli.main(args)

    requested = []

    def finish(problems, **kwargs):
        requested.extend(problems)
        yield from real_collect(problems, **kwargs)

    monkeypatch.setattr(training, "collect_problems_parallel", finish)
    assert cli.main(args) == 0
    assert len(completed) == len(requested) == 1
    assert set(completed).isdisjoint(requested)
    text = output.read_text()
    rows = (
        list(csv.DictReader(io.StringIO(text)))
        if use_csv
        else [json.loads(line) for line in text.splitlines()]
    )

    assert len([row for row in rows if row.get("event") == "problem"]) == 2
    assert all(row.get("event") != "summary" for row in rows)
    completion = json.loads(Path(str(output) + ".resume/completion.json").read_text())
    assert completion["metrics"]["problems_proved"] == 2
    assert cli.main(args) == 0
    assert output.read_text() == text
    assert capsys.readouterr().out == ""

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

def test_partial_csv_header_can_resume(tmp_path):
    path = tmp_path / "results"
    path.write_text("problem,part,tptp_sta")
    journal = open_journal(path, use_csv = True)
    journal.write({"problem": "a.p", "proved": True})
    journal.close()
    with path.open() as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 1
    assert rows[0]["problem"] == "a"

@pytest.mark.parametrize("use_csv", [False, True])
def test_journal_recovers_torn_utf8(tmp_path, use_csv):
    path = tmp_path / "results"
    journal = open_journal(path, use_csv = use_csv)
    journal.write({"problem": "a.p", "proved": True})
    journal.close()
    committed = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'"unfinished\xe2\x82')

    journal = open_journal(path, use_csv = use_csv)
    journal.close()
    assert path.read_bytes() == committed

def test_csv_preserves_unicode_separators_and_boolean_errors(tmp_path):
    path = tmp_path / "results"
    journal = open_journal(path, use_csv = True)
    row = {"problem": "a\u2028b.p", "error": False, "proved": False}
    journal.write(row)
    journal.close()

    journal = open_journal(path, use_csv = True)
    assert journal.records[0]["problem"] == row["problem"]
    assert journal.records[0]["error"] is False
    assert journal.exit_code == 0
    journal.close()

@pytest.mark.parametrize("use_csv", [False, True])
def test_completed_run_preserves_error_status_and_allows_tracking_change(
    tmp_path,
    monkeypatch,
    use_csv,
):
    from axiom_prediction import cli, run

    path = tmp_path / "results"
    monkeypatch.setattr(cli, "selected_problems", lambda args: ["a.p"])
    result = {"problem": "a.p", "proved": False, "error": True, "outcome": "WorkerExited"}
    monkeypatch.setattr(run, "run_problems", lambda *args, **kwargs: iter([result]))
    args = ["run", "--output", str(path), "--device", "cpu"]
    if use_csv:
        args.append("--csv")

    assert cli.main([*args, "--no-wandb"]) == 2
    committed = path.read_bytes()
    assert cli.main([*args, "--wandb"]) == 2
    assert path.read_bytes() == committed

def test_older_resume_metadata_allows_tracking_change(tmp_path):
    path = tmp_path / "results"
    journal = open_journal(path)
    metadata = journal.directory / "config.json"
    journal.close()
    metadata.write_text(json.dumps({"command": "run", "wandb": True}))

    journal = open_journal(path, identity = {"command": "run", "wandb": False})
    journal.close()

@pytest.mark.parametrize("use_csv", [False, True])
def test_completion_marker_rejects_truncated_output(tmp_path, use_csv):
    path = tmp_path / "results"
    journal = open_journal(path, use_csv = use_csv)
    journal.write({"problem": "a.p", "proved": True})
    journal.metrics = {"problems": 1}
    journal.finish(0)
    assert journal.complete
    journal.close()
    path.write_text("")

    journal = open_journal(path, use_csv = use_csv)
    assert not journal.complete
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

    assert [row["problem"] for row in rows] == ["SYN001-1", "SYN001-1"]
    journal = open_journal(path, use_csv = use_csv)
    assert [row["problem"] for row in journal.records] == problems
    for problem in problems:
        journal.write({"problem": problem, "proved": True})

    journal.close()
    assert path.read_text() == contents

@pytest.mark.parametrize("use_csv", [False, True])
@pytest.mark.parametrize("guided", [False, True])
def test_resumed_timeouts_keep_run_metadata(tmp_path, monkeypatch, use_csv, guided):
    from axiom_prediction import cli, run

    path = tmp_path / "results"
    monkeypatch.setattr(cli, "selected_problems", lambda args: ["a.p", "b.p"])
    requested = []

    def interrupted(function, arguments, **kwargs):
        requested.append([args[0] for args in arguments])
        yield {"problem": "a.p", "outcome": "Timeout", "proved": False, "seconds": 1.0}
        raise KeyboardInterrupt()

    monkeypatch.setattr(run, "supervised_results", interrupted)
    args = ["run", "--output", str(path), "--seed", "42", "--no-wandb", "--device", "cpu"]
    if guided:
        from axiom_prediction.model import AxiomPredictor

        predictor = SimpleNamespace(training_config = {"sat_policy": "satcop"})
        monkeypatch.setattr(AxiomPredictor, "load", lambda *args, **kwargs: predictor)
        args.extend(["--model", str(tmp_path / "model.pt")])

    if use_csv:
        args.append("--csv")

    with pytest.raises(KeyboardInterrupt):
        cli.main(args)

    def finish(function, arguments, **kwargs):
        requested.append([args[0] for args in arguments])
        yield {"problem": "b.p", "outcome": "Timeout", "proved": False, "seconds": 1.0}

    monkeypatch.setattr(run, "supervised_results", finish)
    assert cli.main(args) == 0
    assert requested == [["a.p", "b.p"], ["b.p"]]
    text = path.read_text()
    rows = (
        list(csv.DictReader(io.StringIO(text)))
        if use_csv
        else [json.loads(line) for line in text.splitlines()]
    )

    assert len(rows) == 2
    assert all(int(row["seed"]) == 42 for row in rows)
    assert all(row["mode"] == ("weighted" if guided else "base") for row in rows)
    assert all(row["policy"] == ("satcop" if guided else "satresetcop") for row in rows)
    assert cli.main(args) == 0
    assert path.read_text() == text

@pytest.mark.parametrize("use_csv", [False, True])
def test_declared_status_survives_timeout_and_resume(tmp_path, use_csv):
    problem = tmp_path / "SYN001-1.p"
    problem.write_text("% Status : Theorem\nfof(a, axiom, p).\n")
    path = tmp_path / "results"
    row = {"problem": str(problem), "outcome": "Timeout", "proved": False}
    journal = open_journal(path, use_csv = use_csv)
    journal.write(row)
    journal.close()
    contents = path.read_text()
    records = (
        list(csv.DictReader(io.StringIO(contents)))
        if use_csv
        else [json.loads(line) for line in contents.splitlines()]
    )

    assert records[0]["tptp_status"] == "Theorem"
    assert "outcome" not in records[0]
    journal = open_journal(path, use_csv = use_csv)
    assert journal.records[0]["outcome"] == "Timeout"
    journal.write(row)
    journal.close()
    assert path.read_text() == contents
