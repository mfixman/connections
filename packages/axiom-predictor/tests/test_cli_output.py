import csv
import io
import json

import pytest

from axiom_prediction.cli import main
from axiom_prediction.output import output_format, write_record
from axiom_prediction.split import ProblemSplit

@pytest.mark.parametrize("use_csv", [False, True])
def test_all_commands_emit_parseable_records(tmp_path, tiny_problem_path, capsys, use_csv):
    dataset = tmp_path / "dataset"
    trained = tmp_path / "trained"
    flag = ["--csv"] if use_csv else []
    common = ["--num-workers", "1", "--no-wandb", "--device", "cpu"]
    commands = [
        ["collect", str(tiny_problem_path), "--dataset", str(dataset), "--num-workers", "1"],
        ["train", "--dataset", str(dataset), "--data-dir", str(trained), "--epochs", "1", "--network", "SmallFull", *common],
        ["evaluate", str(trained / "model"), "--dataset", str(dataset), *common],
        ["evaluate", "--model", str(trained / "model"), "--dataset", str(dataset), *common],
        ["evaluate", "--model", str(trained / "model"), "--dataset", str(dataset), "--multiprocess", *common],
        ["evaluate", str(trained / "model"), str(tiny_problem_path), "--multiprocess", *common],
        ["evaluate", "--model", str(trained / "model"), str(tiny_problem_path), *common],
        ["run", str(tiny_problem_path), "--model", str(trained / "model"), "--multiprocess", *common],
        ["predict", str(trained / "model"), str(tiny_problem_path), "--device", "cpu"],
        ["run", str(tiny_problem_path), "--num-workers", "1", "--no-wandb"],
    ]

    for command in commands:
        if command[0] in ("evaluate", "run"):
            part = ProblemSplit(10, (8, 9)).part(tiny_problem_path)
            command.extend(["--split", "10", "--parts", str(part)])

        assert main([*flag, *command]) == 0
        stdout = capsys.readouterr().out
        if use_csv:
            records = list(csv.DictReader(io.StringIO(stdout)))
            assert all(None not in record for record in records)
        else:
            records = [json.loads(line) for line in stdout.splitlines()]

        assert records
        assert all(record.get("event") != "summary" for record in records)
        if command[0] == "run":
            assert all("event" not in record for record in records)
        if command[0] == "predict":
            assert 0 <= float(records[0]["probability"]) <= 1
        if command[0] in ("evaluate", "run"):
            problems = [row for row in records if row.get("problem")]
            assert len(problems) == 1
            assert int(problems[0]["part"]) == part

def test_csv_preserves_nested_values_and_quoting(capsys):
    record = {"event": "dataset", "skipped": [{"problem": "a,b\n\"c\"", "outcome": "Timeout"}], "dataset": None}
    with output_format("evaluate", True):
        write_record(record)

    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert json.loads(rows[0]["skipped"]) == record["skipped"]
    assert rows[0]["dataset"] == "null"
    assert rows[0]["roc_auc"] == ""
    write_record(record)
    assert json.loads(capsys.readouterr().out) == record
