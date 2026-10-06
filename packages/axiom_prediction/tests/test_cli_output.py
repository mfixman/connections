import csv
import io
import json

import pytest

from axiom_prediction.cli import main
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
            part = ProblemSplit(10, [8, 9]).part(tiny_problem_path)
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
            assert all(record["error"] == ("false" if use_csv else False) for record in records)
            assert all("kept_axioms" not in record for record in records)
            assert all("guidance_fallback" not in record for record in records)
            assert all("parseable" not in record for record in records)
        if command[0] == "predict":
            assert 0 <= float(records[0]["probability"]) <= 1
        if command[0] in ("evaluate", "run"):
            problems = [row for row in records if row.get("problem")]
            assert len(problems) == 1
            assert problems[0]["problem"] == tiny_problem_path.name
            assert "outcome" not in problems[0]
            assert problems[0]["tptp_status"] == ""
            assert int(problems[0]["part"]) == part

@pytest.mark.parametrize("use_csv", [False, True])
def test_parse_failure_is_an_error_with_stdout_diagnostic(tmp_path, capfd, use_csv):
    problem = tmp_path / "broken.p"
    problem.write_text("fof(broken, axiom, ).\n")
    output = tmp_path / "results"
    flags = ["--csv"] if use_csv else []
    args = [
        "run", str(problem), "--output", str(output), "--num-workers", "1",
        "--no-wandb", *flags,
    ]

    assert main(args) == 2

    captured = capfd.readouterr()
    assert str(problem) in captured.out
    assert "TPTPParseError:" in captured.out
    text = output.read_text()
    if use_csv:
        rows = list(csv.DictReader(io.StringIO(text)))
    else:
        rows = list(map(json.loads, text.splitlines()))

    assert len(rows) == 1
    assert rows[0]["error"] == ("true" if use_csv else True)
    assert rows[0]["proved"] == ("false" if use_csv else False)
    assert "parseable" not in rows[0]
