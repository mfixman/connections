import csv
import io

from axiom_prediction.cli import main
from axiom_prediction.split import ProblemSplit

def test_all_commands_emit_parseable_records(tmp_path, tiny_problem_path, capsys):
    dataset = tmp_path / "dataset"
    trained = tmp_path / "trained"
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

        assert main(command) == 0
        stdout = capsys.readouterr().out
        records = list(csv.DictReader(io.StringIO(stdout)))
        assert all(None not in record for record in records)

        assert records
        assert all(record.get("event") != "summary" for record in records)
        model_records = [
            record
            for record in records
            if command[0] in ("run", "predict")
            or record.get("event") in ("model", "problem")
        ]

        # SmallFull's scorer is Linear(160, 32), ReLU, Linear(32, 1).
        expected_size = 0 if command[0] == "run" and "--model" not in command else 5185
        assert all(int(record["network_size"]) == expected_size for record in model_records)
        if command[0] == "run":
            assert all("event" not in record for record in records)
            assert all(record["error"] == "false" for record in records)
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

def test_parse_failure_is_an_error_with_stdout_diagnostic(tmp_path, capfd):
    problem = tmp_path / "broken.p"
    problem.write_text("fof(broken, axiom, ).\n")
    output = tmp_path / "results"
    args = [
        "run", str(problem), "--output", str(output), "--num-workers", "1",
        "--no-wandb",
    ]

    assert main(args) == 2

    captured = capfd.readouterr()
    assert str(problem) in captured.out
    assert "TPTPParseError:" in captured.out
    text = output.read_text()
    rows = list(csv.DictReader(io.StringIO(text)))

    assert len(rows) == 1
    assert rows[0]["error"] == "true"
    assert rows[0]["proved"] == "false"
    assert int(rows[0]["network_size"]) == 0
    assert "parseable" not in rows[0]
