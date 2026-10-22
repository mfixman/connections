import json
from pathlib import Path

import pytest

from axiom_prediction import cli
from axiom_prediction.argument_overrides import parse_arguments
from axiom_prediction.choices import ProverPolicy
from axiom_prediction.output import report_metrics, write_record

@pytest.fixture
def override_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path / "argument_override.json"

def parse(argv):
    return parse_arguments(cli.build_parser(), argv)

def test_missing_file_preserves_arguments(override_file):
    args = parse(["run", "input.p", "--split", "10"])
    assert args.split == 10
    assert args.split_scheme == 5
    assert args.problems == ["input.p"]

@pytest.mark.parametrize("command", ["run", "collect", "train", "evaluate", "predict"])
def test_overrides_all_commands(override_file, command):
    override_file.write_text(json.dumps({"split-scheme": 3, "split": 5}))
    argv = [command, "--split", "10"]
    if command == "train":
        argv.extend(["--data-dir", "training"])

    args = parse(argv)
    assert args.command == command
    assert args.split == 5
    assert args.split_scheme == 3

def test_types_aliases_arrays_and_flags(override_file):
    override_file.write_text(json.dumps({
        "splits": 5, "part": [1, 3], "timeout-seconds": 2.5,
        "model": "model.ts", "policy": "satcop", "cpu": True,
        "multiprocess": False, "no-wandb": True,
    }))
    args = parse([
        "run", "input.p", "--split", "10", "--parts", "0", "2",
        "--sine", "--device", "cuda", "--multiprocess", "--wandb",
    ])
    assert args.split == 5
    assert args.parts == [1, 3]
    assert args.timeout_s == 2.5
    assert args.model == Path("model.ts")
    assert args.policy == ProverPolicy.SatCoP
    assert args.device == "cpu"
    assert not args.sine and not args.multiprocess and not args.wandb
    assert args.problems == ["input.p"]

def test_required_argument_from_json(override_file):
    override_file.write_text(json.dumps({"data-dir": "training", "evaluate": [8, 9]}))
    args = parse(["train"])
    assert args.data_dir == Path("training")
    assert args.evaluate == [8, 9]

@pytest.mark.parametrize("overrides, argv, attribute, expected", [
    ({"sine": False}, ["--model", "model.ts"], "model", Path("model.ts")),
    ({"model": None}, ["--sine"], "sine", True),
    ({"cpu": False}, ["--cpu"], "device", "cuda"),
    ({"device": "cpu"}, ["--cpu"], "device", "cpu"),
    ({"no-wandb": False}, ["--no-wandb"], "wandb", True),
    ({"wandb": False}, [], "wandb", False),
    ({"model": None}, ["--model", "model.ts"], "model", None),
])
def test_clear_and_replace_options(override_file, overrides, argv, attribute, expected):
    override_file.write_text(json.dumps(overrides))
    assert getattr(parse(["run", *argv]), attribute) == expected

@pytest.mark.parametrize("text", [
    "{broken", "[]", '{"split": null}', '{"unknown": 1}', '{"split": 0}', '{"split-scheme": 2}',
    '{"parts": 1}', '{"parts": []}', '{"split": true}', '{"split": 1.5}',
    '{"multiprocess": "false"}', '{"policy": "unknown"}', '{"split": {}}',
    '{"split": 3, "splits": 4}', '{"model": "model.ts", "sine": true}',
])
def test_invalid_overrides_fail_before_execution(override_file, monkeypatch, text, capsys):
    override_file.write_text(text)
    monkeypatch.setattr(cli, "execute_command", lambda args: pytest.fail("must not execute"))
    with pytest.raises(SystemExit) as error:
        cli.main(["run"])

    assert error.value.code == 2
    assert "argument_override.json" in capsys.readouterr().err

@pytest.mark.parametrize("name", [None, "trial"])
def test_main_uses_overrides_for_execution_and_resume(override_file, monkeypatch, name):
    if name is not None:
        override_file = override_file.with_name(f"argument_override_{name}.json")

    override_file.write_text(json.dumps({"split-scheme": 3, "split": 5}))
    calls = []
    def execute(args):
        calls.append(cli.selected_split(args).to_dict())
        write_record({"problem": "input.p", "error": False})
        report_metrics({"problems": 1})
        return 0

    monkeypatch.setattr(cli, "execute_command", execute)
    args = ["run", "--split", "10", "--output", "result.csv"]
    if name is not None:
        args.extend(["--run-name", name])

    assert cli.main(args) == 0
    assert calls == [{"split": 5, "parts": list(range(5)), "scheme": 3}]
    config = json.loads(Path("result.csv.resume/config.json").read_text())
    assert config["split"] == 5 and config["split_scheme"] == 3
    assert cli.main(args) == 0
    assert len(calls) == 1
    override_file.write_text(json.dumps({"split": 4}))
    assert cli.main(args) == 2
    assert len(calls) == 1

def test_help_works_with_invalid_override_file(override_file):
    override_file.write_text("invalid json")
    with pytest.raises(SystemExit) as error:
        parse(["run", "--help"])

    assert error.value.code == 0

@pytest.mark.parametrize("command", ["run", "train", "evaluate"])
@pytest.mark.parametrize("name_args", [["--run-name", "trial"], ["--run-name=trial"]])
def test_named_file_without_global_file(override_file, command, name_args):
    named = override_file.with_name("argument_override_trial.json")
    named.write_text(json.dumps({"split-scheme": 3, "split": 5}))
    argv = [command, *name_args, "--split", "10"]
    if command == "train":
        argv.extend(["--data-dir", "training"])

    args = parse(argv)
    assert args.run_name == "trial"
    assert args.split_scheme == 3 and args.split == 5
    assert not override_file.exists()

def test_named_file_overrides_global_aliases_and_conflicting_flags(override_file):
    override_file.write_text(json.dumps({"split": 10, "split-scheme": 3, "cpu": True, "sine": True}))
    override_file.with_name("argument_override_trial.json").write_text(json.dumps({
        "splits": 5, "device": "auto", "model": "model.ts",
    }))
    args = parse(["run", "--run-name", "trial", "--split", "20"])
    assert args.split == 5 and args.split_scheme == 3
    assert args.device == "auto"
    assert args.model == Path("model.ts") and not args.sine

@pytest.mark.parametrize("name_args", [[], ["--run-name", "other"], ["--", "--run-name", "trial"]])
def test_nonmatching_named_file_is_ignored(override_file, name_args):
    override_file.with_name("argument_override_trial.json").write_text("invalid json")
    assert parse(["run", *name_args]).split == 1

def test_name_selection_uses_last_cli_name_not_override_name(override_file):
    override_file.write_text(json.dumps({"run-name": "renamed"}))
    override_file.with_name("argument_override_first.json").write_text("invalid json")
    override_file.with_name("argument_override_renamed.json").write_text("invalid json")
    override_file.with_name("argument_override_trial.json").write_text('{"split": 5}')
    args = parse(["run", "--run-name", "first", "--run-name", "trial"])
    assert args.split == 5 and args.run_name == "renamed"

@pytest.mark.parametrize("text", ["invalid json", "[]", '{"split": 0}'])
def test_invalid_named_file_reports_its_path(override_file, text, capsys):
    override_file.with_name("argument_override_trial.json").write_text(text)
    with pytest.raises(SystemExit) as error:
        parse(["run", "--run-name", "trial"])

    assert error.value.code == 2
    assert "argument_override_trial.json" in capsys.readouterr().err
