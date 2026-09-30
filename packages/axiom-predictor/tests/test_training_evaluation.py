import json
import sys
from dataclasses import replace

import pytest

from axiom_prediction import training
from axiom_prediction.cli import build_parser, main
from axiom_prediction.model import save_checkpoint
from axiom_prediction.models import load_model_class
from axiom_prediction.split import ProblemSplit
from axiom_prediction.tptp import collect_proof_example
from axiom_prediction.wandb_tracking import WandbConfig
from test_axiom_prediction_wandb import _FakeRun, fake_wandb

pytestmark = pytest.mark.training

@pytest.mark.parametrize("network", ["DefaultFull", "DefaultNoTerms"])
@pytest.mark.parametrize("run_name", [None, "my-evaluation-job"])
def test_evaluation_run_name_uses_checkpoint_network(tmp_path, tiny_problem_path, monkeypatch, network, run_name):
    checkpoint = tmp_path / "renamed-checkpoint.pt"
    save_checkpoint(checkpoint, load_model_class(network)(), training_config = {})
    run = _FakeRun()
    init = {}
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb(run, init))
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    training.evaluate_axiom_predictor(
        checkpoint,
        [str(tiny_problem_path)],
        device = "cpu",
        config = training.AxiomTrainingConfig(device = "cpu", num_workers = 1),
        wandb_config = WandbConfig(name = run_name, name_prefix = "smol-custom", group = "custom"),
    )

    assert init["name"] == (run_name or f"{network}-evaluate")
    assert init["job_type"] == "evaluate"
    assert init["group"] == "custom"
    assert run.finished == [0]

@pytest.mark.parametrize("interval, slow, expected", [
    (1, False, [1, 2, 3, 4]),
    (2, False, [1, 3, 4]),
    (0, False, [1, 2, 3, 4]),
    (0, True, [1, 4]),
])
def test_held_out_logging(tmp_path, tiny_problem_path, monkeypatch, interval, slow, expected):
    example, _ = collect_proof_example(str(tiny_problem_path), step_limit = 100)
    assert example is not None
    split = ProblemSplit(10, tuple(range(8)))
    held = ProblemSplit(10, (8,))
    names = [f"problem{i}.p" for i in range(100)]
    train_name = next(name for name in names if split.contains(name))
    eval_name = next(name for name in names if held.contains(name))

    examples = [replace(example, problem_path = name) for name in (train_name, eval_name)]
    monkeypatch.setattr(training, "load_axiom_dataset", lambda _: (examples, [], {}))
    run = _FakeRun()
    init = {}
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb(run, init))
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    clock = [0.0]
    def monotonic():
        clock[0] += 1
        return clock[0]

    monkeypatch.setattr(training.time, "monotonic", monotonic)
    outputs = training.example_outputs
    def timed_outputs(model, selected, **kwargs):
        if selected[0].problem_path == eval_name:
            clock[0] += 100 if slow else -0.99

        return outputs(model, selected, **kwargs)

    monkeypatch.setattr(training, "example_outputs", timed_outputs)
    result = training.train_axiom_predictor(
        dataset = tmp_path / "dataset",
        output_dir = tmp_path / "model",
        split = split,
        evaluation_split = held,
        evaluate_every = interval,
        config = training.AxiomTrainingConfig(
            epochs = 4, hidden_dim = 8, message_rounds = 1,
            device = "cpu", num_workers = 1,
        ),
    )

    assert result["problems_proved"] == 1
    assert init["config"]["problems"] == [train_name]
    evaluations = [payload for payload, _ in run.logs if "evaluation/bce" in payload]
    assert [payload["epoch"] for payload in evaluations] == expected
    assert all(payload["evaluation/problems"] == 1 for payload in evaluations)
    scores = [payload["evaluation/macro_average_precision"] for payload in evaluations]
    assert run.summary["evaluation/best_macro_average_precision"] == max(scores)
    assert run.summary["evaluation/best_epoch"] == expected[scores.index(max(scores))]
    saved = json.loads((tmp_path / "model/evaluation_metrics.json").read_text())
    assert saved["macro_average_precision"] == scores[-1]
    assert saved["split"]["parts"] == [8]
    assert saved["epoch"] == 4
    table = next(p["evaluation_results/per_problem"] for p, _ in run.logs
                 if "evaluation_results/per_problem" in p)
    assert table[1][0][0] == eval_name
    assert run.finished == [0]

def test_evaluation_split_validation(tmp_path, monkeypatch):
    config = training.AxiomTrainingConfig(device = "cpu")
    with pytest.raises(ValueError, match = "overlap"):
        training.train_axiom_predictor(
            dataset = tmp_path, output_dir = tmp_path / "model", config = config,
            evaluation_split = ProblemSplit(),
            wandb_config = WandbConfig(enabled = False),
        )

    monkeypatch.setattr(training, "load_axiom_dataset", lambda _: ([], [], {}))
    with pytest.raises(ValueError, match = "no evaluation examples"):
        training.train_axiom_predictor(
            dataset = tmp_path, output_dir = tmp_path / "model", config = config,
            split = ProblemSplit(10, (0,)), evaluation_split = ProblemSplit(10, (8,)),
            wandb_config = WandbConfig(enabled = False),
        )

def test_evaluation_cli():
    args = build_parser().parse_args([
        "train", "--data-dir", "run", "--split", "10", "--parts",
        "0", "1", "2", "3", "4", "5", "6", "7", "--evaluate", "8",
    ])

    assert args.evaluate == [8]
    assert args.evaluate_every == 1

def test_fresh_input_evaluation_without_wandb(tmp_path, tiny_problem_path, monkeypatch):
    example, _ = collect_proof_example(str(tiny_problem_path), step_limit = 100)
    assert example is not None
    split = ProblemSplit(10, tuple(range(8)))
    held = ProblemSplit(10, (8,))
    names = [f"problem{i}.p" for i in range(100)]
    train_name = next(name for name in names if split.contains(name))
    eval_name = next(name for name in names if held.contains(name))

    problems = tmp_path / "problems"
    problems.mkdir()
    for name in (train_name, eval_name):
        (problems / name).write_text(tiny_problem_path.read_text())

    collected = []
    def collect(paths, **kwargs):
        collected.append(paths)
        return [replace(example, problem_path = path) for path in paths], []

    monkeypatch.setattr(training, "collect_examples", collect)
    output = tmp_path / "run"
    assert main([
        "train", str(problems), "--data-dir", str(output), "--device", "cpu",
        "--network", "SmallFull", "--epochs", "2", "--num-workers", "1",
        "--no-wandb", "--split", "10", "--parts",
        "0", "1", "2", "3", "4", "5", "6", "7", "--evaluate", "8",
    ]) == 0

    assert collected == [[str(problems / train_name)], [str(problems / eval_name)]]
    saved = json.loads((output / "model/evaluation_metrics.json").read_text())
    assert saved["problems_proved"] == 1
    assert saved["epoch"] == 2
