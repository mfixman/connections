"""W&B integration tests without making network requests."""

from __future__ import annotations

import sys
from typing import Any
from types import SimpleNamespace

import pytest

from axiom_prediction.training import AxiomTrainingConfig, train_axiom_predictor
from axiom_prediction.wandb_tracking import WandbConfig, WandbTracker

pytestmark = pytest.mark.training

@pytest.mark.parametrize("run_name", [None, "my-training-job"])
def test_training_names_follow_model_and_resume_count(tmp_path, tiny_problem_path, monkeypatch, run_name):
    run = _FakeRun()
    init_arguments = {}
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb(run, init_arguments))
    monkeypatch.setenv("WANDB_API_KEY", "test-key")

    for count in range(1, 4):
        train_axiom_predictor(
            [str(tiny_problem_path)],
            output_dir = tmp_path,
            config = AxiomTrainingConfig(
                model = "DefaultNoTerms",
                epochs = count,
                device = "cpu",
                num_workers = 1,
            ),
            wandb_config = WandbConfig(name = run_name, name_prefix = "smol-custom"),
            resume = count > 1,
        )

        expected = "DefaultNoTerms" if count == 1 else f"DefaultNoTerms-{count}"
        assert init_arguments["name"] == (run_name or expected)
        assert init_arguments["config"]["run_count"] == count

        if count == 1:
            (tmp_path / "training_runs.json").unlink()

def test_best_evaluation_retains_peak_and_first_epoch_on_ties():
    run = _FakeRun()
    tracker = WandbTracker(fake_wandb(run, {}), run)
    scores = [None, 0.4, 0.8, 0.8, 0.5, None]
    for epoch, score in enumerate(scores, start = 1):
        tracker.log_epoch(
            epoch,
            loss = 0.1,
            evaluation = {"macro_average_precision": score},
        )

        if epoch == 1:
            assert "evaluation/best_epoch" not in run.summary

    tracker.log_epoch(7, loss = 0.1)
    assert run.summary["evaluation/best_macro_average_precision"] == 0.8
    assert run.summary["evaluation/best_epoch"] == 3
    assert run.logs[4][0]["evaluation/macro_average_precision"] == 0.5
    assert "evaluation/macro_average_precision" not in run.logs[5][0]

class _FakeRun:
    def __init__(self):
        self.logs = []
        self.summary = {}
        self.models = []
        self.finished = []

    def log(self, payload, step = None):
        self.logs.append((payload, step))

    def log_model(self, *, path, name):
        self.models.append((path, name))

    def finish(self, *, exit_code = 0):
        self.finished.append(exit_code)

def fake_wandb(run: _FakeRun, init_arguments: dict) -> Any:
    module = SimpleNamespace(__name__ = "wandb")
    module.login = lambda **kwargs: True

    def init(**kwargs):
        init_arguments.update(kwargs)
        return run

    module.init = init
    module.Histogram = lambda values, num_bins: ("histogram", list(values), num_bins)
    module.Table = lambda data, columns: ("table", data, columns)
    module.plot = SimpleNamespace(
        line = lambda table,
        x,
        y,
        title: ("line", table, x, y, title),
    )

    return module

def test_training_logs_progress_results_and_model(
    tmp_path,
    tiny_problem_path,
    monkeypatch,
):
    run = _FakeRun()
    init_arguments = {}
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb(run, init_arguments))
    key_file = tmp_path / "wandb_key"
    key_file.write_text("test-key", encoding = "utf-8")

    train_axiom_predictor(
        [str(tiny_problem_path)],
        output_dir = tmp_path / "output",
        config = AxiomTrainingConfig(
            epochs = 2,
            hidden_dim = 8,
            message_rounds = 1,
            num_hidden_layers = 1,
            device = "cpu",
        ),
        wandb_config = WandbConfig(enabled = True, key_file = key_file),
        run_properties = {"seed": 17, "split": 4, "parts": [0, 2]},
    )

    assert [payload["epoch"] for payload, _step in run.logs if "epoch" in payload] == [
        1,
        2,
    ]

    logged_keys = {key for payload, _step in run.logs for key in payload}
    assert "collection/problems_processed" in logged_keys
    assert "train/loss" in logged_keys
    assert "results/ranked_axioms" in logged_keys
    assert "results/per_problem" in logged_keys
    ranked = next(
        payload["results/ranked_axioms"]
        for payload, _ in run.logs
        if "results/ranked_axioms" in payload
    )

    per_problem = next(
        payload["results/per_problem"]
        for payload, _ in run.logs
        if "results/per_problem" in payload
    )

    assert "in_sat_core" in ranked[2]
    assert "axioms_in_sat_core" in per_problem[2]
    assert run.models and run.models[0][1] == "axiom-predictor"
    assert run.finished == [0]
    assert init_arguments["config"]["cli_arguments"] == {
        "seed": 17,
        "split": 4,
        "parts": [0, 2],
    }

    assert init_arguments["config"]["sat_policy"] == "satresetcop"

    def failed_optimizer(*args, **kwargs):
        raise RuntimeError("optimizer setup failed")

    monkeypatch.setattr("axiom_prediction.training.torch.optim.Adam", failed_optimizer)
    with pytest.raises(RuntimeError, match = "optimizer setup failed"):
        train_axiom_predictor(
            [str(tiny_problem_path)],
            output_dir = tmp_path / "failed",
            config = AxiomTrainingConfig(
                epochs = 1,
                hidden_dim = 8,
                device = "cpu",
                num_workers = 1,
            ),
            wandb_config = WandbConfig(key_file = key_file),
        )

    assert run.finished == [0, 1]
