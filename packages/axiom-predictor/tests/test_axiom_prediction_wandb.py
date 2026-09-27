"""W&B integration tests without making network requests."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any
from types import SimpleNamespace

import pytest


from axiom_prediction.training import AxiomTrainingConfig, train_axiom_predictor
from axiom_prediction.wandb_tracking import WandbConfig, WandbTracker

pytestmark = pytest.mark.training


class _FakeRun:
    def __init__(self):
        self.logs = []
        self.summary = {}
        self.models = []
        self.finished = []

    def log(self, payload, step=None):
        self.logs.append((payload, step))

    def log_model(self, *, path, name):
        self.models.append((path, name))

    def finish(self, *, exit_code=0):
        self.finished.append(exit_code)


def _fake_wandb(run: _FakeRun, init_arguments: dict) -> Any:
    module = SimpleNamespace(__name__="wandb")
    module.login = lambda **kwargs: True

    def init(**kwargs):
        init_arguments.update(kwargs)
        return run

    module.init = init
    module.Histogram = lambda values, num_bins: ("histogram", list(values), num_bins)
    module.Table = lambda data, columns: ("table", data, columns)
    module.plot = SimpleNamespace(line=lambda table, x, y, title: ("line", table, x, y, title))
    return module


def test_automatic_wandb_disables_with_warning_when_key_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    tracker = WandbTracker.start(
        WandbConfig(key_file=tmp_path / "missing"),
        job_type="train",
        run_config={},
    )
    assert tracker is None
    assert "W&B disabled" in capsys.readouterr().err


class _NamedRun(_FakeRun):
    def __init__(self, name):
        super().__init__()
        self.name = name


@pytest.mark.parametrize(
    ("generated", "prefix", "expected"),
    [
        ("honest-forest-6", "smol", "smol-honest-forest-6"),
        ("smol-honest-forest-6", "smol", "smol-honest-forest-6"),
        ("honest-forest-6", None, "honest-forest-6"),
    ],
)
def test_wandb_run_name_prefix(monkeypatch, generated, prefix, expected):
    run = _NamedRun(generated)
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run, {}))
    tracker = WandbTracker.start(
        WandbConfig(enabled=True, name_prefix=prefix), job_type="train", run_config={}
    )
    assert tracker is not None and tracker.run.name == expected


def _failing_wandb() -> Any:
    module = SimpleNamespace(__name__="wandb")
    module.login = lambda **kwargs: True

    def init(**kwargs):
        raise RuntimeError("Failed to read port info after 300 seconds")

    module.init = init
    return module


def test_automatic_wandb_disables_with_warning_when_startup_fails(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "wandb", _failing_wandb())
    tracker = WandbTracker.start(WandbConfig(), job_type="train", run_config={})
    assert tracker is None
    assert "failed to start" in capsys.readouterr().err


def test_explicit_wandb_raises_when_startup_fails(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "wandb", _failing_wandb())
    with pytest.raises(RuntimeError, match="port info"):
        WandbTracker.start(WandbConfig(enabled=True), job_type="train", run_config={})


def test_service_wait_defaults_to_long_timeout_but_respects_environment(
    monkeypatch,
):
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "wandb", _failing_wandb())
    monkeypatch.delenv("WANDB__SERVICE_WAIT", raising=False)
    WandbTracker.start(WandbConfig(), job_type="train", run_config={})
    assert os.environ["WANDB__SERVICE_WAIT"] == "300"

    monkeypatch.setenv("WANDB__SERVICE_WAIT", "45")
    WandbTracker.start(WandbConfig(), job_type="train", run_config={})
    assert os.environ["WANDB__SERVICE_WAIT"] == "45"


def test_collection_logs_every_problem():
    run = _FakeRun()
    tracker = WandbTracker(SimpleNamespace(), run)
    tracker.log_collection(1, 3, 1, 0, "one.p", "proved")
    tracker.log_collection(2, 3, 1, 1, "two.p", "failed")
    tracker.log_collection(3, 3, 2, 1, "three.p", "proved")
    assert [payload["collection/problems_processed"] for payload, _ in run.logs] == [
        1,
        2,
        3,
    ]


def test_training_logs_progress_results_and_model(tmp_path, tiny_problem_path, monkeypatch):
    run = _FakeRun()
    init_arguments = {}
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run, init_arguments))
    key_file = tmp_path / "wandb_key"
    key_file.write_text("test-key", encoding="utf-8")

    train_axiom_predictor(
        [str(tiny_problem_path)],
        output_dir=tmp_path / "output",
        config=AxiomTrainingConfig(
            epochs=2,
            hidden_dim=8,
            message_rounds=1,
            num_hidden_layers=1,
            device="cpu",
        ),
        wandb_config=WandbConfig(enabled=True, key_file=key_file),
        run_properties={"seed": 17, "split": 4, "parts": [0, 2]},
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


def test_model_name_prefixes_groups_and_tags_wandb_runs(tmp_path):
    from axiom_prediction.cli import _wandb_config, _wandb_name_prefix, build_parser

    assert _wandb_name_prefix(Path("smol-agt"), "split4-a") == "smol-split4-a"
    assert _wandb_name_prefix(Path("artifacts/run"), "split4-a") == "split4-a"
    config = _wandb_config(
        build_parser().parse_args(
            [
                "train",
                "--data-dir",
                str(tmp_path / "smol-run"),
                "--model-name",
                "split4-a",
            ]
        )
    )
    assert config.name_prefix == "smol-split4-a"
    assert config.group == "smol-run/split4-a"
    assert config.tags == ("split4-a",)
    unnamed = _wandb_config(
        build_parser().parse_args(["train", "--data-dir", str(tmp_path / "run")])
    )
    assert unnamed.group is None and unnamed.tags == () and unnamed.name_prefix is None


def test_wandb_start_passes_group_and_tags(monkeypatch):
    run = _NamedRun("honest-forest-6")
    init_arguments = {}
    monkeypatch.setenv("WANDB_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "wandb", _fake_wandb(run, init_arguments))
    tracker = WandbTracker.start(
        WandbConfig(
            enabled=True,
            name_prefix="smol-split4-a",
            group="smol-run/split4-a",
            tags=("split4-a",),
        ),
        job_type="train",
        run_config={},
    )
    assert tracker is not None
    assert tracker.run.name == "smol-split4-a-honest-forest-6"
    assert init_arguments["group"] == "smol-run/split4-a"
    assert init_arguments["tags"] == ["split4-a"]


def test_run_results_are_logged_per_problem_and_as_a_table():
    run = _FakeRun()
    tracker = WandbTracker(_fake_wandb(run, {}), run)
    results = [
        {"problem": "a.p", "outcome": "Proved", "proved": True, "seconds": 1.5},
        {"problem": "b.p", "outcome": "Timeout", "proved": False, "seconds": 120.0},
    ]
    for index, result in enumerate(results, start=1):
        tracker.log_run_result(index, len(results), 1, result)
    tracker.log_run_results(results, {"mode": "weighted", "problems": 2, "proved": 1})
    assert [
        payload["run/problems_processed"]
        for payload, _ in run.logs
        if "run/problems_processed" in payload
    ] == [1, 2]
    table = next(payload["run/results"] for payload, _ in run.logs if "run/results" in payload)
    assert table[2][0] == "problem" and len(table[1]) == 2
    assert run.summary["run/proved"] == 1
