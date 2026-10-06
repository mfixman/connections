from dataclasses import replace
import json
import multiprocessing as mp
import signal
import time
from typing import Any

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from axiom_prediction.cli import main
from axiom_prediction.dataset import collect_axiom_dataset
from axiom_prediction.model import AxiomPredictionNetwork
from axiom_prediction.run import RunConfig, run_problem, run_problems
from axiom_prediction.search_workers import supervised_results
from axiom_prediction.training import AxiomTrainingConfig, train_axiom_predictor

from axiom_prediction.wandb_tracking import WandbConfig
from axiom_prediction.resume import restore_training, snapshot_training

def test_training_resume_ignores_collection_completion_order(tmp_path, tiny_problem_path, monkeypatch):
    from axiom_prediction import training
    from axiom_prediction.dataset import CollectedAxiomProblem
    from axiom_prediction.tptp import collect_proof_example

    example, _ = collect_proof_example(str(tiny_problem_path), step_limit = 100)
    assert example is not None
    other = replace(example, problem_path = "b.p", labels = [1 - y for y in example.labels])
    records = [
        CollectedAxiomProblem("a.p", replace(example, problem_path = "a.p"), "proved"),
        CollectedAxiomProblem("b.p", other, "proved"),
    ]

    monkeypatch.setattr(training, "collect_problems_parallel", lambda *args, **kwargs: iter(records))
    config = AxiomTrainingConfig(
        epochs = 2,
        hidden_dim = 8,
        message_rounds = 1,
        device = "cpu",
        num_workers = 1,
        batch_size = 1,
    )

    options = dict(problems = ["a.p", "b.p"], wandb_config = WandbConfig(enabled = False))
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    train_axiom_predictor(output_dir = full, config = config, **options)
    train_axiom_predictor(output_dir = resumed, config = replace(config, epochs = 1), **options)

    records.reverse()
    train_axiom_predictor(output_dir = resumed, config = config, resume = True, **options)
    expected = torch.load(full / "model.pt", weights_only = True)
    actual = torch.load(resumed / "model.pt", weights_only = True)
    assert actual["epoch"] == 2
    for name, weights in expected["model_state_dict"].items():
        assert torch.equal(weights, actual["model_state_dict"][name]), name

def test_resume_matches_uninterrupted_training_and_rejects_damage(
    tmp_path,
    tiny_problem_path,
    monkeypatch,
    capsys,
):
    dataset = tmp_path / "dataset"
    problems = [str(tiny_problem_path)]
    for index in range(2):
        problem = tmp_path / f"variant{index}.p"
        problem.write_text(
            tiny_problem_path.read_text()
            + f"fof(extra, axiom, q({index})).\n"
        )

        problems.append(str(problem))

    collect_axiom_dataset(problems, output_dir = dataset, num_workers = 1)
    config = AxiomTrainingConfig(
        epochs = 3,
        hidden_dim = 8,
        message_rounds = 1,
        device = "cpu",
        num_workers = 1,
        batch_size = 1,

        log_every = 0,
    )

    options: dict[str, Any] = {
        "dataset": dataset,
        "wandb_config": WandbConfig(enabled = False),
    }
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    import axiom_prediction.training as training

    saved_epochs = []
    save = training.save_checkpoint

    def tracked_save(*args, **kwargs):
        saved_epochs.append(kwargs["epoch"])
        return save(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(training, "save_checkpoint", tracked_save)
        train_axiom_predictor(output_dir = full, config = config, **options)

    assert saved_epochs == [1, 2, 3]
    assert sorted(path.name for path in full.glob("epoch-*.pt")) == [
        "epoch-0001.pt", "epoch-0002.pt", "epoch-0003.pt",
    ]

    assert (full / "model.pt").read_bytes() == (full / "epoch-0003.pt").read_bytes()
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [event["epoch"] for event in events if event["event"] == "epoch"] == [1, 2, 3]

    train_axiom_predictor(
        output_dir = resumed,
        config = replace(config, epochs = 1),
        **options,
    )

    with pytest.raises(ValueError, match = "already exists"):
        train_axiom_predictor(output_dir = resumed, config = config, **options)

    first_epoch = (resumed / "epoch-0001.pt").read_bytes()
    train_axiom_predictor(output_dir = resumed, config = config, resume = True, **options)
    assert (resumed / "epoch-0001.pt").read_bytes() == first_epoch
    for epoch in range(1, 4):
        name = f"epoch-{epoch:04d}.pt"
        archived = torch.load(resumed / name, weights_only = True)
        uninterrupted = torch.load(full / name, weights_only = True)
        assert archived["epoch"] == epoch
        assert archived["training_state"]["optimizer"]["state"]
        for key, weights in uninterrupted["model_state_dict"].items():
            assert torch.equal(weights, archived["model_state_dict"][key]), key

    assert (resumed / "model.pt").read_bytes() == (resumed / "epoch-0003.pt").read_bytes()
    expected = torch.load(full / "model.pt", weights_only = True)
    actual = torch.load(resumed / "model.pt", weights_only = True)
    for name, weights in expected["model_state_dict"].items():
        assert torch.equal(weights, actual["model_state_dict"][name]), name

    assert actual["epoch"] == 3
    assert actual["training_state"]["optimizer"]["state"]
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: True)
        state = snapshot_training(optimizer, training.random.Random(0), "data")
        assert state["cuda_rng"] == []

    check_cuda_resume(actual, monkeypatch)
    extended = replace(config, epochs = 4)
    with pytest.raises(ValueError, match = "same data and configuration"):
        train_axiom_predictor(
            output_dir = resumed,
            config = replace(extended, seed = 42),
            resume = True,
            **options,
        )

    before = (resumed / "model.pt").read_bytes()
    forward = AxiomPredictionNetwork.forward
    monkeypatch.setattr(
        AxiomPredictionNetwork,
        "forward",
        lambda self, batch: forward(self, batch) * float("nan"),
    )

    with pytest.raises(FloatingPointError, match = "non-finite"):
        train_axiom_predictor(
            output_dir = resumed,
            config = extended,
            resume = True,
            **options,
        )

    assert (resumed / "model.pt").read_bytes() == before
    assert not (resumed / "epoch-0004.pt").exists()

    record = next((dataset / "examples").glob("*.json"))
    payload = json.loads(record.read_text())
    payload["labels"][0] = 1 - payload["labels"][0]
    record.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match = "same data and configuration"):
        train_axiom_predictor(
            output_dir = resumed,
            config = extended,
            resume = True,
            **options,
        )

def check_cuda_resume(checkpoint, monkeypatch):
    saved = checkpoint["training_state"]
    parameter = SimpleNamespace(device = torch.device("cuda:0"))
    model = SimpleNamespace(
        parameters = lambda: iter([parameter]),
        load_state_dict = lambda state: None,
    )

    optimizer = SimpleNamespace(load_state_dict = lambda state: None)
    config = {**checkpoint["training_config"], "epochs": 4, "device": "cuda:0"}
    checkpoint["training_config"]["device"] = "cuda:1"
    restored = []
    import random

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
        patch.setattr(torch.cuda, "device_count", lambda: 1)
        patch.setattr(
            torch.cuda,
            "set_rng_state",
            lambda state, device: restored.append((state, device)),
        )

        for states in ([torch.tensor([1])], [torch.tensor([0]), torch.tensor([1])]):
            saved["cuda_rng"] = states
            assert restore_training(
                "unused.pt",
                model,
                optimizer,
                random.Random(0),
                config,
                saved["dataset_fingerprint"],
            ) == 4

    assert all(state.item() == 1 and device == parameter.device for state, device in restored)

def ignores_alarm(problem, *args):
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, signal.SIG_IGN)

    Path(problem).touch()
    time.sleep(30)
    return {"problem": problem, "proved": True}

def test_supervisor_terminates_unresponsive_workers(tmp_path, monkeypatch):
    import axiom_prediction.dataset as dataset

    before = {child.pid for child in mp.active_children()}
    started = time.monotonic()
    results = list(
        supervised_results(
            ignores_alarm,
            [(str(tmp_path / "one"),), (str(tmp_path / "two"),)],
            workers = 1,
            timeout = 0.05,
        )
    )

    assert [result["outcome"] for result in results] == ["Timeout", "Timeout"]
    assert time.monotonic() - started < 3
    assert {child.pid for child in mp.active_children()} == before

    options: dict[str, Any] = {
        "tptp_root": None,
        "step_limit": 100,
        "timeout_s": 3,
        "sat_policy": "satcop",
        "num_workers": 1,
    }

    monkeypatch.setattr(dataset, "collect_one_problem", ignores_alarm)
    marker = tmp_path / "started"
    results = list(dataset.collect_problems_parallel([str(marker)], **options))
    assert marker.exists(), "worker must enter the hanging function before timing out"
    assert results[0].outcome == "Timeout"

    monkeypatch.setattr(dataset, "collect_one_problem_to_partial", ignores_alarm)
    output = tmp_path / "dataset"
    summary = dataset.collect_axiom_dataset(
        [str(tmp_path / "record")],
        output_dir = output,
        **options,
    )

    assert summary["problems_failed"] == 1
    failure = next((output / "failures").glob("*.json"))
    assert json.loads(failure.read_text())["outcome"] == "Timeout"
    assert {child.pid for child in mp.active_children()} == before

def test_policy_inheritance_fallback_and_fresh_evaluation(
    tmp_path,
    tiny_problem_path,
    capsys,
):
    config = AxiomTrainingConfig(
        epochs = 1,
        hidden_dim = 8,
        device = "cpu",
        num_workers = 1,
        sat_policy = "SatCoP",
    )

    train_axiom_predictor(
        [str(tiny_problem_path)],
        output_dir = tmp_path,
        config = config,
        wandb_config = WandbConfig(enabled = False),
    )

    common = ["--device", "cpu", "--num-workers", "1", "--no-wandb"]
    capsys.readouterr()
    assert main(["evaluate", str(tmp_path), str(tiny_problem_path), *common]) == 0
    metrics = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert metrics["sat_policy"] == "satcop"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model").symlink_to(tmp_path, target_is_directory = True)
    assert main(
        ["evaluate", "--data-dir", str(run_dir), str(tiny_problem_path), *common]
    ) == 0

    metrics = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert metrics["sat_policy"] == "satcop"
    assert main(
        [
            "evaluate", str(tmp_path), str(tiny_problem_path), *common,
            "--policy", "SatResetCoP",
        ]
    ) == 2

    cnf = tmp_path / "plain.p"
    cnf.write_text("cnf(a, axiom, p).\ncnf(b, axiom, ~p).\n")
    search = RunConfig(checkpoint = str(tmp_path), device = "cpu", timeout_s = 10)
    result = run_problem(str(cnf), tptp_root = None, config = search)
    assert result["proved"] and result["policy"] == "satcop"
    assert "conjecture" in result["guidance_fallback"]

    result = list(
        run_problems(
            [str(cnf)],
            tptp_root = None,
            config = search,
            num_workers = 1,
        )
    )[0]

    assert result["proved"] and result["guidance_fallback"]
    capsys.readouterr()
    assert main(["run", str(cnf), "--model", str(tmp_path), *common]) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["policy"] == "satcop"

    assert main(
        [
            "run", str(cnf), "--model", str(tmp_path), *common,
            "--policy", "SatResetCoP",
        ]
    ) == 0

    captured = capsys.readouterr()
    assert "warning: dataset labels were collected" in captured.err
    assert json.loads(captured.out.splitlines()[0])["policy"] == "satresetcop"
