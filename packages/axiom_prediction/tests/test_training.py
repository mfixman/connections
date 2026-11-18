from dataclasses import replace
import json
import math

import pytest
import torch

from axiom_prediction.model import load_checkpoint
from axiom_prediction.split import ProblemSplit
from axiom_prediction.training import AxiomTrainingConfig, problem_loss, train_axiom_predictor
from axiom_prediction.wandb_tracking import WandbConfig

small = AxiomTrainingConfig(epochs = 3, hidden_dim = 8, message_rounds = 1, device = "cpu", num_workers = 1, batch_size = 4)
train_parts = ProblemSplit(10, list(range(8)))
no_wandb = WandbConfig(enabled = False)

def weights(directory):
    return load_checkpoint(directory / "model.ts")["model_state_dict"]

def test_resume_matches_uninterrupted_training_and_rejects_changes(tmp_path, joined_dataset):
    options = dict(dataset = joined_dataset, split = train_parts, wandb_config = no_wandb)
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train_axiom_predictor(output_dir = full, config = small, **options)
    assert sorted(path.name for path in full.glob("epoch-*.ts")) == ["epoch-0001.ts", "epoch-0002.ts", "epoch-0003.ts"]
    assert (full / "model.ts").read_bytes() == (full / "epoch-0003.ts").read_bytes()

    train_axiom_predictor(output_dir = resumed, config = replace(small, epochs = 1), **options)
    with pytest.raises(ValueError, match = "already exists"):
        train_axiom_predictor(output_dir = resumed, config = small, **options)

    with pytest.raises(ValueError, match = "same data and configuration"):
        train_axiom_predictor(output_dir = resumed, config = replace(small, seed = 1), resume = True, **options)

    with pytest.raises(ValueError, match = "same data and configuration"):
        train_axiom_predictor(output_dir = resumed, config = small, resume = True, **{**options, "min_axioms": 4})

    train_axiom_predictor(output_dir = resumed, config = small, resume = True, **options)
    assert load_checkpoint(resumed / "model.ts")["epoch"] == 3
    for name, expected in weights(full).items():
        assert torch.equal(expected, weights(resumed)[name]), name

    with pytest.raises(ValueError, match = "already reached epoch 3"):
        train_axiom_predictor(output_dir = resumed, config = small, resume = True, **options)

def test_held_out_parts_are_scored_but_never_trained_on(tmp_path, joined_dataset, wandb_run):
    held_out = ProblemSplit(10, [8])
    with pytest.raises(ValueError, match = "not overlap"):
        train_axiom_predictor(dataset = joined_dataset, output_dir = tmp_path / "overlap", split = train_parts, evaluation_split = ProblemSplit(10, [7]))

    with pytest.raises(ValueError, match = "training split"):
        train_axiom_predictor(dataset = joined_dataset, output_dir = tmp_path / "scheme", split = train_parts, evaluation_split = ProblemSplit(10, [8], 4))

    metrics = train_axiom_predictor(
        dataset = joined_dataset,
        output_dir = tmp_path / "model",
        config = replace(small, epochs = 4),
        split = train_parts,
        evaluation_split = held_out,
        evaluate_every = 2,
        wandb_config = WandbConfig(name = "held-out"),
        run_properties = {"seed": 17},
    )

    trained = wandb_run.init["config"]["problems"]
    assert len(trained) == metrics["problems_proved"] > 0
    assert all(train_parts.contains(problem) and not held_out.contains(problem) for problem in trained)
    evaluation = json.loads((tmp_path / "model" / "evaluation_metrics.json").read_text())
    assert evaluation["epoch"] == 4 and evaluation["problems_proved"] > 0
    assert [log["epoch"] for log in wandb_run.logs if "evaluation/roc_auc" in log] == [1, 3, 4]

    assert [log["epoch"] for log in wandb_run.logs if "train/loss" in log] == [1, 2, 3, 4]
    assert any("results/ranked_axioms" in log for log in wandb_run.logs)
    assert wandb_run.models[0][1] == "axiom-predictor"
    assert wandb_run.finished == [0]
    assert wandb_run.init["name"] == "held-out"
    assert wandb_run.init["config"]["cli_arguments"] == {"seed": 17}

def test_resumed_training_continues_its_wandb_run_and_reports_failures(tmp_path, joined_dataset, wandb_run, monkeypatch):
    options = dict(dataset = joined_dataset, output_dir = tmp_path, split = train_parts, wandb_config = WandbConfig(name = "job"))
    train_axiom_predictor(config = replace(small, epochs = 1), **options)
    first = dict(wandb_run.init)
    train_axiom_predictor(config = replace(small, epochs = 2), resume = True, **options)
    assert (first["name"], wandb_run.init["name"]) == ("job", None)
    assert first["id"] == wandb_run.init["id"] == json.loads((tmp_path / "wandb-run.json").read_text())["id"]
    assert wandb_run.init["config"]["run_count"] == 2

    monkeypatch.setattr("axiom_prediction.training.dataset_fingerprint", lambda examples: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        train_axiom_predictor(config = replace(small, epochs = 3), resume = True, **options)

    assert wandb_run.finished == [0, 0, 1]

def test_problem_loss_counts_problems_not_clauses():
    logits = torch.zeros(4)
    labels = torch.tensor([1.0, 1.0, 1.0, 0.0])
    even = problem_loss(logits, labels, [3, 1], torch.ones(2))
    assert even.item() == pytest.approx(math.log(2))

    logits = torch.tensor([9.0, 9.0, 9.0, 0.0])
    large_is_easy = problem_loss(logits, labels, [3, 1], torch.ones(2))
    assert large_is_easy.item() == pytest.approx((1.234e-4 + math.log(2)) / 2, abs = 1e-3)

    shared = problem_loss(logits, labels, [3, 1], torch.tensor([1.0, 0.0]))
    assert shared.item() < 1e-3
