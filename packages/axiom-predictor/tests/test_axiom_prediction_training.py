"""Training-level checks for the axiom predictor."""

from __future__ import annotations

import pytest


import axiom_prediction.training as training_module
import axiom_prediction.dataset as dataset_module

from axiom_prediction.dataset import collect_axiom_dataset, load_axiom_dataset

from axiom_prediction.training import AxiomTrainingConfig, train_axiom_predictor
from axiom_prediction.wandb_tracking import WandbConfig


pytestmark = pytest.mark.training

def test_real_multiprocess_collection():
    problems = [
        "packages/axiom-predictor/tests/fixtures/problems/tiny_theorem.p",
        "packages/axiom-predictor/tests/fixtures/problems/t11_wellord1.p",
    ]

    results = list(
        dataset_module.collect_problems_parallel(
            problems,
            tptp_root = None,
            step_limit = 2_000,
            timeout_s = 30,
            sat_policy = "satcop",
            num_workers = 2,
        )
    )

    assert {result.problem for result in results} == set(problems)
    assert all(result.example is not None for result in results)
    assert all(result.outcome == "proved" for result in results)

def test_evaluate_scores_a_dataset_split_without_proof_search(
    tmp_path,
    monkeypatch,
    capsys,
):
    from axiom_prediction.split import ProblemSplit
    from axiom_prediction.training import evaluate_axiom_predictor

    problems = [
        "packages/axiom-predictor/tests/fixtures/problems/tiny_theorem.p",
        "packages/axiom-predictor/tests/fixtures/problems/marked_conjecture_clausification.p",
        "packages/axiom-predictor/tests/fixtures/problems/t11_wellord1.p",
    ]

    dataset = tmp_path / "dataset"
    collect_axiom_dataset(
        problems,
        output_dir = dataset,
        timeout_s = 30,
        num_workers = 1,
    )

    examples, _, _ = load_axiom_dataset(dataset)
    names = [example.problem_path for example in examples]
    train_part = next(
        part
        for part in range(4)
        if 0 < len(ProblemSplit(4, [part]).select(names)) < len(names)
    )

    held_out = ProblemSplit(4, [part for part in range(4) if part != train_part])
    train_axiom_predictor(
        dataset = dataset,
        output_dir = tmp_path / "model",
        config = AxiomTrainingConfig(
            epochs = 1,
            hidden_dim = 8,
            message_rounds = 1,
            num_hidden_layers = 1,
            device = "cpu",
        ),
        wandb_config = WandbConfig(enabled = False),
        split = ProblemSplit(4, [train_part]),
    )

    monkeypatch.setattr(
        training_module,
        "collect_examples",
        lambda *args,
        **kwargs: pytest.fail("dataset evaluation repeated proof search"),
    )

    capsys.readouterr()

    metrics = evaluate_axiom_predictor(
        tmp_path / "model",
        dataset = dataset,
        device = "cpu",
        split = held_out,
        wandb_config = WandbConfig(enabled = False),
    )

    assert metrics["problems_proved"] == len(held_out.select(names))
    assert metrics["split"] == held_out.to_dict()
    assert metrics["training_split"] == ProblemSplit(4, [train_part]).to_dict()
    assert "not held-out" not in capsys.readouterr().err

    evaluate_axiom_predictor(
        tmp_path / "model",
        dataset = dataset,
        device = "cpu",
        split = ProblemSplit(4, [train_part]),
        wandb_config = WandbConfig(enabled = False),
    )

    assert "not held-out" in capsys.readouterr().err
