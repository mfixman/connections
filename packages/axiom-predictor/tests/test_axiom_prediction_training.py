"""Training-level checks for the axiom predictor."""

from __future__ import annotations

import pytest
from typing import Callable, NoReturn, cast

import torch

import axiom_prediction.training as training_module
import axiom_prediction.dataset as dataset_module

from axiom_prediction.data import axiom_training_example_to_json
from axiom_prediction.dataset import collect_axiom_dataset, load_axiom_dataset
from axiom_prediction.model import AxiomPredictor

from axiom_prediction.tptp import collect_proof_example, load_tptp_problem
from axiom_prediction.training import AxiomTrainingConfig, train_axiom_predictor
from axiom_prediction.wandb_tracking import WandbConfig

from axiom_prediction.io import write_json_atomic

fail = cast(Callable[..., NoReturn], pytest.fail)
skip = cast(Callable[..., NoReturn], pytest.skip)

pytestmark = pytest.mark.training

def test_cuda_inference_smoke_when_available(tmp_path, tiny_problem_path):
    if not torch.cuda.is_available():
        skip(reason = "CUDA is unavailable")

    train_axiom_predictor(
        [str(tiny_problem_path)],
        output_dir = tmp_path,
        config = AxiomTrainingConfig(
            epochs = 1,
            hidden_dim = 8,
            message_rounds = 1,
            device = "cpu",
        ),
        wandb_config = WandbConfig(enabled = False),
    )

    loaded = load_tptp_problem(tiny_problem_path)
    predictions = AxiomPredictor.load(tmp_path, device = "cuda").predict(
        loaded.matrix,
        axiom_clause_ids = loaded.axiom_clause_ids,
        conjecture_clause_ids = loaded.conjecture_clause_ids,
    )

    assert predictions

def test_dataset_recovers_worker_result_left_in_partial(
    tmp_path,
    tiny_problem_path,
    monkeypatch,
):
    example, outcome = collect_proof_example(
        tiny_problem_path,
        step_limit = 2_000,
        timeout_seconds = 30,
    )

    assert outcome == "proved"
    assert example is not None

    dataset = tmp_path / "dataset"
    partial_examples = dataset / "partial" / "examples"
    partial_examples.mkdir(parents = True)

    problem = str(tiny_problem_path)
    key = dataset_module.problem_key(problem)
    write_json_atomic(
        partial_examples / f"{key}.json",
        axiom_training_example_to_json(example),
    )

    monkeypatch.setattr(
        dataset_module,
        "collect_proof_example",
        lambda *args,
        **kwargs: fail(reason = "partial result was recomputed"),
    )

    summary = collect_axiom_dataset(
        [problem],
        output_dir = dataset,
        step_limit = 2_000,
        timeout_seconds = 30,
    )

    assert summary["problems_proved"] == 1
    assert summary["problems_reused"] == 1
    assert (dataset / "examples" / f"{key}.json").is_file()
    assert not (partial_examples / f"{key}.json").exists()

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
            timeout_seconds = 30,
            sat_policy = "satcop",
            num_workers = 2,
        )
    )

    assert {result.problem for result in results} == set(problems)
    assert all(result.example is not None for result in results)
    assert all(result.outcome == "proved" for result in results)

def test_dataset_training_uses_only_the_requested_split(tmp_path):
    from axiom_prediction.split import ProblemSplit

    problems = [
        "packages/axiom-predictor/tests/fixtures/problems/tiny_theorem.p",
        "packages/axiom-predictor/tests/fixtures/problems/marked_conjecture_clausification.p",
        "packages/axiom-predictor/tests/fixtures/problems/t11_wellord1.p",
    ]

    dataset = tmp_path / "dataset"
    collect_axiom_dataset(
        problems,
        output_dir = dataset,
        timeout_seconds = 30,
        num_workers = 1,
    )

    examples, _, _ = load_axiom_dataset(dataset)
    split = next(
        ProblemSplit(4, (part,))
        for part in range(4)
        if 0
        < len(
            ProblemSplit(4, (part,)).select(
                [example.problem_path for example in examples]
            )
        )
        < len(examples)
    )

    expected = split.select([example.problem_path for example in examples])

    metrics = train_axiom_predictor(
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
        split = split,
    )

    assert metrics["problems_proved"] == len(expected)

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
        timeout_seconds = 30,
        num_workers = 1,
    )

    examples, _, _ = load_axiom_dataset(dataset)
    names = [example.problem_path for example in examples]
    train_part = next(
        part
        for part in range(4)
        if 0 < len(ProblemSplit(4, (part,)).select(names)) < len(names)
    )

    held_out = ProblemSplit(4, tuple(part for part in range(4) if part != train_part))
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
        split = ProblemSplit(4, (train_part,)),
    )

    monkeypatch.setattr(
        training_module,
        "collect_examples",
        lambda *args,
        **kwargs: fail(reason = "dataset evaluation repeated proof search"),
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
    assert metrics["training_split"] == ProblemSplit(4, (train_part,)).to_dict()
    assert "not held-out" not in capsys.readouterr().err

    evaluate_axiom_predictor(
        tmp_path / "model",
        dataset = dataset,
        device = "cpu",
        split = ProblemSplit(4, (train_part,)),
        wandb_config = WandbConfig(enabled = False),
    )

    assert "not held-out" in capsys.readouterr().err
