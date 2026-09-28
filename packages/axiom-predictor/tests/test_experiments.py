import json

import pytest
import torch

from axiom_prediction.cli import main
from axiom_prediction.graph import build_axiom_graph
from axiom_prediction.inputs import select_graph_input
from axiom_prediction.model import AxiomModelConfig, AxiomPredictionNetwork, AxiomPredictor, save_checkpoint
from axiom_prediction.tptp import load_tptp_problem
from axiom_prediction.models import available_models, load_model_class

from axiom_prediction.choices import GraphInputKind, ProverPolicy

@pytest.mark.parametrize("policy", ["satcop", "satresetcop"])
def test_policy_and_network_experiments(tmp_path, policy, capsys):
    problem = "packages/axiom-predictor/tests/fixtures/problems/marked_conjecture_clausification.p"
    common = ["--data-dir", str(tmp_path), "--num-workers", "1"]
    assert main(
        ["collect", problem, *common, "--policy", ProverPolicy(policy).value]
    ) == 0

    loaded = load_tptp_problem(problem)
    for name in available_models():
        model_class = load_model_class(name)
        assert (
            main(
                [
                    "train",
                    *common,

                    "--policy",
                    policy,
                    "--model-name",
                    name,

                    "--network",
                    name,
                    "--epochs",
                    "1",

                    "--device",
                    "cpu",
                    "--no-wandb",
                ]
            )
            == 0
        )

        directory = tmp_path / "models" / name
        predictor = AxiomPredictor.load(directory)
        expected = model_class.default_config.to_dict()

        assert type(predictor.model) is model_class
        assert predictor.model.config.to_dict() == expected
        assert predictor.training_config["sat_policy"] == policy

        checkpoint = torch.load(directory / "model.pt", weights_only = True)
        assert checkpoint["epoch"] == 1
        assert checkpoint["model_config"] == expected
        assert type(checkpoint["model_config"]["graph_input"]) is str
        assert type(checkpoint["training_config"]["sat_policy"]) is str

        assert isinstance(predictor.model.config.graph_input, GraphInputKind)

        predictions = predictor.predict(
            loaded.matrix,
            axiom_clause_ids = loaded.axiom_clause_ids,
            conjecture_clause_ids = loaded.conjecture_clause_ids,
        )

        assert len(predictions) == len(loaded.axiom_clause_ids)
        assert all(0 <= p.probability <= 1 for p in predictions)
        capsys.readouterr()
        assert (
            main(
                [
                    "evaluate",
                    *common,

                    "--model-name",
                    name,

                    "--device",
                    "cpu",
                    "--no-wandb",
                ]
            )
            == 0
        )

        metrics = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert metrics["sat_policy"] == policy
        assert metrics["model_config"] == expected

    other = "satcop" if policy == "satresetcop" else "satresetcop"
    for command in ("train", "evaluate"):
        assert (
            main(
                [
                    command,
                    *common,

                    "--model-name",
                    "SmallFull",

                    "--policy",
                    other,

                    "--device",
                    "cpu",
                    "--no-wandb",
                ]
            )
            == 2
        )

        assert "dataset labels were collected" in capsys.readouterr().err

def test_input_ablations_preserve_dataset_graph():
    problem = load_tptp_problem(
        "packages/axiom-predictor/tests/fixtures/problems/marked_conjecture_clausification.p"
    )

    graph = build_axiom_graph(
        problem.matrix,
        axiom_clause_ids = problem.axiom_clause_ids,
        conjecture_clause_ids = problem.conjecture_clause_ids,
    ).graph

    original = graph.to_dict()
    no_complements = select_graph_input(graph, "no-complements")
    assert graph.edges["complement"]
    assert not no_complements.edges["complement"]

    no_terms = select_graph_input(graph, "no-terms")
    assert graph.nodes["term"]
    assert not no_terms.nodes["term"] and not no_terms.nodes["var"]
    assert no_terms.edges["lit_sym"] == graph.edges["lit_sym"]
    assert all(
        not no_terms.edges[name] for name in ("atom", "arg_term", "arg_var", "sym")
    )

    assert graph.to_dict() == original

def test_original_checkpoint_keeps_full_input_predictions(tmp_path, tiny_problem_path):
    model = AxiomPredictionNetwork(AxiomModelConfig(hidden_dim = 8))
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, training_config = {})

    payload = torch.load(path, weights_only = True)
    payload["version"] = 2
    del payload["model_name"]
    del payload["model_config"]["graph_input"]
    torch.save(payload, path)

    restored = AxiomPredictor.load(path)
    problem = load_tptp_problem(tiny_problem_path)

    kwargs = {
        "axiom_clause_ids": problem.axiom_clause_ids,
        "conjecture_clause_ids": problem.conjecture_clause_ids,
    }

    assert restored.model.config.graph_input is GraphInputKind.Full
    assert restored.predict(problem.matrix, **kwargs) == AxiomPredictor(model).predict(
        problem.matrix,
        **kwargs,
    )

@pytest.mark.parametrize("policy", ["satcop", "satresetcop"])
def test_fresh_training_requires_cuda_or_explicit_auto(
    tmp_path,
    tiny_problem_path,
    policy,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = [
        "train",
        str(tiny_problem_path),
        "--data-dir",
        str(tmp_path),

        "--policy",
        policy,
        "--network",
        "SmallFull.py",

        "--epochs",
        "1",
        "--num-workers",
        "1",

        "--no-wandb",
    ]

    assert main(args) == 2
    assert "CUDA is unavailable" in capsys.readouterr().err
    assert not (tmp_path / "model" / "model.pt").exists()

    assert main([*args, "--device", "auto"]) == 0
    predictor = AxiomPredictor.load(tmp_path / "model")
    assert type(predictor.model) is load_model_class("SmallFull")
    assert predictor.training_config["sat_policy"] == policy
