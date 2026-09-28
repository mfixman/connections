import json

import pytest
import torch

from axiom_prediction.cli import main
from axiom_prediction.graph import build_axiom_graph
from axiom_prediction.inputs import GRAPH_INPUTS, select_graph_input
from axiom_prediction.model import (
    MODEL_PRESETS,
    AxiomModelConfig,
    AxiomPredictionNetwork,
    AxiomPredictor,
    save_checkpoint,
)
from axiom_prediction.tptp import load_tptp_problem


@pytest.mark.parametrize("policy", ["satcop", "satresetcop"])
def test_policy_and_network_experiments(tmp_path, policy, capsys):
    problem = "packages/axiom-predictor/tests/fixtures/problems/marked_conjecture_clausification.p"
    common = ["--data-dir", str(tmp_path), "--num-workers", "1"]
    assert main(["collect", problem, *common, "--sat-policy", policy]) == 0
    loaded = load_tptp_problem(problem)
    for size, preset in MODEL_PRESETS.items():
        for graph_input in GRAPH_INPUTS:
            name = f"{size}-{graph_input}"
            assert main([
                "train", *common, "--sat-policy", policy, "--model-name", name,
                "--model-preset", size, "--graph-input", graph_input,
                "--epochs", "1", "--device", "cpu", "--no-wandb",
            ]) == 0
            directory = tmp_path / "models" / name
            predictor = AxiomPredictor.load(directory)
            expected = {**preset.to_dict(), "graph_input": graph_input}
            assert predictor.model.config.to_dict() == expected
            assert predictor.training_config["sat_policy"] == policy
            checkpoint = torch.load(directory / "model.pt", weights_only=True)
            assert checkpoint["epoch"] == 1
            assert checkpoint["model_config"] == expected
            predictions = predictor.predict(
                loaded.matrix, axiom_clause_ids=loaded.axiom_clause_ids,
                conjecture_clause_ids=loaded.conjecture_clause_ids,
            )
            assert len(predictions) == len(loaded.axiom_clause_ids)
            assert all(0 <= p.probability <= 1 for p in predictions)
            capsys.readouterr()
            assert main([
                "evaluate", *common, "--model-name", name, "--device", "cpu", "--no-wandb",
            ]) == 0
            metrics = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
            assert metrics["sat_policy"] == policy
            assert metrics["model_config"] == expected

    other = "satcop" if policy == "satresetcop" else "satresetcop"
    for command in ("train", "evaluate"):
        assert main([
            command, *common, "--model-name", "small-full", "--sat-policy", other,
            "--device", "cpu", "--no-wandb",
        ]) == 2
        assert "dataset labels were collected" in capsys.readouterr().err


@pytest.mark.parametrize("policy", ["satcop", "satresetcop"])
def test_fresh_training_policy_and_model_overrides(tmp_path, tiny_problem_path, policy):
    assert main([
        "train", str(tiny_problem_path), "--data-dir", str(tmp_path),
        "--sat-policy", policy, "--model-preset", "large", "--hidden-dim", "8",
        "--message-rounds", "0", "--num-hidden-layers", "0",
        "--epochs", "1", "--learning-rate", "0.002", "--weight-decay", "0.01",
        "--device", "cpu", "--num-workers", "1", "--no-wandb",
    ]) == 0
    predictor = AxiomPredictor.load(tmp_path / "model")
    assert predictor.model.config == AxiomModelConfig(8, 0, 0)
    assert predictor.training_config["sat_policy"] == policy
    assert predictor.training_config["learning_rate"] == 0.002
    assert predictor.training_config["weight_decay"] == 0.01


def test_input_ablations_preserve_dataset_graph():
    problem = load_tptp_problem(
        "packages/axiom-predictor/tests/fixtures/problems/marked_conjecture_clausification.p"
    )
    graph = build_axiom_graph(
        problem.matrix, axiom_clause_ids=problem.axiom_clause_ids,
        conjecture_clause_ids=problem.conjecture_clause_ids,
    ).graph
    original = graph.to_dict()
    no_complements = select_graph_input(graph, "no-complements")
    assert graph.edges["complement"]
    assert not no_complements.edges["complement"]
    no_terms = select_graph_input(graph, "no-terms")
    assert graph.nodes["term"]
    assert not no_terms.nodes["term"] and not no_terms.nodes["var"]
    assert no_terms.edges["lit_sym"] == graph.edges["lit_sym"]
    assert all(not no_terms.edges[name] for name in ("atom", "arg_term", "arg_var", "sym"))
    assert graph.to_dict() == original


def test_original_checkpoint_keeps_full_input_predictions(tmp_path, tiny_problem_path):
    model = AxiomPredictionNetwork(AxiomModelConfig(hidden_dim=8))
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, training_config={})
    payload = torch.load(path, weights_only=True)
    del payload["model_config"]["graph_input"]
    torch.save(payload, path)
    restored = AxiomPredictor.load(path)
    problem = load_tptp_problem(tiny_problem_path)
    kwargs = {
        "axiom_clause_ids": problem.axiom_clause_ids,
        "conjecture_clause_ids": problem.conjecture_clause_ids,
    }
    assert restored.model.config.graph_input == "full"
    assert restored.predict(problem.matrix, **kwargs) == AxiomPredictor(model).predict(
        problem.matrix, **kwargs
    )
