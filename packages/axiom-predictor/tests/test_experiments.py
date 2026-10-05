import json

import pytest
import torch

from axiom_prediction.cli import main
from axiom_prediction.model import AxiomPredictor
from axiom_prediction.tptp import load_tptp_problem
from axiom_prediction.models import load_model_class

from axiom_prediction.choices import GraphInputKind, ProverPolicy

@pytest.mark.parametrize("policy", ["satcop", "satresetcop"])
def test_policy_and_network_experiments(tmp_path, policy, capsys, monkeypatch):
    problem = "packages/axiom-predictor/tests/fixtures/problems/marked_conjecture_clausification.p"
    common = ["--data-dir", str(tmp_path), "--num-workers", "1"]
    assert main(
        ["collect", problem, *common, "--policy", ProverPolicy(policy).value]
    ) == 0

    loaded = load_tptp_problem(problem)
    for name in ("SmallFull", "SmallNoTerms", "SmallNoComplements"):
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
        torch_load = torch.load
        locations = []

        def tracked_load(*args, **kwargs):
            locations.append(kwargs.get("map_location"))
            return torch_load(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch, "load", tracked_load)
            predictor = AxiomPredictor.load(directory)

        assert locations == ["cpu"]
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

        metrics = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert metrics["sat_policy"] == policy
        assert metrics["model_config"] == expected

    other = "satcop" if policy == "satresetcop" else "satresetcop"
    for command in ("train", "evaluate"):
        assert (
            main(
                [
                        command,
                        *(["--resume"] if command == "train" else []),
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
