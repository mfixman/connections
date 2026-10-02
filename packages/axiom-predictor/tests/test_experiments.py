import json

import pytest
import torch

from axiom_prediction.cli import main
from axiom_prediction.graph import build_axiom_graph
from axiom_prediction.inputs import select_graph_input
from axiom_prediction.model import AxiomModelConfig, AxiomPredictionNetwork, AxiomPredictor, save_checkpoint
from axiom_prediction.tptp import load_tptp_problem
from axiom_prediction.models import load_model_class

from axiom_prediction.choices import GraphInputKind, ProverPolicy

def test_shared_dataset_collection_and_run_directories(
    tmp_path,
    tiny_problem_path,
    monkeypatch,
    capsys,
):
    import axiom_prediction.dataset as dataset_module

    shared = tmp_path / "shared"
    inputs = []
    for name in ("First", "Second"):
        directory = tmp_path / name
        directory.mkdir()
        problem = directory / f"{name}.p"
        problem.write_text(tiny_problem_path.read_text())
        inputs.append(str(directory))

    common = ["--dataset", str(shared), "--num-workers", "1"]
    requests = [[item] for item in inputs]
    for problems in requests:
        assert main(["collect", *problems, *common]) == 0

    expected = {"First.jsonl", "Second.jsonl"}
    assert {path.name for path in shared.glob("*.jsonl")} == expected

    def unexpected_collection(*args, **kwargs):
        raise AssertionError("a reused dataset must not re-prove problems")

    monkeypatch.setattr(dataset_module, "collect_proof_example", unexpected_collection)
    unused = tmp_path / "unused"
    for problems in requests:
        assert main(["collect", *problems, *common, "--data-dir", str(unused)]) == 0

    assert not unused.exists()
    before = {path: path.read_bytes() for path in shared.rglob("*") if path.is_file()}
    cpu = ["--device", "cpu", "--no-wandb"]
    for network in ("SmallFull", "SmallNoTerms"):
        output = tmp_path / network
        run = ["--data-dir", str(output), *common, *cpu]
        assert main(["train", *run, "--network", network, "--epochs", "1"]) == 0

        assert (output / "model" / "model.pt").is_file()
        assert not (output / "dataset").exists()
        assert main(["evaluate", *run]) == 0

        checkpoint = str(output / "model")
        assert main(["evaluate", checkpoint, *common, *cpu]) == 0
        assert main(["train", *run, "--resume", "--policy", "SatCoP"]) == 2
        assert "dataset labels were collected" in capsys.readouterr().err

        assert main(["train", inputs[0], *run]) == 2
        assert main(["evaluate", checkpoint, inputs[0], *common, *cpu]) == 2

    after = {path: path.read_bytes() for path in shared.rglob("*") if path.is_file()}
    assert after == before
    assert main(["collect", inputs[0], "--num-workers", "1"]) == 2

    shard = shared / "Second.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    rows[0]["collection"]["tptp_root"] = "/different/machine/TPTP"
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert len(dataset_module.load_axiom_dataset(shared)[0]) == 2

    rows[0]["collection"]["sat_policy"] = "satcop"
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match = "different collection settings"):
        dataset_module.load_axiom_dataset(shared)

    (shared / "examples").mkdir()
    with pytest.raises(ValueError, match = "mixed dataset formats"):
        dataset_module.load_axiom_dataset(shared)

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
