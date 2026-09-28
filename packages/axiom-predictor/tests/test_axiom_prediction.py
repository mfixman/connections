from __future__ import annotations

import math

import pytest
from typing import Callable, NoReturn, cast

import torch

from connections.clausification import matrix_from_file

import axiom_prediction.cli as cli_module
import axiom_prediction.training as training_module

from axiom_prediction.graph import build_axiom_graph, collate_axiom_graphs
from axiom_prediction.data import axiom_training_example_from_json, axiom_training_example_to_json, training_example_from_sat_core
from axiom_prediction.metrics import prediction_metrics
from axiom_prediction.cli import main as axiom_predictor_main

from axiom_prediction.model import AxiomModelConfig, AxiomPredictionNetwork, AxiomPredictor
from axiom_prediction.tptp import expand_problem_inputs, find_tptp_root, problem_input_directory, resolve_tptp_problem

fail = cast(Callable[..., NoReturn], pytest.fail)

def fixture_problem(path):
    matrix = matrix_from_file(path, mark_conjecture = True)
    conjectures = tuple(matrix.conjecture_clauses)
    axioms = tuple(index for index in range(len(matrix)) if index not in conjectures)
    return matrix, axioms, conjectures

def test_axiom_training_example_json_is_self_contained(tiny_problem_path):
    matrix, axioms, conjectures = fixture_problem(tiny_problem_path)
    example = training_example_from_sat_core(
        matrix,
        problem_path = str(tiny_problem_path),
        axiom_clause_ids = axioms,
        conjecture_clause_ids = conjectures,
        core_clause_ids = (axioms[0], conjectures[0]),
    )

    restored = axiom_training_example_from_json(axiom_training_example_to_json(example))
    assert restored.problem_path == example.problem_path
    assert restored.matrix is None
    assert restored.graph == example.graph
    assert restored.labels == example.labels
    assert restored.axiom_clause_texts == example.axiom_clause_texts

def test_batched_axiom_logits_match_individual_graphs(tiny_problem_path):
    matrix, axioms, conjectures = fixture_problem(tiny_problem_path)
    example = build_axiom_graph(
        matrix,
        axiom_clause_ids = axioms,
        conjecture_clause_ids = conjectures,
    )

    torch.manual_seed(0)
    model = AxiomPredictionNetwork(
        AxiomModelConfig(hidden_dim = 8, message_rounds = 2, num_hidden_layers = 1)
    )

    model.eval()
    with torch.no_grad():
        individual = model(collate_axiom_graphs([(example, None)]))
        batched = model(collate_axiom_graphs([(example, None), (example, None)]))

    assert torch.allclose(batched[: len(axioms)], individual, atol = 1e-6)
    assert torch.allclose(batched[len(axioms) :], individual, atol = 1e-6)

def test_metrics_cover_ranking_and_degenerate_classes():
    metrics = prediction_metrics([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], problem_sizes = [4])
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx((1.0 + 2 / 3) / 2)

    assert metrics["macro_recall_at_1"] == 0.5
    assert metrics["macro_recall_at_3"] == 1.0

    assert metrics["bce"] is not None
    assert math.isfinite(metrics["bce"])

    no_positives = prediction_metrics([0, 0], [0.2, 0.1])
    assert no_positives["roc_auc"] is None
    assert no_positives["average_precision"] is None

def test_tptp_root_precedence(tmp_path, monkeypatch):
    import axiom_prediction.tptp as tptp

    explicit = tmp_path / "explicit"
    environment = tmp_path / "environment"

    explicit.mkdir()
    environment.mkdir()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TPTP", str(environment))

    assert find_tptp_root(explicit) == explicit.resolve()
    assert find_tptp_root() == environment.resolve()

    discovered = tmp_path / "TPTP"
    (discovered / "Problems").mkdir(parents = True)
    (discovered / "Axioms").mkdir()
    monkeypatch.setattr(tptp, 'default_tptp_roots', lambda: (discovered,))
    monkeypatch.delenv("TPTP")

    assert find_tptp_root() == discovered.resolve()
    monkeypatch.setenv("TPTP", str(tmp_path / "bad-env"))
    with pytest.raises(FileNotFoundError, match = "not a directory"):
        find_tptp_root()

    with pytest.raises(FileNotFoundError, match = "not a directory"):
        find_tptp_root(tmp_path / "missing")

def test_problem_input_detection_modes(tmp_path, monkeypatch):
    category = tmp_path / "Problems" / "ABC"
    nested = category / "nested"
    nested.mkdir(parents = True)

    first = category / "ABC001+1.p"
    second = category / "ABC002-1.p"
    ignored = nested / "ABC003-1.p"

    for path in (first, second, ignored):
        path.write_text("fof(a,conjecture,a).\n", encoding = "utf-8")

    monkeypatch.setenv("TPTP", str(tmp_path))

    assert expand_problem_inputs([str(first)]) == (str(first.resolve()),)
    assert expand_problem_inputs([str(category)]) == (
        str(first.resolve()),
        str(second.resolve()),
    )

    assert expand_problem_inputs([first.name]) == (str(first.resolve()),)
    assert expand_problem_inputs(["ABC"]) == (
        str(first.resolve()),
        str(second.resolve()),
    )

    assert problem_input_directory("ABC") == category.resolve()
    assert resolve_tptp_problem(first.name)[0] == first.resolve()

def test_split_parts_are_disjoint_complete_and_path_independent(tmp_path):
    from axiom_prediction.split import ProblemSplit

    names = [
        f"ABC{index:03d}{sign}{version}.p"
        for index in range(60)
        for sign, version in (("+", 1), ("+", 2), ("-", 1))
    ]

    parts = [ProblemSplit(4, (part,)) for part in range(4)]
    for name in names:
        owners = [part.parts[0] for part in parts if part.contains(name)]
        assert len(owners) == 1
        assert ProblemSplit(4, (owners[0],)).contains(f"/rds/TPTP/Problems/ABC/{name}")
        assert ProblemSplit(4, (owners[0],)).contains(f"Problems/ABC/{name}")

    assert any(
        parts[0].part(f"ABC{index:03d}+1.p") != parts[0].part(f"ABC{index:03d}-1.p")
        for index in range(60)
    )

    assert len({parts[0].part(name) for name in names}) == 4
    assert ProblemSplit(4, (0, 1, 2, 3)).select(names) == tuple(names)
    assert ProblemSplit().select(names) == tuple(names)
    from axiom_prediction.split import split_key

    assert split_key("/elsewhere/ABC001+1.p") == "ABC001+1.p"

def test_split_sizes_partition_independently():
    from axiom_prediction.split import ProblemSplit

    names = [
        f"ABC{index:03d}+{version}.p"
        for index in range(1000)
        for version in range(1, 5)
    ]

    subset = ProblemSplit(40, (0,), "problem").select(names)
    quarters = [
        len(ProblemSplit(4, (part,), "problem").select(subset))
        for part in range(4)
    ]

    assert all(count > len(subset) / 8 for count in quarters)
    assert ProblemSplit(4, (0,)).to_dict()["scheme"] == 2

def test_evaluate_expands_tptp_category(tmp_path, monkeypatch):
    category = tmp_path / "Problems" / "ABC"
    category.mkdir(parents = True)

    first = category / "ABC001+1.p"
    second = category / "ABC002-1.p"
    first.write_text("", encoding = "utf-8")
    second.write_text("", encoding = "utf-8")

    captured = {}

    def fake_evaluate(checkpoint, problems, **kwargs):
        captured["checkpoint"] = checkpoint
        captured["problems"] = problems
        return {"ok": True}

    monkeypatch.setattr(training_module, "evaluate_axiom_predictor", fake_evaluate)
    assert (
        axiom_predictor_main(
            [
                "evaluate",
                "model.pt",
                "ABC",

                "--tptp",
                str(tmp_path),
                "--device",

                "cpu",
                "--no-wandb",
            ]
        )
        == 0
    )

    assert str(captured["checkpoint"]) == "model.pt"
    assert captured["problems"] == [str(first.resolve()), str(second.resolve())]

def test_predict_expands_category_and_skips_unparseable_file(
    tmp_path,
    tiny_problem_path,
    monkeypatch,
):
    category = tmp_path / "Problems" / "ABC"
    category.mkdir(parents = True)
    valid = category / "ABC001+1.p"
    valid.symlink_to(tiny_problem_path.resolve())
    (category / "ABC002^1.p").write_text("thf(a,axiom,a).\n", encoding = "utf-8")
    seen = []

    class FakePredictor:
        def predict(self, matrix, **kwargs):
            seen.append(len(matrix))
            return ()

    monkeypatch.setattr(AxiomPredictor, "load", lambda *args, **kwargs: FakePredictor())
    assert (
        axiom_predictor_main(
            [
                "predict",
                "model.pt",
                "ABC",

                "--tptp",
                str(tmp_path),
                "--device",

                "cpu",
            ]
        )
        == 0
    )

    assert len(seen) == 1

def test_predict_with_data_dir_treats_every_positional_as_a_problem(
    tmp_path,
    monkeypatch,
):
    seen = {}

    class FakePredictor:
        @staticmethod
        def load(checkpoint, device):
            seen["checkpoint"] = checkpoint
            raise FileNotFoundError("stop")

    import axiom_prediction.model as model_module

    monkeypatch.setattr(model_module, "AxiomPredictor", FakePredictor)
    assert (
        cli_module.main(
            [
                "predict",
                "a.p",
                "b.p",

                "--data-dir",
                str(tmp_path),
                "--model-name",

                "m",
                "--device",
                "cpu",
            ]
        )
        == 2
    )

    assert seen["checkpoint"] == tmp_path / "models" / "m"
