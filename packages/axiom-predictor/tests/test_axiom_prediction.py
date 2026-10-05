from __future__ import annotations

import math

import pytest

import torch
from copy import deepcopy

from connections.clausification import matrix_from_file

from axiom_prediction.graph import build_axiom_graph, collate_axiom_graphs
from axiom_prediction.metrics import prediction_metrics

from axiom_prediction.model import AxiomModelConfig, AxiomPredictionNetwork

def fixture_problem(path):
    matrix = matrix_from_file(path, mark_conjecture = True)
    conjectures = tuple(matrix.conjecture_clauses)
    axioms = tuple(index for index in range(len(matrix)) if index not in conjectures)
    return matrix, axioms, conjectures

def test_batched_axiom_logits_match_individual_graphs(tiny_problem_path, monkeypatch):
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

    dense = model.encoder.double()
    sparse = deepcopy(dense)
    states = {"literal": torch.randn(5, 8, dtype = torch.double, requires_grad = True)}
    edges = torch.tensor([[0, 1], [0, 1], [2, 1], [1, 2]])
    relations = [("complement", "literal", "literal", edges, None)]
    probe = torch.randn(5, 8, dtype = torch.double)
    expected = dense.message_rounds(states, relations)["literal"]
    (expected * probe).sum().backward()
    assert states["literal"].grad is not None
    expected_grad = states["literal"].grad.clone()

    states["literal"].grad = None
    actual = sparse.message_rounds(states, relations, sparse_edge_threshold = 0)["literal"]
    (actual * probe).sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(states["literal"].grad, expected_grad)

    for before, after in zip(dense.parameters(), sparse.parameters(), strict = True):
        if before.grad is not None:
            torch.testing.assert_close(after.grad, before.grad)

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

    for labels in ([1, 0], [0, 1]):
        tied = prediction_metrics(labels, [0.5, 0.5])
        assert tied["average_precision"] == 0.5

    for value in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match = "finite and between"):
            prediction_metrics([1, 0], [value, 0.5])

def test_split_parts_are_disjoint_complete_and_path_independent(tmp_path):
    from axiom_prediction.split import ProblemSplit

    names = [
        f"ABC{index:03d}{sign}{version}.p"
        for index in range(60)
        for sign, version in (("+", 1), ("+", 2), ("-", 1))
    ]

    parts = [ProblemSplit(4, [part]) for part in range(4)]
    for name in names:
        owners = [part.parts[0] for part in parts if part.contains(name)]
        assert len(owners) == 1
        assert ProblemSplit(4, [owners[0]]).contains(f"/rds/TPTP/Problems/ABC/{name}")
        assert ProblemSplit(4, [owners[0]]).contains(f"Problems/ABC/{name}")

    assert all(
        parts[0].part(f"ABC{index:03d}+1.p") == parts[0].part(f"ABC{index:03d}-1.p")
        for index in range(60)
    )

    assert len({parts[0].part(name) for name in names}) == 4
    assert ProblemSplit(4, [0, 1, 2, 3]).select(names) == names
    assert ProblemSplit().select(names) == names
    from axiom_prediction.split import split_key

    assert split_key("/elsewhere/ABC001+1.p") == "ABC001"
    assert split_key("ABC001-2.p") == "ABC001"
    assert split_key("ABC002+1.p") == "ABC002"
    assert split_key("tiny_theorem.p") == "tiny_theorem"
