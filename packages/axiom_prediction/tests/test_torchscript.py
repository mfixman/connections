import pytest
import torch

from axiom_prediction.graph import build_axiom_graph, collate_axiom_graphs
from axiom_prediction.model import (
    AxiomModelConfig,
    AxiomPredictionNetwork,
    AxiomPredictor,
    load_checkpoint,
    save_checkpoint,
)
from axiom_prediction.tptp import load_tptp_problem

@pytest.mark.parametrize("kind", ["full", "no-terms", "no-complements"])
@pytest.mark.parametrize("source", [
    "cnf(a,axiom,p(X,X)|p(f(X),a)|~p(Y,f(Y))).\n"
    "cnf(b,axiom,~p(f(Z),Z)|q(f(a))).\ncnf(g,negated_conjecture,~q(f(a))).",
    "cnf(a,axiom,a=b).\ncnf(g,negated_conjecture,a!=b).",
    "cnf(a,axiom,p|q).\ncnf(b,axiom,~p).",
    "cnf(g,negated_conjecture,~p(X)).",
    "cnf(a,axiom,$false).",
])
def test_script_matches_python_and_legacy_checkpoint(tmp_path, kind, source):
    path = tmp_path / "problem.p"
    path.write_text(source)
    problem = load_tptp_problem(str(path))
    graph = build_axiom_graph(
        problem.matrix,
        axiom_clause_ids = problem.axiom_clause_ids,
        conjecture_clause_ids = problem.conjecture_clause_ids,
    )

    config = AxiomModelConfig(hidden_dim = 8, message_rounds = 2, graph_input = kind)
    model = AxiomPredictionNetwork(config).eval()
    target = tmp_path / "model.ts"
    save_checkpoint(target, model, training_config = {"sat_policy": "satresetcop"}, epoch = 7)
    assert not list(tmp_path.glob("*.pt"))

    scripted = torch.jit.load(target)
    node_names = ["symbol", "clause", "literal", "term", "var"]
    edge_names = ["contains", "atom", "arg_term", "arg_var", "sym", "lit_sym", "complement"]
    nodes = [
        torch.tensor(graph.graph.nodes[name], dtype = torch.long).reshape(-1, width)
        for name, width in zip(node_names, [2, 4, 1, 1, 1])
    ]

    edges = [
        torch.tensor(graph.graph.edges[name], dtype = torch.long).reshape(-1, width)
        for name, width in zip(edge_names, [2, 2, 3, 3, 2, 2, 2])
    ]

    batch = collate_axiom_graphs([(graph, None)])
    with torch.no_grad():
        expected = model(batch)
        actual = scripted(nodes, edges, batch.axiom_clause_indices, batch.conjecture_clause_indices)

    torch.testing.assert_close(actual, expected, atol = 2e-6, rtol = 2e-5)
    payload = load_checkpoint(target)
    assert payload["epoch"] == 7
    assert payload["training_config"]["sat_policy"] == "satresetcop"

    legacy = tmp_path / "model.pt"
    torch.save(payload, legacy)
    for path in (target, legacy, tmp_path):
        predictor = AxiomPredictor.load(path, device = "cpu")
        with torch.no_grad():
            torch.testing.assert_close(predictor.model(batch), expected, rtol = 0, atol = 0)

    legacy.write_bytes(b"invalid legacy file")
    assert AxiomPredictor.load(tmp_path).training_config == payload["training_config"]

    with pytest.raises(ValueError, match = ".ts extension"):
        save_checkpoint(legacy, model, training_config = {})

    exported_only = tmp_path / "missing-metadata.ts"
    torch.jit.save(scripted, exported_only)
    with pytest.raises(ValueError, match = "no Connections checkpoint metadata"):
        AxiomPredictor.load(exported_only)
