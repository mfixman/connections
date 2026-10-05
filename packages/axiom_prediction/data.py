from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from connections.syntax.matrix import Matrix
from axiom_prediction.representation.schema import GraphInput

from .graph import AxiomGraph, build_axiom_graph

def sat_core_clause_ids(diagnostics: Mapping[str, Any]) -> list[int]:
    if not isinstance(diagnostics, Mapping) or "sat_core_clause_ids" not in diagnostics:
        raise ValueError("SAT result is missing SAT-core clause provenance")

    ids = list(diagnostics["sat_core_clause_ids"])
    return ids

@dataclass(frozen = True, slots = True)
class AxiomTrainingExample:
    problem_path: str
    matrix: Matrix | None
    graph: AxiomGraph
    labels: list[float]
    axiom_clause_texts: list[str] = field(default_factory = list)

def training_example_from_sat_core(
    matrix: Matrix,
    *,
    problem_path: str,
    axiom_clause_ids: list[int],
    conjecture_clause_ids: list[int],
    core_clause_ids: list[int],
) -> AxiomTrainingExample:
    graph = build_axiom_graph(
        matrix,
        axiom_clause_ids = axiom_clause_ids,
        conjecture_clause_ids = conjecture_clause_ids,
    )

    return AxiomTrainingExample(
        problem_path = problem_path,
        matrix = matrix,
        graph = graph,
        labels = [float(index in core_clause_ids) for index in axiom_clause_ids],
        axiom_clause_texts = [str(matrix.clauses[index]) for index in axiom_clause_ids],
    )

axiom_example_schema = "learncop.axiom_prediction.example.v2"

def axiom_training_example_to_json(example: AxiomTrainingExample) -> dict[str, Any]:
    texts = example.axiom_clause_texts
    if not texts and example.matrix is not None:
        texts = [
            str(example.matrix.clauses[index])
            for index in example.graph.axiom_clause_ids
        ]

    return {
        "schema": axiom_example_schema,
        "problem_path": example.problem_path,

        "graph": example.graph.graph.to_dict(),
        "axiom_clause_ids": list(example.graph.axiom_clause_ids),
        "conjecture_clause_ids": list(example.graph.conjecture_clause_ids),

        "labels": list(example.labels),
        "axiom_clause_texts": list(texts),
    }

def axiom_training_example_from_json(payload: Mapping[str, Any]) -> AxiomTrainingExample:
    if payload.get("schema") != axiom_example_schema:
        raise ValueError(f"unsupported axiom-example schema: {payload.get('schema')!r}")

    graph_input = GraphInput.from_dict(payload["graph"])
    if graph_input.preprocessor != "axiom_prediction" or graph_input.version != "1":
        raise ValueError(
            "unsupported axiom graph format: "
            f"{graph_input.preprocessor!r} version {graph_input.version!r}"
        )

    axiom_ids = list(payload["axiom_clause_ids"])
    conjecture_ids = list(payload["conjecture_clause_ids"])
    labels = list(map(float, payload["labels"]))
    texts = list(payload.get("axiom_clause_texts", []))

    if len(labels) != len(axiom_ids):
        raise ValueError("label count must match axiom clause count")

    if texts and len(texts) != len(axiom_ids):
        raise ValueError("axiom clause text count must match axiom clause count")

    return AxiomTrainingExample(
        problem_path = str(payload["problem_path"]),
        matrix = None,
        graph = AxiomGraph(graph_input, axiom_ids, conjecture_ids),
        labels = labels,
        axiom_clause_texts = texts,
    )
