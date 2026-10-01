from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from connections.syntax.matrix import Matrix
from axiom_prediction.representation.schema import GraphInput

from .graph import AxiomGraph, build_axiom_graph

def sat_core_clause_ids(diagnostics: object, *, matrix_size: int) -> list[int]:
    if not isinstance(diagnostics, Mapping) or "sat_core_clause_ids" not in diagnostics:
        raise ValueError("SAT result is missing SAT-core clause provenance")

    value = cast(Mapping[str, Any], diagnostics)["sat_core_clause_ids"]
    if not isinstance(value, list) or any(type(index) is not int for index in value):
        raise ValueError("SAT-core clause provenance must be a list of integers")

    ids = list(cast(list[int], value))
    if ids != sorted(set(ids)):
        raise ValueError("SAT-core clause provenance must be sorted and unique")

    if any(index < 0 or index >= matrix_size for index in ids):
        raise ValueError("SAT-core clause provenance contains an out-of-range index")

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

AXIOM_EXAMPLE_SCHEMA = "learncop.axiom_prediction.example.v2"

def axiom_training_example_to_json(example: AxiomTrainingExample) -> dict[str, Any]:
    texts = example.axiom_clause_texts
    if not texts and example.matrix is not None:
        texts = [
            str(example.matrix.clauses[index])
            for index in example.graph.axiom_clause_ids
        ]

    return {
        "schema": AXIOM_EXAMPLE_SCHEMA,
        "problem_path": example.problem_path,

        "graph": example.graph.graph.to_dict(),
        "axiom_clause_ids": list(example.graph.axiom_clause_ids),
        "conjecture_clause_ids": list(example.graph.conjecture_clause_ids),

        "labels": list(example.labels),
        "axiom_clause_texts": list(texts),
    }

def axiom_training_example_from_json(payload: Mapping[str, Any]) -> AxiomTrainingExample:
    if payload.get("schema") != AXIOM_EXAMPLE_SCHEMA:
        raise ValueError(f"unsupported axiom-example schema: {payload.get('schema')!r}")

    raw_graph = payload.get("graph")
    if not isinstance(raw_graph, Mapping):
        raise TypeError("axiom-example graph must be an object")

    graph_input = GraphInput.from_dict(dict(raw_graph))
    if graph_input.preprocessor != "axiom_prediction" or graph_input.version != "1":
        raise ValueError(
            "unsupported axiom graph format: "
            f"{graph_input.preprocessor!r} version {graph_input.version!r}"
        )

    axiom_ids = integer_list(payload.get("axiom_clause_ids"), "axiom_clause_ids")
    conjecture_ids = integer_list(
        payload.get("conjecture_clause_ids"),
        "conjecture_clause_ids",
    )

    labels = number_list(payload.get("labels"), "labels")
    if any(label not in (0.0, 1.0) for label in labels):
        raise ValueError("labels must contain only 0 or 1")

    raw_texts = payload.get("axiom_clause_texts", [])
    if not isinstance(raw_texts, list) or any(
        not isinstance(value, str) for value in raw_texts
    ):
        raise TypeError("axiom_clause_texts must be a list of strings")

    texts = list(raw_texts)
    if len(labels) != len(axiom_ids):
        raise ValueError("label count must match axiom clause count")

    if texts and len(texts) != len(axiom_ids):
        raise ValueError("axiom clause text count must match axiom clause count")

    clause_count = len(graph_input.nodes.get("clause", []))
    if any(index < 0 or index >= clause_count for index in (*axiom_ids, *conjecture_ids)):
        raise ValueError("axiom-example clause ID is out of range")

    if set(axiom_ids).intersection(conjecture_ids):
        raise ValueError("axiom and conjecture clause IDs must be disjoint")

    return AxiomTrainingExample(
        problem_path = str(payload["problem_path"]),
        matrix = None,
        graph = AxiomGraph(graph_input, axiom_ids, conjecture_ids),
        labels = labels,
        axiom_clause_texts = texts,
    )

def integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise TypeError(f"{name} must be a list of integers")

    result = list(cast(list[int], value))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")

    return result

def number_list(value: object, name: str) -> list[float]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int | float) for item in value
    ):
        raise TypeError(f"{name} must be a list of numbers")

    return [float(item) for item in cast(list[int | float], value)]

__all__ = [
    "AxiomTrainingExample",
    "AXIOM_EXAMPLE_SCHEMA",
    "axiom_training_example_from_json",
    "axiom_training_example_to_json",
    "sat_core_clause_ids",
    "training_example_from_sat_core",
]
