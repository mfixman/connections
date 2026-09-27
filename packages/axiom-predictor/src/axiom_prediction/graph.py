from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from connections.syntax.matrix import Matrix
from axiom_prediction.representation.schema import GraphInput, NODE_TYPES, RELATIONS
from axiom_prediction.representation.matrix import matrix_graph


def validate_clause_ids(
    matrix: Matrix,
    *,
    axiom_clause_ids: tuple[int, ...] | list[int],
    conjecture_clause_ids: tuple[int, ...] | list[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    axs = tuple(axiom_clause_ids)
    conjs = tuple(conjecture_clause_ids)
    if not axs:
        raise ValueError("axiom_clause_ids must be nonempty")

    if not conjs:
        raise ValueError("conjecture_clause_ids must be nonempty")

    if len(set(axs)) != len(axs):
        raise ValueError("axiom_clause_ids must not contain duplicates")

    if len(set(conjs)) != len(conjs):
        raise ValueError("conjecture_clause_ids must not contain duplicates")

    invalid = [index for index in (*axs, *conjs) if not 0 <= index < len(matrix)]
    if invalid:
        raise ValueError(f"clause IDs out of range for matrix of size {len(matrix)}: {invalid}")

    overlap = sorted(set(axs).intersection(conjs))
    if overlap:
        raise ValueError(f"axiom and conjecture clause IDs must be disjoint: {overlap}")

    return axs, conjs


@dataclass(frozen=True, slots=True)
class AxiomGraph:
    graph: GraphInput
    axiom_clause_ids: tuple[int, ...]
    conjecture_clause_ids: tuple[int, ...]


def build_axiom_graph(
    matrix: Matrix,
    *,
    axiom_clause_ids: tuple[int, ...] | list[int],
    conjecture_clause_ids: tuple[int, ...] | list[int],
) -> AxiomGraph:
    axs, conjs = validate_clause_ids(
        matrix,
        axiom_clause_ids=axiom_clause_ids,
        conjecture_clause_ids=conjecture_clause_ids,
    )
    base = matrix_graph(matrix, conjs)
    axiom_set = frozenset(axs)
    conjecture_set = frozenset(conjs)
    nodes = {name: [list(row) for row in rows] for name, rows in base.nodes.items()}
    for clause_index, row in enumerate(nodes["clause"]):
        row[1] = 0 if clause_index in axiom_set else (1 if clause_index in conjecture_set else 2)
        row[3] = int(clause_index in conjecture_set)

    graph = GraphInput(
        nodes=nodes,
        edges={name: [list(row) for row in rows] for name, rows in base.edges.items()},
        actions=[],
        preprocessor="axiom_prediction",
        version="1",
    )
    return AxiomGraph(graph, axs, conjs)


@dataclass(frozen=True, slots=True)
class AxiomGraphBatch:
    graph: GraphInput
    axiom_clause_indices: torch.Tensor
    conjecture_clause_indices: torch.Tensor
    axiom_batch: torch.Tensor
    conjecture_batch: torch.Tensor
    labels: torch.Tensor | None
    axiom_counts: tuple[int, ...]


def collate_axiom_graphs(
    examples: list[tuple[AxiomGraph, tuple[float, ...] | None]],
) -> AxiomGraphBatch:
    import torch

    if not examples:
        raise ValueError("cannot collate an empty axiom graph batch")

    nodes = {name: [] for name in NODE_TYPES if name != "goal"}
    edges = {name: [] for name, *_ in RELATIONS if name not in {"instance_of", "parent", "path"}}
    relation_types = {name: (src, dst) for name, src, dst, _ in RELATIONS}
    axiom_indices: list[int] = []
    conjecture_indices: list[int] = []
    axiom_batch: list[int] = []
    conjecture_batch: list[int] = []
    labels: list[float] = []
    label_mode = examples[0][1] is not None
    counts: list[int] = []

    for batch_index, (example, example_labels) in enumerate(examples):
        if (example_labels is not None) != label_mode:
            raise ValueError("a batch cannot mix labelled and unlabelled examples")

        offsets = {name: len(rows) for name, rows in nodes.items()}
        for name in nodes:
            nodes[name].extend(example.graph.nodes.get(name, []))

        for name in edges:
            src_type, dst_type = relation_types[name]
            for row in example.graph.edges.get(name, []):
                shifted = list(row)
                shifted[0] += offsets[src_type]
                shifted[1] += offsets[dst_type]
                edges[name].append(shifted)

        axiom_indices.extend(offsets["clause"] + index for index in example.axiom_clause_ids)
        conjecture_indices.extend(
            offsets["clause"] + index for index in example.conjecture_clause_ids
        )
        axiom_batch.extend([batch_index] * len(example.axiom_clause_ids))
        conjecture_batch.extend([batch_index] * len(example.conjecture_clause_ids))
        counts.append(len(example.axiom_clause_ids))
        if example_labels is not None:
            if len(example_labels) != len(example.axiom_clause_ids):
                raise ValueError("label count must match axiom clause count")

            labels.extend(example_labels)

    return AxiomGraphBatch(
        graph=GraphInput(
            nodes=nodes,
            edges=edges,
            actions=[],
            preprocessor="axiom_prediction",
            version="1",
        ),
        axiom_clause_indices=torch.tensor(axiom_indices, dtype=torch.long),
        conjecture_clause_indices=torch.tensor(conjecture_indices, dtype=torch.long),
        axiom_batch=torch.tensor(axiom_batch, dtype=torch.long),
        conjecture_batch=torch.tensor(conjecture_batch, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.float32) if label_mode else None,
        axiom_counts=tuple(counts),
    )


__all__ = [
    "AxiomGraph",
    "AxiomGraphBatch",
    "build_axiom_graph",
    "collate_axiom_graphs",
    "validate_clause_ids",
]
