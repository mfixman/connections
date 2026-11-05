from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from connections.syntax.matrix import Matrix
from axiom_prediction.representation.schema import GraphInput, node_types, relations
from axiom_prediction.representation.matrix import matrix_graph

@dataclass(frozen = True, slots = True)
class AxiomGraph:
    graph: GraphInput
    axiom_clause_ids: list[int]
    conjecture_clause_ids: list[int]

def build_axiom_graph(
    matrix: Matrix,
    *,
    axiom_clause_ids: tuple[int, ...] | list[int],
    conjecture_clause_ids: tuple[int, ...] | list[int],
) -> AxiomGraph:
    axs = list(axiom_clause_ids)
    conjs = list(conjecture_clause_ids)

    base = matrix_graph(matrix, conjs)
    axiom_set = frozenset(axs)
    conjecture_set = frozenset(conjs)
    nodes = {name: list(map(list, rows)) for name, rows in base.nodes.items()}
    for clause_index, row in enumerate(nodes["clause"]):
        if clause_index in axiom_set:
            row[1] = 0
        elif clause_index in conjecture_set:
            row[1] = 1
        else:
            row[1] = 2

        row[3] = int(clause_index in conjecture_set)

    graph = GraphInput(
        nodes = nodes,
        edges = {name: list(map(list, rows)) for name, rows in base.edges.items()},
        actions = [],
        preprocessor = "axiom_prediction",
        version = "1",
    )

    return AxiomGraph(graph, axs, conjs)

@dataclass(frozen = True, slots = True)
class AxiomGraphBatch:
    graph: GraphInput
    axiom_clause_indices: Any
    conjecture_clause_indices: Any

    axiom_batch: Any
    conjecture_batch: Any

    labels: Any | None
    axiom_counts: list[int]

def stacked_rows(tables):
    import numpy as np
    import torch

    if not tables:
        return torch.zeros((0,), dtype = torch.long)

    return torch.from_numpy(np.concatenate(tables))

def collate_axiom_graphs(
    examples: list[tuple[AxiomGraph, list[float] | None]]
) -> AxiomGraphBatch:
    import numpy as np
    import torch

    if not examples:
        raise ValueError("cannot collate an empty axiom graph batch")

    nodes = {name: [] for name in node_types if name != "goal"}
    edges = {
        name: []
        for name, *_ in relations
        if name not in {"instance_of", "parent", "path"}
    }

    relation_types = {name: (src, dst) for name, src, dst, _ in relations}
    offsets = dict.fromkeys(nodes, 0)

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

        for name, tables in edges.items():
            src_type, dst_type = relation_types[name]
            rows = np.array(example.graph.edges.get(name, []), dtype = np.int64)
            if rows.size:
                rows[:, 0] += offsets[src_type]
                rows[:, 1] += offsets[dst_type]
                tables.append(rows)

        axiom_indices.extend(
            offsets["clause"] + index for index in example.axiom_clause_ids
        )

        conjecture_indices.extend(
            offsets["clause"] + index for index in example.conjecture_clause_ids
        )

        for name, tables in nodes.items():
            rows = np.array(example.graph.nodes.get(name, []), dtype = np.int64)
            if rows.size:
                tables.append(rows)
                offsets[name] += len(rows)

        axiom_batch.extend([batch_index] * len(example.axiom_clause_ids))
        conjecture_batch.extend([batch_index] * len(example.conjecture_clause_ids))
        counts.append(len(example.axiom_clause_ids))
        if example_labels is not None:
            if len(example_labels) != len(example.axiom_clause_ids):
                raise ValueError("label count must match axiom clause count")

            labels.extend(example_labels)

    return AxiomGraphBatch(
        graph = GraphInput(
            nodes = {name: stacked_rows(tables) for name, tables in nodes.items()},
            edges = {name: stacked_rows(tables) for name, tables in edges.items()},
            actions = [],
            preprocessor = "axiom_prediction",
            version = "1",
        ),
        axiom_clause_indices = torch.tensor(axiom_indices, dtype = torch.long),
        conjecture_clause_indices = torch.tensor(conjecture_indices, dtype = torch.long),

        axiom_batch = torch.tensor(axiom_batch, dtype = torch.long),
        conjecture_batch = torch.tensor(conjecture_batch, dtype = torch.long),

        labels = torch.tensor(labels, dtype = torch.float32) if label_mode else None,
        axiom_counts = counts,
    )
