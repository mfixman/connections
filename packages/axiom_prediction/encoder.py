# Typed message passing over the matrix graph with plain PyTorch.

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import nn

from .aggregation import sparse_relation_mean, sparse_relations

from .representation.schema import GraphInput, arg_position_buckets, node_types, relations

@dataclass(slots = True, init = False)
class GraphModelConfig:
    hidden_dim: int
    message_rounds: int
    num_hidden_layers: int
    activation: str

    def __init__(
        self,
        hidden_dim: int = 64,
        message_rounds: int = 3,
        num_hidden_layers: int = 2,
        activation: str = "tanh",
    ):
        self.hidden_dim = hidden_dim
        self.message_rounds = message_rounds
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation

        self.validate()

    def validate(self):
        if self.activation not in {"tanh", "relu"}:
            raise ValueError(f"unsupported graph activation: {self.activation!r}")

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

class GraphNetwork(nn.Module):
    matrix_node_types: list[str] = ["symbol", "clause", "literal", "term", "var"]

    def __init__(self, config: GraphModelConfig):
        action_kinds: list[str] = [
            "start",
            "extension",
            "reduction",
            "factorization",
            "backtrack",
        ]

        node_feature_sizes: dict[str, list[int]] = {
            # kind (predicate/function/constant), arity bucket
            "symbol": [3, 5],
            # size bucket, role (axiom/conjecture/other), is_ground, is_start
            "clause": [5, 3, 2, 2],

            # polarity
            "literal": [2],
            # is_ground
            "term": [2],

            # single dummy category: identity comes from edges alone
            "var": [1],
            # is_open, depth bucket
            "goal": [2, 8],
        }

        super().__init__()
        self.config = config
        dim = config.hidden_dim
        self.activation = nn.Tanh() if config.activation == "tanh" else nn.ReLU()
        self.feature_embeddings = nn.ModuleDict(
            {
                node_type: nn.ModuleList( nn.Embedding(size, dim) for size in node_feature_sizes[node_type] )
                for node_type in node_types
            }
        )

        self.position_embedding = nn.Embedding(arg_position_buckets, dim)
        self.relation_fwd = nn.ModuleDict(
            {name: nn.Linear(dim, dim) for name, _, _, _ in relations}
        )

        self.relation_rev = nn.ModuleDict(
            {name: nn.Linear(dim, dim) for name, _, _, _ in relations}
        )

        self.self_transform = nn.ModuleDict(
            {node_type: nn.Linear(dim, dim) for node_type in node_types}
        )
        self.update_norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(dim) for node_type in node_types}
        )

        # The goal nodes, tableau relations and action scorer below are never used: they come from the tableau
        # policy network this encoder was taken from, and stay so that saved checkpoints and optimizer states load.
        self.kind_embedding = nn.Embedding(len(action_kinds), dim)
        self.missing_target = nn.Parameter(torch.zeros(dim))
        layers: list[nn.Module] = []
        input_dim = dim * 3
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.Tanh() if config.activation == "tanh" else nn.ReLU())
            input_dim = dim

        layers.append(nn.Linear(input_dim, 1))
        self.scorer = nn.Sequential(*layers)

    def embed_features(
        self,
        node_type: str,
        rows: list[list[int]] | torch.Tensor,
    ) -> torch.Tensor:
        dim = self.config.hidden_dim
        device = self.missing_target.device
        if isinstance(rows, torch.Tensor):
            if rows.shape[0] == 0:
                return torch.zeros((0, dim), device = device)

            features = rows.to(device)
        elif not rows:
            return torch.zeros((0, dim), device = device)
        else:
            features = torch.tensor(rows, dtype = torch.long, device = device)

        embedded = torch.zeros((features.shape[0], dim), device = device)
        for column, table in enumerate(
            cast(nn.ModuleList, self.feature_embeddings[node_type])
        ):
            embedded = embedded + table(features[:, column])

        return self.activation(embedded)

    def edge_tensors(
        self,
        graph: GraphInput,
        names: list[str],
    ) -> list[tuple[str, str, str, torch.Tensor, torch.Tensor | None]]:
        by_name = {name: (src, dst, pos) for name, src, dst, pos in relations}
        tensors = []
        for name in names:
            rows = graph.edges.get(name, [])
            if isinstance(rows, torch.Tensor):
                if rows.shape[0] == 0:
                    continue
            elif not rows:
                continue

            src_type, dst_type, has_position = by_name[name]
            tensor = (
                rows.to(self.missing_target.device)
                if isinstance(rows, torch.Tensor)
                else torch.tensor(rows, dtype = torch.long, device = self.missing_target.device)
            )

            position = tensor[:, 2] if has_position else None
            tensors.append((name, src_type, dst_type, tensor, position))

        return tensors

    def message_rounds(
        self,
        h: dict[str, torch.Tensor],
        edge_tensors: list[tuple[str, str, str, torch.Tensor, torch.Tensor | None]],
        sparse_edge_threshold: int = 65_536,
    ) -> dict[str, torch.Tensor]:
        sparse = {
            name: sparse_relations(tensor, h[src], h[dst])
            for name, src, dst, tensor, position in edge_tensors
            if position is None and len(tensor) >= sparse_edge_threshold
        }

        for _ in range(self.config.message_rounds):
            incoming = {node_type: torch.zeros_like(values) for node_type, values in h.items()}
            for name, src_type, dst_type, tensor, position in edge_tensors:
                if name in sparse:
                    forward, reverse = sparse[name]
                    incoming[dst_type] += sparse_relation_mean(h[src_type], self.relation_fwd[name], forward)
                    incoming[src_type] += sparse_relation_mean(h[dst_type], self.relation_rev[name], reverse)
                    continue

                src_h = h[src_type][tensor[:, 0]]
                dst_h = h[dst_type][tensor[:, 1]]
                if position is not None:
                    src_h = src_h + self.position_embedding(position)
                    dst_h = dst_h + self.position_embedding(position)

                scatter_mean(incoming[dst_type], tensor[:, 1], self.relation_fwd[name](src_h))
                scatter_mean(incoming[src_type], tensor[:, 0], self.relation_rev[name](dst_h))

            # Residuals and LayerNorm stabilize training on deep graphs.
            h = {
                node_type: self.update_norm[node_type](
                    values + self.activation(self.self_transform[node_type](values) + incoming[node_type])
                )
                for node_type, values in h.items()
            }

        return h

    def encode_matrix(self, graph: GraphInput) -> dict[str, torch.Tensor]:
        matrix_relations: list[str] = [
            "contains",
            "atom",
            "arg_term",
            "arg_var",

            "sym",
            "lit_sym",
            "complement",
        ]

        h = {
            node_type: self.embed_features(node_type, graph.nodes.get(node_type, []))
            for node_type in self.matrix_node_types
        }

        return self.message_rounds(h, self.edge_tensors(graph, matrix_relations))

def scatter_mean(output: torch.Tensor, index: torch.Tensor, values: torch.Tensor):
    counts = torch.zeros(output.shape[0], device = output.device)
    counts.index_add_(0, index, torch.ones(index.shape[0], device = output.device))
    summed = torch.zeros_like(output)
    summed.index_add_(0, index, values)
    output += summed / counts.clamp(min = 1.0).unsqueeze(1)
