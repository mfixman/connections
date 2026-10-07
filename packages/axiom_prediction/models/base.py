from dataclasses import replace

import torch
from torch import nn

from ..configuration import AxiomModelConfig
from ..encoder import GraphModelConfig, GraphNetwork
from ..graph import AxiomGraphBatch
from ..inputs import select_graph_input

class AxiomPredictionNetwork(nn.Module):
    default_config = AxiomModelConfig()

    def __init__(self, config: AxiomModelConfig | None = None):
        super().__init__()
        config = replace(self.default_config) if config is None else config
        self.config = config
        self.encoder = GraphNetwork(
            GraphModelConfig(
                hidden_dim = config.hidden_dim,
                message_rounds = config.message_rounds,
                num_hidden_layers = config.num_hidden_layers,
                activation = config.activation,
            )
        )

        dim = config.hidden_dim
        layers: list[nn.Module] = []
        input_dim = dim * 5
        for _ in range(config.num_hidden_layers):
            layers.extend((nn.Linear(input_dim, dim), nn.ReLU()))
            input_dim = dim

        layers.append(nn.Linear(input_dim, 1))
        self.scorer = nn.Sequential(*layers)

    def network_size(self) -> int:
        return sum(parameter.numel() for parameter in self.scorer.parameters())

    def forward(self, batch: AxiomGraphBatch) -> torch.Tensor:
        graph = select_graph_input(batch.graph, self.config.graph_input)
        encoded = self.encoder.encode_matrix(graph)
        clauses = encoded["clause"]

        axiom_indices = batch.axiom_clause_indices.to(clauses.device)
        conjecture_indices = batch.conjecture_clause_indices.to(clauses.device)
        axiom_batch = batch.axiom_batch.to(clauses.device)
        conjecture_batch = batch.conjecture_batch.to(clauses.device)
        batch_size = len(batch.axiom_counts)

        conjecture_context = pool_mean(
            clauses[conjecture_indices],
            conjecture_batch,
            batch_size,
        )

        axiom_context = pool_mean(clauses[axiom_indices], axiom_batch, batch_size)
        cand = clauses[axiom_indices]

        conj = conjecture_context[axiom_batch]
        all_ax = axiom_context[axiom_batch]
        x = torch.cat((cand, conj, all_ax, cand * conj, cand - conj), dim = 1)

        return self.scorer(x).squeeze(1)

def pool_mean(values: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    output = torch.zeros((size, values.shape[1]), device = values.device)
    output.index_add_(0, batch, values)
    counts = torch.zeros(size, device = values.device)
    counts.index_add_(0, batch, torch.ones(len(batch), device = values.device))
    return output / counts.clamp(min = 1).unsqueeze(1)
