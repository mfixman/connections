# Tensor-only inference interface shared with Vampire's LibTorch loader.
import copy

import torch
from torch import nn

from .choices import GraphInputKind
from .representation.schema import relations

class Features(nn.Module):
    def __init__(self, tables):
        super().__init__()
        self.tables = tables

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        result = self.tables[0](rows[:, 0])
        for column, table in enumerate(self.tables):
            if column > 0:
                result = result + table(rows[:, column])

        return torch.relu(result)

class Relation(nn.Module):
    __constants__ = ["src", "dst", "positioned", "index", "chunk"]

    def __init__(self, encoder, name, src, dst, positioned, index, chunk = 65536):
        super().__init__()
        self.src, self.dst = src, dst
        self.positioned, self.index = positioned, index
        self.fwd = encoder.relation_fwd[name]
        self.rev = encoder.relation_rev[name]
        self.chunk = chunk

    # Edges go in chunks so memory grows with nodes, not edges; Vampire caps its process memory.
    def forward(
        self,
        h: list[torch.Tensor],
        incoming: list[torch.Tensor],
        edge: torch.Tensor,
        positions: torch.Tensor,
    ):
        forward_sums, reverse_sums = torch.zeros_like(h[self.dst]), torch.zeros_like(h[self.src])
        for start in range(0, edge.shape[0], self.chunk):
            part = edge[start:start + self.chunk]
            source, target = part[:, 0], part[:, 1]
            left, right = h[self.src][source], h[self.dst][target]
            if self.positioned:
                pos = positions[part[:, 2]]
                left, right = left + pos, right + pos

            forward_sums.index_add_(0, target, self.fwd(left))
            reverse_sums.index_add_(0, source, self.rev(right))

        incoming[self.dst] = incoming[self.dst] + forward_sums / self.counts(edge[:, 1], h[self.dst])
        incoming[self.src] = incoming[self.src] + reverse_sums / self.counts(edge[:, 0], h[self.src])

    def counts(self, index: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        counts = torch.zeros(reference.shape[0], device = reference.device)
        counts.index_add_(0, index, torch.ones(index.shape[0], device = reference.device))
        return counts.clamp(min = 1).unsqueeze(1)

class TensorModel(nn.Module):
    __constants__ = ["rounds", "no_terms"]

    def __init__(self, model):
        super().__init__()
        encoder = model.encoder
        names = encoder.matrix_node_types

        self.features = nn.ModuleList(Features(encoder.feature_embeddings[n]) for n in names)
        self.transforms = nn.ModuleList(encoder.self_transform[n] for n in names)
        self.norms = nn.ModuleList(encoder.update_norm[n] for n in names)
        self.no_terms = model.config.graph_input == GraphInputKind.NoTerms
        excluded = set()
        if self.no_terms:
            excluded = {"atom", "arg_term", "arg_var", "sym"}
        elif model.config.graph_input == GraphInputKind.NoComplements:
            excluded = {"complement"}

        self.relations = nn.ModuleList(
            Relation(encoder, name, names.index(src), names.index(dst), positioned, index)
            for index, (name, src, dst, positioned) in enumerate(relations[:7])
            if name not in excluded
        )

        self.positions = encoder.position_embedding
        self.scorer = model.scorer
        self.rounds = encoder.config.message_rounds

    def forward(
        self,
        nodes: list[torch.Tensor],
        edges: list[torch.Tensor],
        axioms: torch.Tensor,
        conjectures: torch.Tensor,
    ) -> torch.Tensor:
        h = torch.jit.annotate(list[torch.Tensor], [])
        for i, feature in enumerate(self.features):
            rows = nodes[i]
            if self.no_terms and i >= 3:
                rows = rows[:0]

            h.append(feature(rows))

        for step in range(self.rounds):
            incoming = [torch.zeros_like(value) for value in h]
            for relation in self.relations:
                relation(h, incoming, edges[relation.index], self.positions.weight)

            updated = torch.jit.annotate(list[torch.Tensor], [])
            for i, transform in enumerate(self.transforms):
                updated.append(h[i] + torch.relu(transform(h[i]) + incoming[i]))

            for i, norm in enumerate(self.norms):
                h[i] = norm(updated[i])

        candidates = h[1][axioms]
        conj = h[1][conjectures].sum(0, keepdim = True) / max(conjectures.numel(), 1)
        context = candidates.sum(0, keepdim = True) / max(axioms.numel(), 1)
        conj, context = conj.expand_as(candidates), context.expand_as(candidates)
        features = torch.cat([candidates, conj, context, candidates * conj, candidates - conj], 1)
        return self.scorer(features).squeeze(1)

def export_model(model):
    return torch.jit.script(copy.deepcopy(TensorModel(model)).cpu().eval())
