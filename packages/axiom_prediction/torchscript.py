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
    __constants__ = ["src", "dst", "positioned", "index"]

    def __init__(self, encoder, name, src, dst, positioned, index):
        super().__init__()
        self.src, self.dst = src, dst
        self.positioned, self.index = positioned, index
        self.fwd = encoder.relation_fwd[name]
        self.rev = encoder.relation_rev[name]

    def forward(
        self,
        h: list[torch.Tensor],
        incoming: list[torch.Tensor],
        edge: torch.Tensor,
        positions: torch.Tensor,
    ):
        source, target = edge[:, 0], edge[:, 1]
        left, right = h[self.src][source], h[self.dst][target]
        if self.positioned:
            pos = positions[edge[:, 2]]
            left, right = left + pos, right + pos

        incoming[self.dst] = incoming[self.dst] + self.mean(self.fwd(left), target, h[self.dst])
        incoming[self.src] = incoming[self.src] + self.mean(self.rev(right), source, h[self.src])

    def mean(self, values: torch.Tensor, index: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        sums = torch.zeros_like(reference).index_add_(0, index, values)
        counts = torch.zeros(reference.shape[0], device = reference.device)
        counts.index_add_(0, index, torch.ones(index.shape[0], device = reference.device))
        return sums / counts.clamp(min = 1).unsqueeze(1)

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
