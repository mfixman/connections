# Untrained control: an axiom's score is the fraction of its symbols that also occur in the conjectures.
import argparse

import torch
from torch import nn

class SymbolOverlap(nn.Module):
    def forward(self, nodes: list[torch.Tensor], edges: list[torch.Tensor], axioms: torch.Tensor, conjectures: torch.Tensor) -> torch.Tensor:
        clauses, symbols = nodes[1].shape[0], max(nodes[0].shape[0], 1)
        contains, atom, arg_term, sym = edges[0], edges[1], edges[2], edges[4]
        literal_clause = torch.full([max(nodes[2].shape[0], 1)], -1, dtype = torch.long)
        literal_clause[contains[:, 1]] = contains[:, 0]
        pair_clause, pair_term = literal_clause[atom[:, 0]], atom[:, 1]
        all_clause, all_term = [pair_clause], [pair_term]

        order = torch.argsort(arg_term[:, 0])
        parents, children = arg_term[order, 0], arg_term[order, 1]
        terms = max(nodes[3].shape[0], 1)
        counts = torch.bincount(parents, minlength = terms)
        starts = torch.cumsum(counts, 0) - counts

        # Walk from the atoms down to every subterm, keeping (clause, term) pairs.
        while pair_term.numel() > 0:
            per = counts[pair_term]
            pair_clause = torch.repeat_interleave(pair_clause, per)
            offsets = torch.repeat_interleave(starts[pair_term], per)
            first = torch.repeat_interleave(torch.cumsum(per, 0) - per, per)
            pair_term = children[offsets + torch.arange(pair_clause.numel()) - first]
            keys = torch.unique(pair_clause * terms + pair_term)
            pair_clause, pair_term = keys // terms, keys % terms
            all_clause.append(pair_clause)
            all_term.append(pair_term)

        clause_of, term_of = torch.cat(all_clause), torch.cat(all_term)
        term_symbol = torch.full([terms], -1, dtype = torch.long)
        term_symbol[sym[:, 0]] = sym[:, 1]
        keys = torch.unique(clause_of * symbols + term_symbol[term_of])
        pair_clause, pair_symbol = keys // symbols, keys % symbols

        goal = torch.zeros([symbols], dtype = torch.bool)
        is_conjecture = torch.zeros([clauses], dtype = torch.bool)
        is_conjecture[conjectures] = True
        goal[pair_symbol[is_conjecture[pair_clause]]] = True
        shared = torch.zeros([clauses]).index_add_(0, pair_clause, goal[pair_symbol].float())
        total = torch.zeros([clauses]).index_add_(0, pair_clause, torch.ones(pair_clause.shape))

        # Logits, so callers that apply a sigmoid keep the order.
        return torch.logit((shared[axioms] / total[axioms].clamp(min = 1)).clamp(1e-4, 1 - 1e-4))

def main():
    parser = argparse.ArgumentParser(description = 'export the symbol-overlap TorchScript control')
    parser.add_argument('output')
    args = parser.parse_args()
    torch.jit.script(SymbolOverlap()).save(args.output)

main()
