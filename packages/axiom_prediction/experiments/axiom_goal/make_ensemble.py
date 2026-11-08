# One TorchScript artifact averaging the logits of several checkpoints (MizAR 60-style averaged ranking).
import argparse

import torch
from torch import nn

class Ensemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, nodes: list[torch.Tensor], edges: list[torch.Tensor], axioms: torch.Tensor, conjectures: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(axioms.shape[0])
        for member in self.members:
            total = total + member(nodes, edges, axioms, conjectures)

        return total / len(self.members)

def main():
    parser = argparse.ArgumentParser(description = 'average the logits of several TorchScript checkpoints')
    parser.add_argument('output')
    parser.add_argument('checkpoints', nargs = '+')
    args = parser.parse_args()
    members = [torch.jit.load(path, map_location = 'cpu') for path in args.checkpoints]
    torch.jit.script(Ensemble(members)).save(args.output)
    print('saved', args.output, len(members), 'members')

main()
