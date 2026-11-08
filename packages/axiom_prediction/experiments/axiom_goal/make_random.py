# Exports a model with the Sharbaraz interface that returns random logits: a control for the learned ranking.
import argparse
from typing import List

import torch

class RandomScores(torch.nn.Module):
    def forward(self, nodes: List[torch.Tensor], edges: List[torch.Tensor], axioms: torch.Tensor, conjectures: torch.Tensor) -> torch.Tensor:
        return torch.randn(axioms.shape[0])

def main():
    parser = argparse.ArgumentParser(description = 'export the random-scores TorchScript control')
    parser.add_argument('output', nargs = '?', default = 'random.ts')
    args = parser.parse_args()
    torch.jit.save(torch.jit.script(RandomScores()), args.output)
    print(torch.jit.load(args.output)([torch.zeros(1, 2, dtype = torch.long)], [], torch.arange(5), torch.arange(1)))

main()
