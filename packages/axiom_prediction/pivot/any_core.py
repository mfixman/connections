# Whole-core coverage when any of the four external provers' proofs counts (the same Connections matrix and scores for
# all four), against the coverage of each prover's own proof. usage: any_core.py RANKING_DIR CHECKPOINT_SUBSTRING
import csv
from pathlib import Path
import sys

provers = ('Prover9', 'E', 'Vampire', 'iProver')

def main():
    directory, wanted = Path(sys.argv[1]), sys.argv[2]
    worst, axioms = {}, {}
    for prover in provers:
        for row in csv.DictReader((directory / f'{prover}.csv').open()):
            if wanted in row['checkpoint']:
                worst.setdefault(row['problem'], {})[prover] = int(row['worst_rank'])
                axioms[row['problem']] = int(row['axioms'])

    for least in (64, 256):
        names = [name for name in worst if axioms[name] > least]
        print(f'\n{wanted}: {len(names)} part-8 problems with more than {least} axiom clauses and a proof by some prover')
        for k in (16, 64, 256):
            if k < least or k == 64:
                cells = []
                for prover in provers:
                    mine = [name for name in names if prover in worst[name]]
                    cells.append(f'{prover} {sum(worst[name][prover] <= k for name in mine) / len(mine):.3f} (n {len(mine)})')

                several = [name for name in names if len(worst[name]) > 1]
                any_core = sum(min(worst[name].values()) <= k for name in names) / len(names)
                first = sum(next(iter(worst[name].values())) <= k for name in several) / len(several)
                best = sum(min(worst[name].values()) <= k for name in several) / len(several)
                print(f'  whole core in top {k}: ' + ', '.join(cells) + f'; any prover {any_core:.3f}; with 2+ proofs (n {len(several)}): one proof {first:.3f}, best proof {best:.3f}')

if __name__ == '__main__':
    main()
