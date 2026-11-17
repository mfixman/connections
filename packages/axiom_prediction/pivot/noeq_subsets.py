# Current rows (Sharbaraz5 e20, no equality axioms) on the problem sets of the 9 November Pareto table:
# all problems and CNF problems only, per part: plain, ladder 16/64/256, ladder of 13 cut-offs.
from pathlib import Path
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from noeq_pareto import collect, methods, provers
from ladder_table import proves

def main():
    for part in (8, 9):
        names = {Path(p).name for p in (run / f'noeq/p{part}-all.txt').read_text().split()}
        redo = {Path(p).name for p in (run / f'noeq/p{part}-redo.txt').read_text().split()}
        for prover in provers:
            rows, scoring = collect(prover, 'Sharbaraz5', part, names, redo)
            for label, subset in (('all', names), ('CNF', {n for n in names if '+' not in n})):
                have = [n for n in subset if ('plain', 0) in rows[n]]
                counts = [sum(proves(rows[n], stages, scoring.get(n, 0) if stages else 0, 'plain') for n in have) for _, _, stages in methods]
                print(part, prover, label, len(have), *counts)

if __name__ == '__main__':
    main()
