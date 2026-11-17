# Ladder of 13 cut-offs against the ladder 16/64/256 (the ladder used since the start), paired, per model, part and
# prover, on the rows of the final table.
from pathlib import Path
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from noeq_pareto import collect, methods, provers
from ladder_table import proves
from library_table import mcnemar

def main():
    for part in (8, 9):
        names = {Path(p).name for p in (run / f'noeq/p{part}-all.txt').read_text().split()}
        redo = {Path(p).name for p in (run / f'noeq/p{part}-redo.txt').read_text().split()}
        for model in ('Sharbaraz5', 'Sharbaraz15'):
            print(f'\n{model} e20, part {part}: prover, 16/64/256, 13 cut-offs, 13 against 16/64/256')
            for prover in provers:
                rows, scoring = collect(prover, model, part, names, redo)
                have = [name for name in names if ('plain', 0) in rows[name]]
                old, new = (
                    {name for name in have if proves(rows[name], stages, scoring.get(name, 0), 'plain')}
                    for _, _, stages in methods[1:]
                )
                gains, losses = len(new - old), len(old - new)
                print(f'  {prover:12s} {len(old):5d} {len(new):5d}  +{gains}/-{losses} ({(len(new) - len(old)) / len(old):+.1%}, p = {mcnemar(gains, losses):.2g})')

if __name__ == '__main__':
    main()
