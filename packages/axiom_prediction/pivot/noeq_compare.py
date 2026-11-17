# Point 4: the new rows (no explicit equality axioms) against the previous ones (sine/p*-* jobs, with the axioms),
# Sharbaraz5 e20, on the problems both have, per prover and part: plain, ladder 16/64/256, ladder of 13 cut-offs.
from collections import defaultdict
import json
from pathlib import Path
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from noeq_pareto import collect, methods, old_places, read
from ladder_table import proves
from library_table import mcnemar

def old_rows(prover, part, names):
    rows, scoring = defaultdict(dict), {}
    for place in old_places[prover]:
        read(run / f'sine/p{part}-{place}', prover, rows, scoring, names, ('plain', 'learned'))

    return rows, scoring

def solved(rows, scoring, names, stages):
    return {name for name in names if ('plain', 0) in rows[name] and proves(rows[name], stages, scoring.get(name, 0) if stages else 0, 'plain')}

def main():
    for part in (8, 9):
        names = {Path(p).name for p in (run / f'noeq/p{part}-all.txt').read_text().split()}
        redo = {Path(p).name for p in (run / f'noeq/p{part}-redo.txt').read_text().split()}
        changed = {name for name in redo if '+' in name}
        print(f'\n## part {part}: first-order problems with equality that both versions have; old (with axioms) -> new (without)\n')
        print('| Prover | problems | Plain | Ladder 16/64/256 | Ladder, 13 cut-offs |\n|---|---|---|---|---|')
        for prover in old_places:
            before = old_rows(prover, part, changed)
            after = collect(prover, 'Sharbaraz5', part, names, redo)
            common = [name for name in changed if ('plain', 0) in before[0][name] and ('plain', 0) in after[0][name]]
            cells = []
            for label, score, stages in methods:
                a, b = solved(*before, common, stages), solved(*after, common, stages)
                cells.append(f'{len(a)} -> {len(b)} (+{len(b - a)}/-{len(a - b)}, p = {mcnemar(len(b - a), len(a - b)):.2g})')

            print(f'| {prover} | {len(common)} | ' + ' | '.join(cells) + ' |')

if __name__ == '__main__':
    main()
