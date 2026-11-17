# Final Pareto table without explicit equality axioms: seven provers, every solvable CNF and first-order problem of a
# part, rows plain / ladder 16-64-256 x 8 s / ladder of 13 cut-offs x 2 s, per model and part.
# Rows of first-order problems with equality and of problems left out before come from the noeq/ jobs; the other
# problems keep their rows from the sine/p*-* jobs (Sharbaraz5 learned rungs and plain). usage: noeq_pareto.py
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from ladder_table import dense, proves
from library_table import mcnemar
from sine_pool import settings

esine = tuple(f'esine-{setting}' for setting in settings)
sine_ladder = [((arm, 0), 2) for arm in esine]

provers = ('SATCoP', 'Prover9', 'E', 'SPASS', 'iProver', 'Vampire', 'VampireSInE')
old_places = {'Prover9': ('cnf', 'fof1'), 'SPASS': ('cnf', 'fof1'), 'E': ('cnf', 'fof2'), 'iProver': ('cnf', 'fof2'), 'Vampire': ('cnf', 'fof2'), 'SATCoP': ('cnf3', 'fof3')}
methods = [
    ('Plain', 1, []),
    ('Ladder 16/64/256 x 8 s', 11, [(('learned', k), 8) for k in (16, 64, 256)]),
    ('Ladder, 13 cut-offs x 2 s', 31, [(('learned', k), 2) for k in dense]),
    ('SInE ladder, then 16/64/256', 30, sine_ladder + [(('learned', k), 8) for k in (16, 64, 256)]),
    ('SInE ladder, then 13 cut-offs', 50, sine_ladder + [(('learned', k), 2) for k in dense]),
]

def family(name):
    return re.match(r'[A-Z]{3}\d{3}', name).group(0)

def read(directory, prover, rows, scoring, only = None, arms = None):
    if not (directory / 'results.jsonl').exists():
        return

    prepared = json.loads((directory / 'prepared.json').read_text())
    for row in map(json.loads, (directory / 'results.jsonl').open()):
        name = row['problem']
        if row['prover'] == prover and (only is None or name in only) and (arms is None or row['arm'] in arms):
            rows[name][(row['arm'], row['k'])] = row

    # Problems with at most 8 axiom clauses have no top-k stage but were still scored.
    if arms is None or 'learned' in arms:
        for name, entry in prepared.items():
            if 'learned' in entry.get('timings', {}) and (only is None or name in only):
                scoring[name] = entry['timings']['learned']

def collect(prover, model, part, names, redo):
    rows, scoring = defaultdict(dict), {}
    keep = names - redo
    if prover in old_places:
        for place in old_places[prover]:
            arms = ('plain', 'learned', *esine) if model == 'Sharbaraz5' else ('plain', *esine)
            read(run / f'sine/p{part}-{place}', prover, rows, scoring, keep, arms)

    for directory in sorted((run / 'noeq').glob(f'*/p{part}-*')):
        learned = directory.parent.name == model and '-learned-' in directory.name
        if learned or '-plain-' in directory.name:
            read(directory, prover, rows, scoring, names, ('learned',) if learned else ('plain',))
        elif '-esine-' in directory.name:
            read(directory, prover, rows, {}, names, esine)

    fallback = run / f'noeq/fallback/p{part}/results.jsonl'
    if fallback.exists():
        for row in map(json.loads, fallback.open()):
            name = row['problem']
            if row['prover'] == prover and name in names and ('plain', 0) not in rows[name] and 'error' not in row:
                rows[name][('plain', 0)] = row

    return rows, scoring

def main():
    for part in (8, 9):
        names = {Path(p).name for p in (run / f'noeq/p{part}-all.txt').read_text().split()}
        redo = {Path(p).name for p in (run / f'noeq/p{part}-redo.txt').read_text().split()}
        for model in ('Sharbaraz5', 'Sharbaraz15'):
            print(f'\n## {model} e20, part {part} ({"validation" if part == 8 else "test"}), {len(names)} problems; files / families\n')
            print('| Prover | problems with a plain row | Plain (S 1) | Ladder 16/64/256 (S 11) | Ladder, 13 cut-offs (S 31) | SInE ladder + 16/64/256 (S 30) | SInE ladder + 13 cut-offs (S 50) | not scored |')
            print('|---|---|---|---|---|---|---|---|')
            for prover in provers:
                rows, scoring = collect(prover, model, part, names, redo)
                have = [name for name in names if ('plain', 0) in rows[name]]
                unscored = sum(name not in scoring for name in have)
                cells = []
                plain = None
                for label, score, stages in methods:
                    solved = {name for name in have if proves(rows[name], stages, scoring.get(name, 0) if stages else 0, 'plain')}
                    plain = solved if plain is None else plain
                    gain = '' if not stages else f' ({(len(solved) - len(plain)) / max(len(plain), 1):+.1%}, +{len(solved - plain)}/-{len(plain - solved)}, p = {mcnemar(len(solved - plain), len(plain - solved)):.2g})'
                    cells.append(f'{len(solved)} / {len(set(map(family, solved)))}{gain}')

                print(f'| {prover} | {len(have)} of {len(names)} | ' + ' | '.join(cells) + f' | {unscored} |')

if __name__ == '__main__':
    main()
