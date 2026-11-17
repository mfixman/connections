# Pareto table for all six provers on CNF + first-order problems, every row from the stage jobs that hold plain, the
# learned cut-offs and the SInE settings together (sine/p{8,9}-*). Files / families; * = on the front (test files).
from collections import defaultdict
from pathlib import Path
import re
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from ladder_table import dense, proves
from sine_pool import load, settings

groups = {'Prover9': ('cnf', 'fof1'), 'SPASS': ('cnf', 'fof1'), 'E': ('cnf', 'fof2'), 'iProver': ('cnf', 'fof2'), 'Vampire': ('cnf', 'fof2'), 'SATCoP': ('cnf3', 'fof3')}
sine = [((f'esine-{setting}', 0), 2) for setting in settings]
methods = [
    ('Plain', 1, [], False),
    ('Ladder 16/64/256 x 8 s', 11, [(('learned', k), 8) for k in (16, 64, 256)], True),
    ('Dense, 7 cut-offs x 3 s', 19, [(('learned', k), 3) for k in dense[::2]], True),
    ('SInE ladder, 8 x 2 s', 21, sine, False),
    ('Dense, 13 cut-offs x 2 s', 31, [(('learned', k), 2) for k in dense], True),
    ('SInE ladder, then 16/64/256', 30, sine + [(('learned', k), 8) for k in (16, 64, 256)], True),
    ('SInE ladder, then dense 13', 50, sine + [(('learned', k), 2) for k in dense], True),
]

def family(name):
    return re.match(r'[A-Z]{3}\d{3}', name).group(0)

def solved(part, prover, stages, charged):
    out = set()
    for place in groups[prover]:
        prepared, results = load(run / f'sine/p{part}-{place}', prover)
        out |= {
            (place, name) for name in prepared
            if ('plain', 0) in results[name] and proves(results[name], stages, prepared[name]['timings'].get('learned', 0) if charged else 0, 'plain')
        }

    return out

def cell(names):
    return f'{len(names)} / {len({family(name) for _, name in names})}'

def main():
    for prover in groups:
        rows = [(label, score, solved(8, prover, stages, charged), solved(9, prover, stages, charged)) for label, score, stages, charged in methods]
        plain = len(rows[0][3])
        print(f'\n### {prover}\n\n| S | Method | val files / fam | test files / fam | test gain | CNF + FOF (test) |\n|--:|---|---|---|---|---|')
        for label, score, val, test in rows:
            front = not any(s <= score and len(t) >= len(test) and (s < score or len(t) > len(test)) for _, s, _, t in rows)
            forms = ' + '.join(str(sum(place.startswith(form) for place, _ in test)) for form in ('cnf', 'fof'))
            gain = '' if label == 'Plain' else f'{(len(test) - plain) / plain:+.1%}'
            print(f"| {score} | {label}{' `*`' if front else ''} | {cell(val)} | {cell(test)} | {gain} | {forms} |")

if __name__ == '__main__':
    main()
