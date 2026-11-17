# Sunday table for part 8 and part 9 in one (cells "part 8 / part 9"): Sunday's rows rebuilt from the sine/p*-* stage jobs (with Connections'
# equality axioms), next to the old frontier-gaps rows of 9 November. Prints one markdown table.
from pathlib import Path
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from pareto_all import groups, sine, solved

ladder = [(('learned', k), 8) for k in (16, 64, 256)]
rows = [
    ('Plain', [], False),
    ('1-ladder (top-16 x 8 s)', [(('learned', 16), 8)], True),
    ('SInE (8 settings x 2 s)', sine, False),
    ('3-ladder (16/64/256 x 8 s)', ladder, True),
    ('3-ladder + SInE', sine + ladder, True),
]
# Old rows per part: (plain, ResumeFilter16, Ladder); None where that part has no such run.
old = {
    8: {'SATCoP': (692, 699, 710), 'Prover9': (379, 394, 403), 'Vampire': (957, 944, 954),
        'E': (468, 468, 468), 'SPASS': (347, 348, 348), 'iProver': (414, 414, 416)},
    9: {'SATCoP': (597, None, None), 'Prover9': (333, 346, 348), 'Vampire': (911, None, 914),
        'E': (422, None, 422), 'SPASS': (None, None, None), 'iProver': (388, None, 393)},
}
broken = {'E', 'SPASS'}

def cell(count, base):
    if count is None:
        return ''
    return str(count) if base is None else f'{count} ({(count - base) / base:+.1%})'

def pair(values, bases, mark = ''):
    text = ' / '.join(cell(v, b) for v, b in zip(values, bases))
    return text + mark if any(v is not None for v in values) else ''

def main():
    provers = [p for p in groups if p in old[8]]
    counts = {(part, p): [len(solved(part, p, stages, charged)) for _, stages, charged in rows] for part in (8, 9) for p in provers}
    provers.sort(key = lambda p: -(counts[(9, p)][-1] - counts[(9, p)][0]) / counts[(9, p)][0])
    print('| Part 8 / part 9 | ' + ' | '.join(provers) + ' |')
    print('|---' * (len(provers) + 1) + '|')
    for index, label in ((0, 'Old plain'), (1, 'Old 1-ladder (ResumeFilter16)'), (2, 'Old 3-ladder')):
        cells = []
        for p in provers:
            values = [old[part][p][index] for part in (8, 9)]
            bases = [None if index == 0 else old[part][p][0] for part in (8, 9)]
            mark = ' †' if p in broken and index > 0 else ''
            cells.append(pair(values, bases, mark))
        print(f'| {label} | ' + ' | '.join(cells) + ' |')

    for index, (label, _, _) in enumerate(rows):
        cells = [pair([counts[(part, p)][index] for part in (8, 9)], [None if index == 0 else counts[(part, p)][0] for part in (8, 9)]) for p in provers]
        print(f'| {label} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
