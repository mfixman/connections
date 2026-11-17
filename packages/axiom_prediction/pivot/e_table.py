# E on the CNF problems of part 8 with correctly printed filtered inputs: rebuilt schedule rows for the learned ranking
# (Sharbaraz5 e20), the shuffled ranking and symbol overlap, with paired gains over plain E and a split by size.
import json
from math import comb
from pathlib import Path
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from simulate_stages import family, load, rows

buckets = [(0, 64), (65, 256), (257, 1024), (1025, 10 ** 9)]
schedules = ('ResumeFilter16', 'Ladder', 'SineGnn')

def mcnemar(gains, losses):
    n, low = gains + losses, min(gains, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(low + 1)) / 2 ** n)

def proved(problems, schedule, names):
    return {name for name in names if rows['eprover'][schedule](problems[name])}

def main():
    skipped = json.loads(Path('/rds/user/msf43/hpc-work/epoch-stages-20261111/ost114-problems.json').read_text())
    excluded = {Path(p).name for paths in skipped.values() for p in paths}
    arms = {
        'learned': load('eprover', run / 'E/controls', run / 'E/Sharbaraz5-e20'),
        'shuffled': load('eprover', run / 'E/controls', run / 'E/Sharbaraz5-e20', True),
        'overlap': load('eprover', run / 'E/controls', run / 'E/overlap'),
    }
    names = sorted(name for name in arms['learned'] if name not in excluded)
    plain = proved(arms['learned'], 'Plain', names)
    print(f'E --auto, 60 s, {len(names)} problems, {len(set(map(family, names)))} families; plain {len(plain)} ({len(set(map(family, plain)))} families)')
    for schedule in schedules:
        for arm, problems in arms.items():
            mine = proved(problems, schedule, [name for name in names if name in problems])
            gains, losses = len(mine - plain), len(plain - mine)
            print(f'  {schedule:15s} {arm:9s} {len(mine)} ({len(set(map(family, mine)))} families), +{gains}/-{losses}, p = {mcnemar(gains, losses):.3g}')

    print('by number of axiom clauses (Ladder):')
    sized = arms['learned']
    for low, high in buckets:
        inside = [name for name in names if sized[name].axioms is not None and low <= sized[name].axioms <= high]
        cells = [f'plain {len(proved(sized, "Plain", inside))}']
        for arm, problems in arms.items():
            cells.append(f'{arm} {len(proved(problems, "Ladder", [name for name in inside if name in problems]))}')

        print(f'  {low}-{high if high < 10 ** 9 else ""}: n {len(inside)}, ' + ', '.join(cells))

    learned = proved(arms['learned'], 'Ladder', names)
    print('Ladder (learned) gains:', sorted(learned - plain))
    print('Ladder (learned) losses:', sorted(plain - learned))

if __name__ == '__main__':
    main()
