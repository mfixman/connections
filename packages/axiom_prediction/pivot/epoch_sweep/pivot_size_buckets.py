# Where do the Prover9 and iProver gains sit? Rebuilt rows of Sharbaraz5 e20 by number of axiom clauses (part 8, CNF).
import json
from pathlib import Path
import sys

sys.path.insert(0, '/rds/user/msf43/hpc-work/epoch-stages-20261111')
from simulate_stages import load, rows

run = Path('/rds/user/msf43/hpc-work/epoch-stages-20261111')
excluded = {Path(p).name for paths in json.loads((run / 'ost114-problems.json').read_text()).values() for p in paths}
buckets = [(0, 16), (17, 64), (65, 256), (257, 1024), (1025, 10 ** 9)]

for prover, schedules in (('prover9', ('Ladder', 'LadderSupport1', 'ResumeFilter16')), ('iprover', ('Ladder',))):
    learned = {k: v for k, v in load(prover, run / 'results/controls', run / 'results/Sharbaraz5/e20').items() if k not in excluded}
    shuffled = {k: v for k, v in load(prover, run / 'results/controls', run / 'results/Sharbaraz5/e20', True).items() if k not in excluded}
    print(prover, 'problems', len(learned), 'without axiom count', sum(p.axioms is None for p in learned.values()))
    for low, high in buckets:
        names = [k for k, p in learned.items() if p.axioms is not None and low <= p.axioms <= high]
        plain = sum(rows[prover]['Plain'](learned[k]) for k in names)
        line = f'  axioms {low}-{high if high < 10 ** 9 else "inf"}: n {len(names)}, plain {plain}'
        for schedule in schedules:
            mine = sum(rows[prover][schedule](learned[k]) for k in names)
            control = sum(rows[prover][schedule](shuffled[k]) for k in names if k in shuffled)
            line += f' | {schedule} {mine} ({mine - plain:+d}), shuffled {control} ({control - plain:+d})'

        print(line)
