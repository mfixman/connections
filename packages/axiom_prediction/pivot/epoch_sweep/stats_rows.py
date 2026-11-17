# Paired statistics for the rebuilt rows: files and families, exact McNemar against plain and against the shuffled control.
import csv
import json
from math import comb
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_stages import family, load, rows

run = Path('/rds/user/msf43/hpc-work/epoch-stages-20261111')
excluded = {Path(p).name for paths in json.loads((run / 'ost114-problems.json').read_text()).values() for p in paths}

def mcnemar(gains, losses):
    n = gains + losses
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(gains, losses) + 1)) / 2 ** n)

def proved(problems, row):
    return {name for name, p in problems.items() if row(p)}

out = []
for model in ('Sharbaraz5', 'Sharbaraz15'):
    for epoch_dir in sorted((run / 'results' / model).glob('e*'), key = lambda d: int(d.name[1:])):
        for prover, schedules in rows.items():
            learned = {k: v for k, v in load(prover, run / 'results/controls', epoch_dir).items() if k not in excluded}
            shuffled = {k: v for k, v in load(prover, run / 'results/controls', epoch_dir, True).items() if k not in excluded}
            plain = proved(learned, schedules['Plain'])
            for schedule, row in schedules.items():
                if schedule == 'Plain':
                    continue

                mine, control = proved(learned, row), proved(shuffled, row)
                out.append({
                    'prover': prover, 'model': model, 'epoch': int(epoch_dir.name[1:]), 'schedule': schedule,
                    'files': len(mine), 'families': len(set(map(family, mine))), 'plain': len(plain),
                    'vs_plain': f'{len(mine - plain)}/{len(plain - mine)}', 'p_plain': round(mcnemar(len(mine - plain), len(plain - mine)), 5),
                    'shuffled': len(control), 'vs_shuffled': f'{len(mine - control)}/{len(control - mine)}',
                    'p_shuffled': round(mcnemar(len(mine - control), len(control - mine)), 5), 'problems': len(learned),
                })

with open(run / 'rows-stats.csv', 'w', newline = '') as handle:
    writer = csv.DictWriter(handle, fieldnames = list(out[0]))
    writer.writeheader()
    writer.writerows(out)

print(len(out), 'rows')
