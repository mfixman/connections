# Equal-split SATCoP schedules on part 8 from per-problem proof times (32-worker runs), learned vs random stages.
import argparse
from pathlib import Path

import results

designs = [
    ('gSat', 'gReset'),
    ('gReset', 'top64'),
    ('gSat', 'top64'),
    ('gSat', 'gReset', 'top64'),
    ('gReset', 'top16', 'top64'),
    ('gSat', 'gReset', 'top16', 'top64'),
]

def stage_runs(folder, name):
    return {
        'gSat': results.times(folder / f'{name}-SatCoP.csv'),
        'gReset': results.times(folder / f'{name}-SatResetCoP.csv'),
        'top16': results.times(folder / f'{name}-top16-SatCoP.csv'),
        'top64': results.times(folder / f'{name}-top64-SatCoP.csv'),
    }

def solved(base, runs, stages, t):
    s = {p for p, x in base.items() if x <= t}
    for name in stages:
        s |= {p for p, x in runs[name].items() if x <= t}

    return s

def main():
    parser = argparse.ArgumentParser(description = 'equal-split schedule grid from part-8 stage CSVs')
    parser.add_argument('folder', type = Path)
    parser.add_argument('--learned-name', default = 'sharbaraz5-e20')
    parser.add_argument('--total-seconds', type = int, default = 60)
    args = parser.parse_args()
    base = results.times(args.folder / 'baseline-SatCoP.csv')
    arms = (('learned', stage_runs(args.folder, args.learned_name)), ('random', stage_runs(args.folder, 'random')))
    b, bf = len(base), len(results.families(base))
    for stages in designs:
        t = args.total_seconds // (len(stages) + 1)
        line = f"base {t}s + {' + '.join(stages)} {t}s each:"
        for label, runs in arms:
            s = solved(base, runs, stages, t)
            line += f'  {label} {len(s)} ({results.pct(len(s), b)}) fam {len(results.families(s))} ({results.pct(len(results.families(s)), bf)})'

        print(line)

main()
