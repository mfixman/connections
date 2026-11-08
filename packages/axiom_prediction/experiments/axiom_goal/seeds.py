# Three-seed SATCoP test on part 9: equal-thirds schedule from real 20 s stage runs, learned vs random, with McNemar.
import argparse
from pathlib import Path

import paths
import results

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'three-seed equal-thirds schedule table against plain 60 s SatCoP')
    parser.add_argument('--stages', type = Path, default = paths.goal / 'satcop-part9-t20', help = 'folder of the 20 s stage CSVs')
    parser.add_argument('--plain', type = Path, default = paths.goal / 'satcop-part9/baseline-SatCoP.csv', help = 'plain 60 s SatCoP CSV')
    parser.add_argument('--learned-name', default = 'sharbaraz5-e20')
    parser.add_argument('--seeds', type = int, nargs = '+', default = [0, 1, 2])
    return parser.parse_args()

def load(path):
    rows = results.rows(path)
    assert not any(map(results.has_error, rows)), path
    return {r['problem'] for r in rows if results.is_proved(r)}

def schedule(folder, name, tag, base):
    return base | load(folder / f'{name}{tag}-SatCoP.csv') | load(folder / f'{name}{tag}-SatResetCoP.csv')

def main():
    args = parse_arguments()
    base60 = results.proved(args.plain)
    base60_families = len(results.families(base60))
    print(f'plain SatCoP 60 s (seed 0): {len(base60)} files, {base60_families} families')
    totals = {'learned': [], 'random': []}
    for seed in args.seeds:
        tag = '' if seed == 0 else f'-seed{seed}'
        base = load(args.stages / f'baseline{tag}-SatCoP.csv')
        for label, name in (('learned', args.learned_name), ('random', 'random')):
            s = schedule(args.stages, name, tag, base)
            g, l = len(s - base60), len(base60 - s)
            totals[label].append(len(s))
            counts = f'files {len(s)} ({results.pct(len(s), len(base60))}) families {len(results.families(s))} ({results.pct(len(results.families(s)), base60_families)})'
            print(f'seed {seed} {label:8s}: {counts} gains/losses vs plain 60 s {g}/{l} p={results.mcnemar(g, l):.2e}')

    for label, values in totals.items():
        mean = sum(values) / len(values)
        print(f'{label}: mean {mean:.1f} files ({results.pct(mean, len(base60))}), range {min(values)}-{max(values)}')

main()
