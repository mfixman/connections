# Equal-thirds SATCoP schedule (SatCoP, then SatCoP with weights, then SatResetCoP with weights), simulated from stage runs.
import argparse
from pathlib import Path

import results

def stage_times(path):
    rows = results.rows(path)
    errors = sum(map(results.has_error, rows))
    return {r['problem']: float(r['seconds']) for r in rows if results.is_proved(r)}, errors

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'equal-thirds schedule from stage-run CSVs in one folder')
    parser.add_argument('folder', type = Path)
    parser.add_argument('base_name', help = 'baseline SatCoP CSV inside the folder')
    parser.add_argument('--learned-name', default = 'sharbaraz5-e20', help = 'prefix of the learned-model CSVs')
    parser.add_argument('--stage-seconds', type = float, default = 20)
    return parser.parse_args()

def main():
    args = parse_arguments()
    base, _ = stage_times(args.folder / args.base_name)
    base_families = len(results.families(base))
    for label, name in (('guided', args.learned_name), ('random', 'random')):
        (g, e1), (r, e2) = stage_times(args.folder / f'{name}-SatCoP.csv'), stage_times(args.folder / f'{name}-SatResetCoP.csv')
        s = {p for run in (base, g, r) for p, t in run.items() if t <= args.stage_seconds}
        fams = len(results.families(s))
        print(f'{label} (errors {e1}+{e2}): files {len(s)} vs {len(base)} ({results.pct(len(s), len(base))}), families {fams} vs {base_families} ({results.pct(fams, base_families)})')

main()
