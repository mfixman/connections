# Simulates "first run for t seconds, then second run for the rest of the budget" from per-problem proof times.
import argparse
from pathlib import Path

import results

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'two-stage schedule simulation from per-problem proof times')
    parser.add_argument('first', help = 'CSV of the first stage, also the comparison baseline')
    parser.add_argument('seconds', help = 'comma-separated first-stage budgets, e.g. 20,30,40')
    parser.add_argument('second', nargs = '+', help = 'CSVs of candidate second stages')
    parser.add_argument('--total-seconds', type = int, default = 60)
    return parser.parse_args()

def main():
    args = parse_arguments()
    base = results.times(args.first)
    base_families = len(results.families(base))
    for second_path in args.second:
        second = results.times(second_path)
        for t in map(int, args.seconds.split(',')):
            solved = {p for p, s in base.items() if s <= t} | {p for p, s in second.items() if s <= args.total_seconds - t}
            fams = len(results.families(solved))
            print(f'{Path(second_path).name} t={t:2d}: files {len(solved)} ({results.pct(len(solved), len(base))}), families {fams} ({results.pct(fams, base_families)})')

main()
