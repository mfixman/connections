# Paired comparison of strategy CSVs against a baseline CSV: proofs, gains, losses, errors.
import argparse
from pathlib import Path

import results

def load(path):
    return {row['problem']: row for row in results.rows(path)}

def proved(rows, problems):
    return {p for p in problems if results.is_proved(rows[p])}

def errors(rows, problems):
    return sum(results.has_error(rows[p]) for p in problems)

def main():
    parser = argparse.ArgumentParser(description = 'paired gains and losses of strategy CSVs against a baseline CSV')
    parser.add_argument('baseline')
    parser.add_argument('arms', nargs = '+')
    args = parser.parse_args()
    base = load(args.baseline)
    print('baseline', len(base), 'proved', len(proved(base, base)), 'errors', errors(base, base))
    for path in args.arms:
        arm = load(path)
        common = set(arm) & set(base)
        b, g = proved(base, common), proved(arm, common)
        name = '/'.join(Path(path).parts[-2:])
        print(f'{name}: {len(common)} problems, {len(b)} -> {len(g)} ({results.pct(len(g), len(b))}), gains {len(g - b)}, losses {len(b - g)}, errors {errors(arm, common)}')

main()
