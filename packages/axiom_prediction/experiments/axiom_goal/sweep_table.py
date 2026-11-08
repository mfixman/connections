# Prints clean part-8 validation results per model and epoch from the checkpoint sweep.
import argparse
import csv
from pathlib import Path
import re

import paths

def main():
    parser = argparse.ArgumentParser(description = 'checkpoint sweep table of the validation-clean-* runs')
    parser.add_argument('--results', type = Path, default = paths.big_train, help = 'folder holding validation-clean-*/')
    args = parser.parse_args()
    base, rows = {}, {}
    for path in args.results.glob('validation-clean-*/results-e*.csv'):
        name = re.search(r'validation-clean-(\w+)/', str(path)).group(1)
        for r in csv.DictReader(open(path)):
            if not r['metric'].endswith('solved_num'):
                continue

            key = r['metric'].replace('test_', '').replace('_solved_num', '')
            if key.startswith('base_'):
                base[key[5:]] = int(float(r['value']))
            else:
                rows.setdefault((name, int(r['epoch'])), {})[key] = int(float(r['value']))

    print('baseline', base)
    for (name, epoch), v in sorted(rows.items()):
        print(name, epoch, ' '.join(f'{k} {n} ({100 * (n - base[k]) / base[k]:+.1f}%)' for k, n in sorted(v.items())))

main()
