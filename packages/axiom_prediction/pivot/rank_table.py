# Summarises rank_transfer.py output: how well each fixed scorer ranks each prover's part-8 proof cores.
# usage: rank_table.py RANKING_DIR [MIN_AXIOMS]
import csv
from pathlib import Path
import sys

def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float('nan')

def short(checkpoint):
    return checkpoint.replace('-model-epoch-00', ' e').replace('.ts', '')

def main():
    directory, least = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(f'problems with more than {least} axiom clauses; recall = share of the core in the top k, full = whole core in the top k')
    print(f"{'cores':8s} {'scorer':22s} {'n':>5s} {'AUC':>6s} {'rec16':>6s} {'full16':>6s} {'rec64':>6s} {'full64':>6s} {'rec256':>6s} {'full256':>7s}")
    for path in sorted(directory.glob('*.csv')):
        rows = [row for row in csv.DictReader(path.open()) if int(row['axioms']) > least]
        common = None
        for checkpoint in sorted({row['checkpoint'] for row in rows}):
            names = {row['problem'] for row in rows if row['checkpoint'] == checkpoint}
            common = names if common is None else common & names

        for checkpoint in sorted({row['checkpoint'] for row in rows}):
            mine = [row for row in rows if row['checkpoint'] == checkpoint and row['problem'] in common]
            cells = [f'{mean(float(row["auc"]) for row in mine):6.3f}']
            for k in (16, 64, 256):
                sized = [row for row in mine if row[f'recall{k}'] != '']
                cells += [f'{mean(float(row[f"recall{k}"]) for row in sized):6.3f}', f'{mean(int(row[f"full{k}"]) for row in sized):6.3f}']

            print(f'{path.stem:8s} {short(checkpoint):22s} {len(mine):5d} ' + ' '.join(cells))

if __name__ == '__main__':
    main()
