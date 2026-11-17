# Tables from oracle_cores.py: what each prover gains when restricted to another prover's proof clauses, how it does on
# its own, and how much the provers' proofs agree. usage: oracle_table.py ORACLE_DIR
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys

def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0

def cross_table(state, rows, provers):
    print('\n== cross runs: target prover restricted to the clauses of a source prover proof (target alone failed in 60 s)')
    print(f"{'target':8s} {'proved':>6s} {'failed':>6s} {'with a source proof':>20s} {'oracle proves':>14s}   per source: proved / tried")
    for target in provers:
        mine = [row for row in rows if row['target'] == target and row['source'] != target]
        failed = [name for name, p in state.items() if target in p['provers'] and not p['provers'][target]['proved'] and not p['provers'][target]['error']]
        proved = sum(p['provers'].get(target, {}).get('proved', False) for p in state.values())
        tried = {row['problem'] for row in mine}
        solved = {row['problem'] for row in mine if row['proved']}
        cells = []
        for source in provers:
            runs = [row for row in mine if row['source'] == source]
            if runs:
                cells.append(f"{source} {sum(row['proved'] for row in runs)}/{len(runs)}")

        errors = sum('error' in row for row in mine)
        print(f"{target:8s} {proved:6d} {len(failed):6d} {len(tried):20d} {len(solved):14d}   {', '.join(cells)}; error rows {errors}")

def kept_table(rows, provers):
    print('\n== cross runs by number of clauses kept (all targets): proved / tried')
    buckets = [(0, 8), (9, 16), (17, 32), (33, 64), (65, 10 ** 9)]
    for target in provers:
        cells = []
        for low, high in buckets:
            runs = [row for row in rows if row['target'] == target and row['source'] != target and low <= row['kept'] <= high]
            cells.append(f"{low}-{high if high < 10 ** 9 else ''}: {sum(row['proved'] for row in runs)}/{len(runs)}")

        print(f"{target:8s} " + '   '.join(cells))

def own_table(state, rows, provers):
    print('\n== self runs: prover on the clauses of its own proof')
    for target in provers:
        runs = [row for row in rows if row['target'] == target and row['source'] == target and 'error' not in row]
        again = [row for row in runs if row['proved']]
        before = [state[row['problem']]['provers'][target] for row in again]
        ratio = median(old['seconds'] / max(row['run_s'], 0.01) for old, row in zip(before, again)) if again else 0
        smaller = sum(len(row['used']) < len(old['used']) for old, row in zip(before, again))
        print(f"{target:8s} tried {len(runs):4d}, proved again {len(again):4d}, median speed-up {ratio:.1f}x, shorter proof premises in {smaller}")

def agreement_table(state, provers):
    print('\n== agreement of proof clauses on problems both provers proved (all clauses a proof cites, conjecture included)')
    print(f"{'pair':18s} {'both':>5s} {'Jaccard':>8s} {'identical':>9s} {'first in second':>16s} {'second in first':>16s} {'median sizes':>13s}")
    for i, first in enumerate(provers):
        for second in provers[i + 1:]:
            pairs = []
            for p in state.values():
                a, b = p['provers'].get(first), p['provers'].get(second)
                if a and b and a['proved'] and b['proved'] and a['used'] and b['used']:
                    pairs.append((set(a['used']), set(b['used'])))

            if pairs:
                same = sum(a == b for a, b in pairs) / len(pairs)
                inside, around = sum(a <= b for a, b in pairs) / len(pairs), sum(b <= a for a, b in pairs) / len(pairs)
                sizes = f"{median(len(a) for a, _ in pairs):.0f}/{median(len(b) for _, b in pairs):.0f}"
                print(f"{first + '-' + second:18s} {len(pairs):5d} {sum(jaccard(a, b) for a, b in pairs) / len(pairs):8.3f} {same:9.3f} {inside:16.3f} {around:16.3f} {sizes:>13s}")

def main():
    directory = Path(sys.argv[1])
    state = json.loads((directory / 'cores.json').read_text())
    rows = [json.loads(line) for line in (directory / 'results.jsonl').open()] if (directory / 'results.jsonl').exists() else []
    provers = sorted({prover for p in state.values() for prover in p['provers']})
    print(f'{len(state)} problems, {len(rows)} oracle rows')
    agreement_table(state, provers)
    if rows:
        cross_table(state, rows, provers)
        kept_table(rows, provers)
        own_table(state, rows, provers)

if __name__ == '__main__':
    main()
