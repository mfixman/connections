# Rebuilds schedules from library_stages.py rows: plain, a top-16/64/256 ladder followed by plain, and SInE then plain,
# at a 60 s and a 10 s total limit. usage: library_table.py OUTPUT_DIR[,LATER_DIR...] [OVERLAP.json]
# Rows of a later directory replace the same (problem, prover, arm, k) of an earlier one; rankings come from the first.
from collections import defaultdict
import json
from math import comb
from pathlib import Path
import sys

sizes = (16, 64, 256)

def mcnemar(gains, losses):
    n, low = gains + losses, min(gains, losses)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(low + 1)) / 2 ** n)

def within(stage, limit):
    return stage is not None and stage['proved'] and stage['run_s'] <= limit

# Time at which a staged schedule proves the problem, or None; a failed stage costs its run time, at most its slice.
def staged(stages, slice_s, plain, total_s, elapsed = 0.0):
    for stage in stages:
        if stage is None or 'error' in stage:
            continue

        if within(stage, slice_s):
            return elapsed + stage['run_s']

        elapsed += min(stage['run_s'], slice_s)

    return elapsed + plain['run_s'] if within(plain, total_s - elapsed) else None

def method_arm(prepared):
    return 'rerank' if 'rerank' in prepared['rankings'] else 'learned'

def schedules(rows, prepared, total_s, fallback = 'plain'):
    slice_s = 8 if total_s == 60 else 2
    plain = rows.get((fallback, 0))
    out = {'plain': within(plain, total_s), 'bushy': within(rows.get(('bushy', 0)), 10)}
    arms = {'method': method_arm(prepared), 'shuffled': 'shuffled', 'overlap': 'overlap'}
    if 'learned' in prepared['rankings'] and 'rerank' in prepared['rankings']:
        arms['full-graph'] = 'learned'

    for label, arm in arms.items():
        ladder = [rows.get((arm, k)) for k in sizes]
        out[f'ladder-{label}'] = staged(ladder, slice_s, plain, total_s) is not None
        out[f'stages-{label}'] = any(within(stage, slice_s) for stage in ladder)

    # End to end: the scoring time measured in the preparation step is charged to the schedule.
    scored = [rows.get((arms['method'], k)) for k in sizes]
    out['ladder-method-e2e'] = staged(scored, slice_s, plain, total_s, prepared['timings'].get(arms['method'], 0)) is not None
    out['sine+plain'] = staged([rows.get(('sine', 0))], slice_s, plain, total_s) is not None
    return out

def table(results, prepared, names, total_s, fallback = 'plain'):
    lines = []
    for prover in sorted({prover for _, prover in results}):
        solved = defaultdict(set)
        for name in names:
            rows = results.get((name, prover))
            if rows and (fallback, 0) in rows:
                for schedule, proved in schedules(rows, prepared[name], total_s, fallback).items():
                    solved[schedule].add(name) if proved else solved[schedule]
                    solved['n'].add(name)

        if not solved['n']:
            continue

        base = solved['plain']
        cells = [f"{prover:8s} n={len(solved['n']):3d} plain {len(base):3d} bushy10 {len(solved['bushy']):3d}"]
        for schedule in ('ladder-method', 'ladder-method-e2e', 'ladder-shuffled', 'ladder-overlap', 'sine+plain', 'stages-method', 'stages-shuffled', 'stages-overlap'):
            gains, losses = len(solved[schedule] - base), len(base - solved[schedule])
            cells.append(f'{schedule} {len(solved[schedule])} (+{gains}/-{losses}, p={mcnemar(gains, losses):.3g})')

        lines.append(' | '.join(cells))

    return lines

def main():
    outputs = list(map(Path, sys.argv[1].split(',')))
    prepared = {name: p for name, p in json.loads((outputs[0] / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for output in outputs:
        for row in map(json.loads, (output / 'results.jsonl').open()):
            results[(row['problem'], row['prover'])][(row['arm'], row['k'])] = row

    errors = sum('error' in row for rows in results.values() for row in rows.values())
    print(f'{len(prepared)} prepared problems, {sum(map(len, results.values()))} rows, {errors} error rows')
    # A TPTP prover that stops within a second without an SZS status rejected its input.
    rejected = defaultdict(int)
    for (_, prover), rows in results.items():
        for (arm, _), row in rows.items():
            rejected[(prover, arm)] += prover != 'Prover9' and not row.get('proved') and row.get('run_s', 9) < 1 and not row.get('status')

    print('inputs rejected at once: ' + (', '.join(f'{prover} {arm} {count}' for (prover, arm), count in sorted(rejected.items()) if count) or 'none'))
    subsets = {'all': sorted(prepared)}
    if len(sys.argv) > 2:
        known = json.loads(Path(sys.argv[2]).read_text())
        subsets['not in TPTP'] = [name for name in subsets['all'] if name not in known]
        subsets['over 1024 axiom clauses'] = [name for name in subsets['all'] if prepared[name]['axioms'] > 1024]

    for label, names in subsets.items():
        for total_s in (60, 10):
            print(f'\n== {label}: {len(names)} problems, {total_s} s total')
            print('\n'.join(table(results, prepared, names, total_s)))

        # The same schedules when the prover's own run reads the unfiltered Connections clause set (60 s runs only).
        if any(row['run_s'] > 12 for rows in results.values() for key, row in rows.items() if key[0] == 'clausified'):
            print(f'\n== {label}: {len(names)} problems, 60 s total, plain = unfiltered Connections clause set')
            print('\n'.join(table(results, prepared, names, 60, 'clausified')))

if __name__ == '__main__':
    main()
