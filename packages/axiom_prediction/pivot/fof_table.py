# Ladder rows from library_stages.py output on TPTP problems (written for the first-order problems of part 8).
# A schedule is a 1 s probe of the fallback run, scoring (charged to every arm), top-16/64/256 stages of 8 s, then the
# paused fallback run for the time left; the fallback is the prover on the original file or on the Connections clauses.
# usage: fof_table.py OUTPUT_DIR
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library_table import mcnemar, within

budget, probe, rungs, slice_s = 60, 1, (16, 64, 256), 8
buckets = ((0, 64), (65, 256), (257, 1024), (1025, 10 ** 9))

def family(name):
    return re.match(r'[A-Z]{3}\d{3}', name).group(0)

def proves(rows, arm, score_s, fallback):
    base = rows.get((fallback, 0))
    if within(base, probe):
        return True

    spent = score_s
    stages = [rows.get(('sine', 0))] if arm == 'sine' else [rows.get((arm, k)) for k in rungs]
    for stage in stages:
        window = min(slice_s, budget - probe - spent)
        if stage is None or 'error' in stage or window <= 0:
            continue

        if within(stage, window):
            return True

        spent += min(stage['run_s'], window)

    return within(base, budget - spent)

def show(label, mine, base):
    gains, losses = len(mine - base), len(base - mine)
    return f'{label} {len(mine)} / {len(set(map(family, mine)))} (+{gains}/-{losses}, p = {mcnemar(gains, losses):.2g})'

def main():
    output = Path(sys.argv[1])
    prepared = {name: p for name, p in json.loads((output / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (output / 'results.jsonl').open()):
        results[(row['problem'], row['prover'])][(row['arm'], row['k'])] = row

    errors = sum('error' in row for rows in results.values() for row in rows.values())
    unscored = sum('learned' not in p['rankings'] for p in prepared.values())
    print(f'{len(prepared)} prepared problems ({unscored} too large to score), {sum(map(len, results.values()))} rows, {errors} error rows; files / families')
    for prover in sorted({prover for _, prover in results}):
        for fallback in ('plain', 'clausified'):
            names = [name for name in prepared if (fallback, 0) in results[(name, prover)]]
            if not names:
                continue

            base = {name for name in names if within(results[(name, prover)][(fallback, 0)], budget)}
            source = 'original file' if fallback == 'plain' and prover not in ('Prover9', 'iProver') else 'Connections clauses'
            print(f'\n{prover}, fallback on the {source}: {len(names)} problems, alone {len(base)} / {len(set(map(family, base)))}')
            solved = {}
            for arm in ('learned', 'shuffled', 'overlap', 'sine'):
                score_s = lambda name: prepared[name]['timings'].get('learned', 0)
                solved[arm] = {name for name in names if proves(results[(name, prover)], arm, score_s(name), fallback)}
                print('  ' + show(f'ladder {arm:8s}' if arm != 'sine' else 'SInE stage     ', solved[arm], base))

            for other in ('shuffled', 'overlap'):
                gains, losses = len(solved['learned'] - solved[other]), len(solved[other] - solved['learned'])
                print(f'  learned against {other}: +{gains}/-{losses}, p = {mcnemar(gains, losses):.2g}')

            for low, high in buckets:
                inside = [name for name in names if low <= prepared[name]['axioms'] <= high]
                cells = [f'alone {len(base & set(inside))}'] + [f'{arm} {len(solved[arm] & set(inside))}' for arm in ('learned', 'shuffled', 'overlap')]
                print(f'  axioms {low}-{high if high < 10 ** 9 else ""}: n {len(inside)}, ' + ', '.join(cells))

if __name__ == '__main__':
    main()
