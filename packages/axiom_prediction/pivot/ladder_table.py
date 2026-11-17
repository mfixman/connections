# Schedules rebuilt from library_stages.py rows for one prover, with either fallback run (original file or the
# unfiltered Connections clause set): alone, the prover's own SInE as an 8 s stage, the 16/64/256 ladder, the dense
# ladder (13 cut-offs x 2 s), and SInE followed by the dense ladder. A schedule is a 1 s probe of the fallback run,
# scoring (charged to every arm that uses a ranking), its stages, then the paused fallback run.
# usage: ladder_table.py OUTPUT_DIR [PROVER...]
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library_table import mcnemar, within

budget, probe = 60, 1
dense = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)

def proves(rows, stages, score_s, fallback):
    base = rows.get((fallback, 0))
    if within(base, probe):
        return True

    spent = score_s
    for stage, slice_s in stages:
        row = rows.get(stage)
        window = min(slice_s, budget - probe - spent)
        if row is None or 'error' in row or window <= 0:
            continue

        if within(row, window):
            return True

        spent += min(row['run_s'], window)

    return within(base, budget - spent)

def schedules(arm):
    sine = [(('sine', 0), 8)]
    return {
        'ladder 16/64/256 x 8 s': [((arm, k), 8) for k in (16, 64, 256)],
        'dense 13 x 2 s': [((arm, k), 2) for k in dense],
        'SInE + dense 13 x 2 s': sine + [((arm, k), 2) for k in dense],
    }

def compare(label, mine, other):
    gains, losses = len(mine - other), len(other - mine)
    return f'{label} +{gains}/-{losses} (p = {mcnemar(gains, losses):.2g})'

def main():
    output = Path(sys.argv[1])
    prepared = {name: p for name, p in json.loads((output / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (output / 'results.jsonl').open()):
        results[(row['problem'], row['prover'])][(row['arm'], row['k'])] = row

    errors = sum('error' in row for rows in results.values() for row in rows.values())
    print(f'{output}: {len(prepared)} prepared problems, {sum(map(len, results.values()))} rows, {errors} error rows')
    for prover in sys.argv[2:] or sorted({prover for _, prover in results}):
        for fallback in ('plain', 'clausified'):
            names = [name for name in prepared if (fallback, 0) in results[(name, prover)]]
            if not names:
                continue

            score_s = lambda name: prepared[name]['timings'].get('learned', 0)
            solved = lambda stages, charged: {name for name in names if proves(results[(name, prover)], stages, score_s(name) if charged else 0, fallback)}
            alone = solved([], False)
            sine = solved([(('sine', 0), 8)], False)
            source = 'original file' if fallback == 'plain' else 'unfiltered Connections clause set'
            print(f'\n{prover}, own run on the {source}: {len(names)} problems, alone {len(alone)}; own SInE stage {len(sine)} ({compare("against alone", sine, alone)})')
            table = {(label, arm): solved(stages, True) for arm in ('learned', 'shuffled', 'overlap') for label, stages in schedules(arm).items()}
            for label in schedules('learned'):
                reference, tag = (sine, 'SInE only') if label.startswith('SInE') else (alone, 'alone')
                cells = [f'{arm} {len(table[(label, arm)])} ({compare("against " + tag, table[(label, arm)], reference)})' for arm in ('learned', 'shuffled', 'overlap')]
                print(f'  {label:24s} ' + ' | '.join(cells))
                learned = table[(label, 'learned')]
                print(f'  {"":24s} learned: {compare("against shuffled", learned, table[(label, "shuffled")])}, {compare("against overlap", learned, table[(label, "overlap")])}, {compare("against alone", learned, alone)}')

if __name__ == '__main__':
    main()
