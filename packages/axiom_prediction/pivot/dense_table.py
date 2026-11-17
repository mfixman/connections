# Dense ladders from library_stages.py rows on the CNF problems of part 8: more top-k rungs with shorter slices.
# A schedule is a 1 s plain probe, scoring, the rungs in increasing k, then the paused plain run for the time left.
# usage: dense_table.py OUTPUT_DIR
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library_table import mcnemar, within

budget, probe = 60, 1
all_rungs = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
# Fixed before the run (NOTES.md): the standard ladder and five dense ones.
ladders = {
    'standard 16/64/256 x 8 s': ((16, 64, 256), 8),
    '13 rungs x 1 s': (all_rungs, 1),
    '13 rungs x 2 s': (all_rungs, 2),
    '13 rungs x 3 s': (all_rungs, 3),
    '7 rungs x 3 s': (all_rungs[::2], 3),
    '7 rungs x 4 s': (all_rungs[::2], 4),
}

def proves(rows, arm, rungs, slice_s, score_s):
    plain = rows.get(('plain', 0))
    if within(plain, probe):
        return True

    spent = score_s
    for k in rungs:
        stage = rows.get((arm, k))
        window = min(slice_s, budget - probe - spent)
        if stage is None or 'error' in stage or window <= 0:
            continue

        if within(stage, window):
            return True

        spent += min(stage['run_s'], window)

    return within(plain, budget - spent)

def main():
    output = Path(sys.argv[1])
    prepared = {name: p for name, p in json.loads((output / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (output / 'results.jsonl').open()):
        results[(row['problem'], row['prover'])][(row['arm'], row['k'])] = row

    errors = sum('error' in row for rows in results.values() for row in rows.values())
    print(f'{len(prepared)} prepared problems, {sum(map(len, results.values()))} rows, {errors} error rows')
    for prover in sorted({prover for _, prover in results}):
        names = [name for name in prepared if ('plain', 0) in results[(name, prover)]]
        plain = {name for name in names if within(results[(name, prover)][('plain', 0)], budget)}
        print(f'\n{prover}: {len(names)} problems, plain {len(plain)}')
        solved = {}
        for label, (rungs, slice_s) in ladders.items():
            cells = []
            for arm in ('learned', 'shuffled', 'overlap'):
                # Every arm is charged the network's scoring time, so the schedules are equal.
                mine = {name for name in names if proves(results[(name, prover)], arm, rungs, slice_s, prepared[name]['timings'].get('learned', 0))}
                solved[(label, arm)] = mine
                gains, losses = len(mine - plain), len(plain - mine)
                cells.append(f'{arm} {len(mine)} (+{gains}/-{losses}, p = {mcnemar(gains, losses):.2g})')

            print(f'  {label:26s} ' + ' | '.join(cells))

        standard = solved[('standard 16/64/256 x 8 s', 'learned')]
        for label in list(ladders)[1:]:
            mine = solved[(label, 'learned')]
            print(f'  {label} against the standard ladder (learned): +{len(mine - standard)}/-{len(standard - mine)}')

        print('  proofs within 8 s per rung (learned / shuffled / overlap): ' + ', '.join(
            f"{k}: " + '/'.join(str(sum(within(results[(name, prover)].get((arm, k)), 8) for name in names)) for arm in ('learned', 'shuffled', 'overlap'))
            for k in all_rungs
        ))

if __name__ == '__main__':
    main()
