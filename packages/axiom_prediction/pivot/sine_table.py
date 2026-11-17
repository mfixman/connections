# External SInE against the learned dense ladder, from one library_stages.py job (design in NOTES.md, 2026-11-15).
# A schedule is a 1 s probe of the plain run, its stages, then the paused plain run; scoring time is charged only to
# schedules that use the network. usage: sine_table.py OUTPUT_DIR
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_table import compare, dense, proves
from library_table import within

settings = ('1-1', '1-2', '1-3', '1-0', '1.5-0', '2-0', '3-0', '5-0')

def main():
    output = Path(sys.argv[1])
    prepared = {name: p for name, p in json.loads((output / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (output / 'results.jsonl').open()):
        results[(row['problem'], row['prover'])][(row['arm'], row['k'])] = row

    errors = sum('error' in row for rows in results.values() for row in rows.values())
    sizes = [len(p['rankings'].get('esine-1-0', ())) for p in prepared.values()]
    empty = sum(size == 0 for size in sizes)
    print(f'{output}: {len(prepared)} prepared problems, {sum(map(len, results.values()))} rows, {errors} error rows; SInE 1:0 selects nothing on {empty} problems')
    ladder = lambda slice_s: [((f'esine-{setting}', 0), slice_s) for setting in settings]
    learned = [(('learned', k), 2) for k in dense]
    schedules = {
        'SInE stage (1:0, 8 s)': ([(('esine-1-0', 0), 8)], False),
        'SInE ladder, 8 settings x 2 s': (ladder(2), False),
        'SInE ladder, 8 settings x 3 s': (ladder(3), False),
        'built-in / package SInE, 8 s': ([(('sine', 0), 8)], False),
        'learned dense ladder, 13 x 2 s': (learned, True),
        'SInE ladder x 2 s, then learned dense': (ladder(2) + learned, True),
    }
    for prover in sorted({prover for _, prover in results}):
        names = [name for name in prepared if ('plain', 0) in results[(name, prover)]]
        score_s = lambda name: prepared[name]['timings'].get('learned', 0)
        solved = {
            label: {name for name in names if proves(results[(name, prover)], stages, score_s(name) if charged else 0, 'plain')}
            for label, (stages, charged) in schedules.items()
        }
        alone = {name for name in names if within(results[(name, prover)][('plain', 0)], 60)}
        print(f'\n{prover}: {len(names)} problems, alone {len(alone)}')
        for label, mine in solved.items():
            print(f'  {label:38s} {len(mine)} ({compare("against alone", mine, alone)})')

        mine = solved['learned dense ladder, 13 x 2 s']
        print(f'  learned dense ladder: {compare("against the SInE ladder x 2 s", mine, solved["SInE ladder, 8 settings x 2 s"])}, {compare("against the SInE ladder x 3 s", mine, solved["SInE ladder, 8 settings x 3 s"])}, {compare("against the SInE stage", mine, solved["SInE stage (1:0, 8 s)"])}')
        both = solved['SInE ladder x 2 s, then learned dense']
        print(f'  SInE ladder then learned dense: {compare("against the SInE ladder x 2 s", both, solved["SInE ladder, 8 settings x 2 s"])}, {compare("against the SInE ladder x 3 s", both, solved["SInE ladder, 8 settings x 3 s"])}, {compare("against the learned dense ladder", both, mine)}')
        stage = lambda arm, k: sum(within(results[(name, prover)].get((arm, k)), 8) for name in names)
        print('  proofs within 8 s per SInE setting: ' + ', '.join(f'{setting} {stage("esine-" + setting, 0)}' for setting in settings) + f'; any setting {sum(any(within(results[(name, prover)].get(("esine-" + s, 0)), 8) for s in settings) for name in names)}; any learned rung {sum(any(within(results[(name, prover)].get(("learned", k)), 8) for k in dense) for name in names)}')

if __name__ == '__main__':
    main()
