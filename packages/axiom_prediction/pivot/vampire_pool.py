# Pools one prover's CNF rows (own run on the original file) with its first-order rows (own run on the unfiltered
# Connections clause set, the stronger plain run on part 8), for the schedules of ladder_table.py.
# usage: vampire_pool.py CNF_DIR FOF_DIR [PROVER]
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ladder_table import compare, proves, schedules

def load(directory, prover, fallback):
    prepared = {name: p for name, p in json.loads((directory / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (directory / 'results.jsonl').open()):
        if row['prover'] == prover:
            results[row['problem']][(row['arm'], row['k'])] = row

    names = [name for name in prepared if (fallback, 0) in results[name]]
    return lambda stages, charged: {
        (directory.name, name)
        for name in names
        if proves(results[name], stages, prepared[name]['timings'].get('learned', 0) if charged else 0, fallback)
    }, len(names)

def main():
    prover = sys.argv[3] if len(sys.argv) > 3 else 'Vampire'
    parts = [load(Path(sys.argv[1]), prover, 'plain'), load(Path(sys.argv[2]), prover, 'clausified')]
    solved = lambda stages, charged: set().union(*[solve(stages, charged) for solve, _ in parts])
    alone, sine = solved([], False), solved([(('sine', 0), 8)], False)
    print(f'{prover}, CNF and first-order pooled: {sum(n for _, n in parts)} problems, alone {len(alone)}; own SInE stage {len(sine)} ({compare("against alone", sine, alone)})')
    table = {(label, arm): solved(stages, True) for arm in ('learned', 'shuffled', 'overlap') for label, stages in schedules(arm).items()}
    for label in schedules('learned'):
        reference, tag = (sine, 'SInE only') if label.startswith('SInE') else (alone, 'alone')
        learned = table[(label, 'learned')]
        print(f'  {label:24s} learned {len(learned)} ({compare("against " + tag, learned, reference)}); shuffled {len(table[(label, "shuffled")])}; overlap {len(table[(label, "overlap")])}')
        print(f'  {"":24s} learned: {compare("against shuffled", learned, table[(label, "shuffled")])}, {compare("against overlap", learned, table[(label, "overlap")])}, {compare("against alone", learned, alone)}')

if __name__ == '__main__':
    main()
