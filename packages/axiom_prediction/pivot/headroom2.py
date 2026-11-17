# Certified headroom against the learned ladder, for stage rows in the library_stages.py format: problems the prover
# fails alone in the stage job but proves on another prover's proof clauses (oracle run), split by where
# Sharbaraz5 e20 ranks those clauses and by whether the learned or the overlap ladder proves them.
# usage: headroom2.py STAGE_DIR TARGET [ORACLE_DIR]
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from fof_table import budget, proves
from library_table import within

buckets = ((1, 16), (17, 64), (65, 256), (257, 10 ** 9))

def main():
    stage_dir, target = Path(sys.argv[1]), sys.argv[2]
    oracle_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else run / 'oracle/p8'
    prepared = {name: p for name, p in json.loads((stage_dir / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (stage_dir / 'results.jsonl').open()):
        if row['prover'] == target:
            results[row['problem']][(row['arm'], row['k'])] = row

    usable = defaultdict(list)
    for row in map(json.loads, (oracle_dir / 'results.jsonl').open()):
        if row['target'] == target and row['source'] != target and row['proved']:
            usable[row['problem']].append(row['source'])

    worst = defaultdict(dict)
    for source in ('Prover9', 'E', 'Vampire', 'iProver'):
        for row in csv.DictReader((run / f'ranking/{source}.csv').open()):
            if 'Sharbaraz5-' in row['checkpoint']:
                worst[row['problem']][source] = int(row['worst_rank'])

    names = [name for name in prepared if ('plain', 0) in results[name]]
    alone = {name for name in names if within(results[name][('plain', 0)], budget)}
    score_s = lambda name: prepared[name]['timings'].get('learned', 0)
    ladder = {arm: {name for name in names if proves(results[name], arm, score_s(name), 'plain')} for arm in ('learned', 'overlap', 'shuffled')}
    certified = [name for name in names if name not in alone and name in usable]
    print(f'{target}, {stage_dir.name}: {len(names)} problems, alone {len(alone)}, learned ladder {len(ladder["learned"])}, overlap ladder {len(ladder["overlap"])}')
    print(f'fails alone but proves on another prover\'s proof clauses: {len(certified)}; of these the learned ladder proves {len(ladder["learned"] & set(certified))}, overlap {len(ladder["overlap"] & set(certified))}, shuffled {len(ladder["shuffled"] & set(certified))}')
    print(f'ladder gains outside the certified set: learned {len(ladder["learned"] - alone - set(certified))}, overlap {len(ladder["overlap"] - alone - set(certified))}')
    table = defaultdict(lambda: [0, 0, 0])
    for name in certified:
        ranks = [worst[name][source] for source in usable[name] if source in worst[name]]
        label = 'no ranking row' if not ranks else next(f'{low}-{high if high < 10 ** 9 else ""}' for low, high in buckets if low <= min(ranks) <= high)
        table[label][0] += 1
        table[label][1] += name in ladder['learned']
        table[label][2] += name in ladder['overlap']

    for label in [f'{low}-{high if high < 10 ** 9 else ""}' for low, high in buckets] + ['no ranking row']:
        total, learned, overlap = table[label]
        print(f'  worst rank of the best usable proof {label:>14s}: {total:3d} problems, learned ladder {learned}, overlap ladder {overlap}')

if __name__ == '__main__':
    main()
