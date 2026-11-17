# Where the certified headroom of a prover sits: CNF part-8 problems it fails alone but proves on another prover's
# proof clauses, split by where Sharbaraz5 e20 ranks those clauses and by whether the learned ladder proved them.
# usage: headroom.py TARGET   (Prover9 or E)
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sweep = Path('/rds/user/msf43/hpc-work/epoch-stages-20261111')
sys.path.insert(0, str(run))
from simulate_stages import load, rows

buckets = ((1, 16), (17, 64), (65, 256), (257, 10 ** 9))

def ladder_rows(target):
    if target == 'Prover9':
        return 'prover9', load('prover9', sweep / 'results/controls', sweep / 'results/Sharbaraz5/e20')

    return 'eprover', load('eprover', run / 'E/controls', run / 'E/Sharbaraz5-e20')

def main():
    target = sys.argv[1]
    state = json.loads((run / 'oracle/p8/cores.json').read_text())
    oracle = [json.loads(line) for line in (run / 'oracle/p8/results.jsonl').open()]
    worst = defaultdict(dict)
    for source in ('Prover9', 'E', 'Vampire', 'iProver'):
        for row in csv.DictReader((run / f'ranking/{source}.csv').open()):
            if 'Sharbaraz5-' in row['checkpoint']:
                worst[row['problem']][source] = int(row['worst_rank'])

    key, staged = ladder_rows(target)
    cnf = [name for name in state if '-' in name and name in staged]
    failed = [name for name in cnf if target in state[name]['provers'] and not state[name]['provers'][target]['proved']]
    plain = sum(rows[key]['Plain'](staged[name]) for name in cnf)
    ladder = sum(rows[key]['Ladder'](staged[name]) for name in cnf)
    print(f'{target}: {len(cnf)} CNF part-8 problems with stage runs; plain {plain}, learned ladder {ladder}; failed in the 60 s collection run: {len(failed)}')

    proved = defaultdict(list)
    for row in oracle:
        if row['target'] == target and row['source'] != target and row['proved'] and row['problem'] in failed:
            proved[row['problem']].append(row['source'])

    print(f'proved on another prover\'s proof clauses: {len(proved)} (ceiling {plain} -> {plain + len(proved)} if every such selection were found)')
    table = defaultdict(lambda: [0, 0])
    for name, sources in proved.items():
        ranks = [worst[name][source] for source in sources if source in worst[name]]
        label = 'no ranking row' if not ranks else next(f'{low}-{high if high < 10 ** 9 else ""}' for low, high in buckets if low <= min(ranks) <= high)
        table[label][0] += 1
        table[label][1] += rows[key]['Ladder'](staged[name])

    for label in [f'{low}-{high if high < 10 ** 9 else ""}' for low, high in buckets] + ['no ranking row']:
        total, got = table[label]
        print(f'  worst rank of the best usable core {label:>14s}: {total:3d} problems, learned ladder proves {got}')

if __name__ == '__main__':
    main()
