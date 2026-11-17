# Splits the oracle premise-transfer gains by input form and relates them to the union of the provers.
# usage: oracle_split.py ORACLE_DIR
from collections import defaultdict
import json
from pathlib import Path
import sys

def main():
    directory = Path(sys.argv[1])
    state = json.loads((directory / 'cores.json').read_text())
    rows = [json.loads(line) for line in (directory / 'results.jsonl').open()]
    prepared = json.loads((directory / 'prepared.json').read_text())
    skipped = {name for name, p in prepared.items() if 'conjectures' not in p}
    provers = sorted({prover for p in state.values() for prover in p['provers']})
    gained = defaultdict(set)
    for row in rows:
        if row['source'] != row['target'] and row['proved']:
            gained[row['target']].add(row['problem'])

    print(f'{len(state)} problems; {len(skipped)} needed problems not prepared (files on the hanging Lustre target)')
    for label, wanted in (('clause form (CNF)', lambda name: '-' in name), ('first-order form (FOF)', lambda name: '+' in name), ('all', lambda name: True)):
        names = [name for name in state if wanted(name)]
        union = {name for name in names if any(p['proved'] for p in state[name]['provers'].values())}
        print(f'\n== {label}: {len(names)} problems, proved by at least one prover alone: {len(union)}')
        for target in provers:
            alone = {name for name in names if state[name]['provers'].get(target, {}).get('proved')}
            oracle = {name for name in gained[target] if wanted(name)}
            gap = len(union - alone)
            share = len(oracle) / gap if gap else 0
            print(f'  {target:8s} alone {len(alone):4d}, with another prover\'s proof clauses +{len(oracle):3d} ({len(oracle) / max(len(alone), 1):+.1%}); gap to the union {gap}, of which premises explain {share:.0%}')

if __name__ == '__main__':
    main()
