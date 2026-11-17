# The oracle premise-transfer table over several parts together and per part (for the replication on other parts).
# usage: oracle_all.py ORACLE_ROOT PART...
from collections import defaultdict
import json
from pathlib import Path
import sys

def load(directory):
    state = json.loads((directory / 'cores.json').read_text())
    gained = defaultdict(set)
    for row in map(json.loads, (directory / 'results.jsonl').open()):
        if row['source'] != row['target'] and row['proved']:
            gained[row['target']].add(row['problem'])

    return state, gained

def lines(state, gained, wanted):
    provers = sorted({prover for p in state.values() for prover in p['provers']})
    names = [name for name in state if wanted(name)]
    union = {name for name in names if any(p['proved'] for p in state[name]['provers'].values())}
    out = [f'{len(names)} problems, proved by at least one prover alone: {len(union)}']
    for target in provers:
        alone = {name for name in names if state[name]['provers'].get(target, {}).get('proved')}
        oracle = {name for name in gained[target] if wanted(name)}
        gap = len(union - alone)
        out.append(f'  {target:8s} alone {len(alone):5d}, with another prover\'s proof clauses +{len(oracle):4d} ({len(oracle) / max(len(alone), 1):+.1%}); gap to the union {gap}, premises explain {len(oracle) / max(gap, 1):.0%}')

    return out

def main():
    root, parts = Path(sys.argv[1]), sys.argv[2:]
    forms = (('CNF', lambda name: '-' in name), ('FOF', lambda name: '+' in name), ('all', lambda name: True))
    merged_state, merged_gained = {}, defaultdict(set)
    for part in parts:
        state, gained = load(root / f'p{part}')
        merged_state.update(state)
        for target, names in gained.items():
            merged_gained[target] |= names

        print(f'== part {part}, all problems: ' + '\n'.join(lines(state, gained, forms[2][1])))

    for label, wanted in forms:
        print(f'\n== parts {", ".join(parts)} together, {label}: ' + '\n'.join(lines(merged_state, merged_gained, wanted)))

if __name__ == '__main__':
    main()
