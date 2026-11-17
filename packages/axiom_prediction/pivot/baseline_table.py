# Is the Connections clause set a fair baseline for provers that cannot read first-order input? Compares each
# prover's 60 s run on the Connections clauses (plain) with its run on E's clausification of the same file.
# usage: baseline_table.py OUTPUT_DIR
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library_table import mcnemar, within

def main():
    output = Path(sys.argv[1])
    prepared = {name: p for name, p in json.loads((output / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (output / 'results.jsonl').open()):
        results[(row['problem'], row['prover'])][row['arm']] = row

    converted = sum(bool(p.get('e_clauses')) for p in prepared.values())
    print(f'{len(prepared)} problems, {converted} clausified by E within 60 s')
    for prover in sorted({prover for _, prover in results}):
        names = [name for name in prepared if {'plain', 'eclausified'} <= set(results[(name, prover)])]
        ours = {name for name in names if within(results[(name, prover)]['plain'], 60)}
        theirs = {name for name in names if within(results[(name, prover)]['eclausified'], 60)}
        gains, losses = len(theirs - ours), len(ours - theirs)
        print(f'{prover:8s} n {len(names)}: Connections clauses {len(ours)}, E clauses {len(theirs)} (+{gains}/-{losses}, p = {mcnemar(gains, losses):.2g}), either {len(ours | theirs)}')

if __name__ == '__main__':
    main()
