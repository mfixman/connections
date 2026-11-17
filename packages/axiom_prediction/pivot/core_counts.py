# Counts, per problem size, which problems have a proof core from SATCoP (the Sharbaraz5 dataset) and from each
# external prover (the Subutai rows): the supply of training labels for large problems.
# usage: core_counts.py OUTPUT.json WORKERS
from collections import defaultdict
import json
from multiprocessing import Pool
from pathlib import Path
import re
import sys

sys.path[:0] = ['/home/msf43/connections/packages', '/home/msf43/connections/src']

subutai = Path('/rds/user/msf43/hpc-work/subutai')
sharbaraz = Path('/home/msf43/connections/results/big-train/datasets/Sharbaraz5')
provers = ('Prover9', 'E', 'Vampire', 'iProver')
buckets = ((0, 16), (17, 64), (65, 256), (257, 1024), (1025, 10 ** 9))
large_line = int(10e6)

def read_rows(task):
    prover, part = task
    found = []
    for line in (subutai / prover / f'p{part}' / 'rows.jsonl').open():
        if '"proved": true' not in line[:400]:
            continue

        name = re.match(r'\{"problem": "([^"]+)"', line).group(1)
        # Lines over 10 MB, and proofs whose example was over the size cap, are problems with thousands of clauses;
        # they are counted in the largest bucket.
        example = None if len(line) > large_line else json.loads(line).get('example')
        axioms = len(example['labels']) if example else 10 ** 6
        found.append((name, axioms))

    return prover, part, found

def satcop_rows():
    from axiom_prediction.joined import load_joined
    from axiom_prediction.split import ProblemSplit

    held = {part: ProblemSplit(split = 10, parts = [part], scheme = 5) for part in (8, 9)}
    found = {}
    for example in load_joined(sharbaraz, max_bytes = large_line)[0]:
        part = next((part for part, split in held.items() if split.contains(example.problem_path)), 0)
        found[Path(example.problem_path).name] = (len(example.labels), part)

    return found

def bucket(axioms):
    return next(i for i, (low, high) in enumerate(buckets) if low <= axioms <= high)

def main():
    output, workers = Path(sys.argv[1]), int(sys.argv[2])
    problems = defaultdict(lambda: {'axioms': 0, 'part': 0, 'cores': []})
    with Pool(workers) as pool:
        for prover, part, found in pool.imap_unordered(read_rows, [(prover, part) for prover in provers for part in range(10)]):
            for name, axioms in found:
                entry = problems[name]
                entry['axioms'], entry['part'] = max(entry['axioms'], axioms), part
                entry['cores'].append(prover)

            print(f'{prover} p{part}: {len(found)} cores', flush = True)

    for name, (axioms, part) in satcop_rows().items():
        entry = problems[name]
        entry['axioms'] = entry['axioms'] or axioms
        entry['part'] = entry['part'] or part
        entry['cores'].append('SATCoP')

    output.write_text(json.dumps(problems))
    print('\nproblems with a proof core, parts 0-7 (training) / part 8, by number of axiom clauses')
    print(f"{'axioms':>10s} " + ' '.join(f'{name:>13s}' for name in ('SATCoP', *provers, 'any external', 'any')))
    for index, (low, high) in enumerate(buckets):
        cells = []
        for source in ('SATCoP', *provers, 'external', 'any'):
            counts = [0, 0]
            for entry in problems.values():
                has = source == 'any' or source in entry['cores'] or (source == 'external' and set(entry['cores']) - {'SATCoP'})
                if has and bucket(entry['axioms']) == index and entry['part'] != 9:
                    counts[entry['part'] == 8] += 1

            cells.append(f'{counts[0]:6d} /{counts[1]:5d}')

        print(f"{str(low) + '-' + (str(high) if high < 10 ** 9 else ''):>10s} " + ' '.join(cells))

if __name__ == '__main__':
    main()
