# Writes the problem lists of one part: the CNF manifest of the earlier runs and the first-order problems of the
# Subutai collection, both without problems that read a file on the hanging Lustre target.
# usage: part_lists.py PART CNF_MANIFEST.json
import json
from multiprocessing.pool import ThreadPool
from pathlib import Path
import re
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
tptp = '/rds/user/msf43/hpc-work/TPTP'
hanging = set(Path('/home/msf43/tmp/tptp-ost114.txt').read_text().split('\n'))

def readable(path):
    if path in hanging:
        return False

    includes = re.findall(r"include\('([^']+)'", Path(path).read_text(errors = 'replace'))
    return not any(f'{tptp}/{name}' in hanging for name in includes)

def write(target, paths):
    with ThreadPool(16) as pool:
        kept = [path for path, fine in zip(paths, pool.map(readable, paths)) if fine]

    target.write_text(''.join(path + '\n' for path in kept))
    print(f'{target}: {len(kept)} of {len(paths)} problems kept')

def main():
    part, manifest = sys.argv[1], json.loads(Path(sys.argv[2]).read_text())
    write(run / f'dense/p{part}-cnf.txt', manifest.get('paths') or manifest['problems'])
    first_order = []
    for line in (Path('/rds/user/msf43/hpc-work/subutai/Prover9') / f'p{part}' / 'rows.jsonl').open():
        found = re.match(r'\{"problem": "([^"]+)", "problem_path": "([^"]+)"', line)
        if '+' in found.group(1):
            first_order.append(found.group(2))

    write(run / f'fof/p{part}-fof.txt', sorted(first_order))

if __name__ == '__main__':
    main()
