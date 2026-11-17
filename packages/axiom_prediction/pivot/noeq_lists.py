# Problem lists for the reruns without explicit equality axioms, per part: every solvable CNF and first-order problem
# (all), and the subset whose earlier rows are not valid any more (redo): first-order problems with equality, plus
# problems that earlier jobs left out (Lustre files that hung, preparation time-outs).
# usage: noeq_lists.py PART CNF_MANIFEST.json
import json
from multiprocessing.pool import ThreadPool
from pathlib import Path
import re
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
tptp = '/rds/user/msf43/hpc-work/TPTP'

def has_equality(path):
    text = Path(path).read_text(errors = 'replace')
    for include in re.findall(r"include\('([^']+)'", text):
        text += Path(f'{tptp}/{include}').read_text(errors = 'replace')

    body = '\n'.join(line for line in text.splitlines() if not line.startswith('%'))
    return bool(re.search(r'[^=!<>~]=[^=>]', body))

def main():
    part = sys.argv[1]
    manifest = json.loads(Path(sys.argv[2]).read_text())
    cnf = manifest.get('paths') or manifest['problems']
    fof = []
    for line in (Path('/rds/user/msf43/hpc-work/subutai/Prover9') / f'p{part}' / 'rows.jsonl').open():
        found = re.match(r'\{"problem": "([^"]+)", "problem_path": "([^"]+)"', line)
        if '+' in found.group(1):
            fof.append(found.group(2))

    fof = sorted(fof)
    with ThreadPool(16) as pool:
        equality = dict(zip(fof, pool.map(has_equality, fof)))

    known = set()
    for place in ('cnf', 'fof1', 'fof2'):
        prepared = json.loads((run / f'sine/p{part}-{place}/prepared.json').read_text())
        known |= {name for name, p in prepared.items() if 'rankings' in p}

    redo = [p for p in fof if equality[p] or Path(p).name not in known] + [p for p in cnf if Path(p).name not in known]
    out = run / 'noeq'
    out.mkdir(exist_ok = True)
    (out / f'p{part}-all.txt').write_text(''.join(p + '\n' for p in cnf + fof))
    (out / f'p{part}-redo.txt').write_text(''.join(p + '\n' for p in redo))
    print(f'part {part}: CNF {len(cnf)}, first-order {len(fof)} ({sum(equality.values())} with equality); all {len(cnf) + len(fof)}, redo {len(redo)}')

if __name__ == '__main__':
    main()
