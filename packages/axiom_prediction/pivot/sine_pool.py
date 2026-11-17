# External SInE ladder (S), learned dense ladder (L) and S followed by L (SL), pooled over the clause-form and the
# first-order problems of one part, per prover; rows come from the sine/p{part}-{cnf,fof1,fof2} jobs.
# usage: sine_pool.py PART
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from ladder_table import compare, dense, proves

settings = ('1-1', '1-2', '1-3', '1-0', '1.5-0', '2-0', '3-0', '5-0')
groups = {'Prover9': 'fof1', 'SPASS': 'fof1', 'E': 'fof2', 'iProver': 'fof2', 'Vampire': 'fof2'}
sine = [((f'esine-{setting}', 0), 2) for setting in settings]
learned = [(('learned', k), 2) for k in dense]
schedules = {'plain': ([], False), 'S': (sine, False), 'L': (learned, True), 'SL': (sine + learned, True)}

def family(name):
    return re.match(r'[A-Z]{3}\d{3}', name).group(0)

def load(directory, prover):
    prepared = {name: p for name, p in json.loads((directory / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (directory / 'results.jsonl').open()):
        if row['prover'] == prover:
            results[row['problem']][(row['arm'], row['k'])] = row

    return prepared, results

def main():
    part = sys.argv[1]
    print(f'part {part}; S = SInE ladder 8 settings x 2 s, L = learned dense ladder 13 x 2 s, SL = S then L; files / families')
    for prover, group in groups.items():
        solved = defaultdict(set)
        total = 0
        for form, place in (('CNF', f'sine/p{part}-cnf'), ('FOF', f'sine/p{part}-{group}')):
            prepared, results = load(run / place, prover)
            names = [name for name in prepared if ('plain', 0) in results[name]]
            total += len(names)
            for label, (stages, charged) in schedules.items():
                solved[label] |= {(form, name) for name in names if proves(results[name], stages, prepared[name]['timings'].get('learned', 0) if charged else 0, 'plain')}

        cells = []
        for label in schedules:
            names = solved[label]
            by_form = '+'.join(str(sum(form == wanted for form, _ in names)) for wanted in ('CNF', 'FOF'))
            gain = '' if label == 'plain' else f', {(len(names) - len(solved["plain"])) / len(solved["plain"]):+.1%}'
            cells.append(f'{label} {len(names)} / {len({family(name) for _, name in names})} ({by_form}{gain})')

        print(f'\n{prover}, {total} problems: ' + '; '.join(cells))
        print(f'  {compare("S against plain", solved["S"], solved["plain"])}; {compare("L against plain", solved["L"], solved["plain"])}; {compare("SL against plain", solved["SL"], solved["plain"])}')
        print(f'  {compare("L against S", solved["L"], solved["S"])}; {compare("SL against S", solved["SL"], solved["S"])}; {compare("SL against L", solved["SL"], solved["L"])}')

if __name__ == '__main__':
    main()
