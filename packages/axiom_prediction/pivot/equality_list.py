# Lists the CNF part-8 problems with equality that E and SPASS were run on: proved by the plain prover, by the old
# ladder (broken equality printing, e_compare.py / spass_compare.py) and by the fixed ladder (16/64/256 x 8 s rebuilt
# from job 41022519). usage: equality_list.py OUTPUT.csv
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
old = Path('/rds/user/msf43/hpc-work/axiom-goal')
sys.path.insert(0, str(run))
from ladder_table import proves
from library_table import within

tptp = '/rds/user/msf43/hpc-work/TPTP'
ladder = [(('learned', k), 8) for k in (16, 64, 256)]

def has_equality(path):
    text = Path(path).read_text(errors = 'replace')
    for include in re.findall(r"include\('([^']+)'", text):
        text += Path(f'{tptp}/{include}').read_text(errors = 'replace')

    clauses = '\n'.join(line for line in text.splitlines() if not line.startswith('%'))
    return bool(re.search(r'[^=!<>]=[^=>]', clauses))

def old_rows(folder, mode):
    path = old / folder / f'results-{mode}.csv'
    return {row['problem']: row['proved'] == 'True' for row in csv.DictReader(path.open())} if path.exists() else {}

def main():
    paths = (run / 'dense/p8-cnf-prepared.txt').read_text().split()
    prepared = json.loads((run / 'sine/p8-cnf/prepared.json').read_text())
    results = defaultdict(dict)
    for row in map(json.loads, (run / 'sine/p8-cnf/results.jsonl').open()):
        results[(row['problem'], row['prover'])][(row['arm'], row['k'])] = row

    sources = {'E': 'eprover/part8-s5e20', 'SPASS': 'spass/part8-s5e20'}
    fields = ['problem', 'prover', 'axiom_clauses', 'plain_new_job', 'plain_old_job', 'old_ladder', 'fixed_ladder']
    counts = defaultdict(int)
    with open(sys.argv[1], 'w', newline = '') as handle:
        writer = csv.DictWriter(handle, fieldnames = fields)
        writer.writeheader()
        for path in paths:
            name = Path(path).name
            if name not in prepared or 'rankings' not in prepared[name] or not has_equality(path):
                continue

            for prover, folder in sources.items():
                rows = results[(name, prover)]
                score_s = prepared[name]['timings'].get('learned', 0)
                row = {
                    'problem': name, 'prover': prover, 'axiom_clauses': prepared[name]['axioms'],
                    'plain_new_job': int(within(rows.get(('plain', 0)), 60)),
                    'plain_old_job': old_rows(folder, 'Baseline').get(name, ''),
                    'old_ladder': old_rows(folder, 'Ladder').get(name, ''),
                    'fixed_ladder': int(proves(rows, ladder, score_s, 'plain')),
                }
                writer.writerow(row)
                counts[(prover, 'problems')] += 1
                for key in fields[3:]:
                    counts[(prover, key)] += row[key] in (1, True)

                counts[(prover, 'old ladder rows found')] += row['old_ladder'] != ''
                counts[(prover, 'fixed proves, plain does not')] += row['fixed_ladder'] and not row['plain_new_job']
                counts[(prover, 'old proves, old plain does not')] += row['old_ladder'] is True and row['plain_old_job'] is False

    for key in sorted(counts):
        print(key, counts[key])

if __name__ == '__main__':
    main()
