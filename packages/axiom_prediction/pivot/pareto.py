# Simplicity / performance table of the schedules measured with library_stages.py, per prover, on part 8 (val) and
# part 9 (test), pooled over the problem forms the prover was run on. S follows handoff analysis/PARETO.md:
# stages + rankings + hand-set numbers + prover changes. A method is on the front (*) if no other method of the prover
# has S no larger and test files no fewer, with one strict.
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from ladder_table import dense, proves
from library_table import mcnemar

sets = {
    'Prover9': {'CNF': ('dense/p{}', 'plain'), 'FOF': ('dense/p{}-fof', 'plain')},
    'SPASS': {'CNF': ('spass/p{}', 'plain')},
    'E': {'CNF': ('dense/p{}', 'plain'), 'FOF': ('dense/p{}-fof-e-iprover', 'plain')},
    'iProver': {'FOF': ('dense/p{}-fof-e-iprover', 'plain')},
    'Vampire': {'CNF': ('vampire/p{}-cnf', 'plain'), 'FOF': ('vampire/p{}-fof', 'clausified')},
}
sine = [(('sine', 0), 8)]
methods = [
    ('Plain prover 60 s', 1, lambda arm: []),
    ('Own SInE 8 s, then plain', 6, lambda arm: sine),
    ('Ladder 16/64/256 x 8 s', 11, lambda arm: [((arm, k), 8) for k in (16, 64, 256)]),
    ('Dense ladder, 7 cut-offs x 3 s', 19, lambda arm: [((arm, k), 3) for k in dense[::2]]),
    ('Dense ladder, 13 cut-offs x 2 s', 31, lambda arm: [((arm, k), 2) for k in dense]),
    ('Own SInE 8 s, then dense 13 x 2 s', 35, lambda arm: sine + [((arm, k), 2) for k in dense]),
]

def family(name):
    return re.match(r'[A-Z]{3}\d{3}', name).group(0)

def load(directory, prover):
    prepared = {name: p for name, p in json.loads((directory / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (directory / 'results.jsonl').open()):
        if row['prover'] == prover:
            results[row['problem']][(row['arm'], row['k'])] = row

    return prepared, results

def solved(data, fallback, stages, charged):
    prepared, results = data
    return {
        name
        for name in prepared
        if (fallback, 0) in results[name] and proves(results[name], stages, prepared[name]['timings'].get('learned', 0) if charged else 0, fallback)
    }

def count(names):
    return f'{len(names)} / {len(set(map(family, names)))}'

def main():
    for prover, forms in sets.items():
        data = {(part, form): (load(run / place.format(part), prover), fallback) for part in (8, 9) for form, (place, fallback) in forms.items()}
        sizes = {part: sum(len(data[(part, form)][0][0]) for form in forms) for part in (8, 9)}
        has_sine = all(any(('sine', 0) in rows for rows in results.values()) for (_, results), _ in data.values())
        rows = []
        for label, score, build in methods:
            uses_sine, uses_arm = 'SInE' in label, 'adder' in label or 'dense' in label
            if uses_sine and not has_sine:
                continue

            def pooled(part, arm):
                return set().union(*[
                    {(form, name) for name in solved(data[(part, form)][0], data[(part, form)][1], build(arm), uses_arm)}
                    for form in forms
                ])

            entry = {'label': label, 'S': score, 'val': pooled(8, 'learned'), 'test': pooled(9, 'learned')}
            entry['forms'] = {form: len({name for f, name in entry['test'] if f == form}) for form in forms}
            if uses_arm:
                entry['shuffled'], entry['overlap'] = pooled(9, 'shuffled'), pooled(9, 'overlap')

            rows.append(entry)

        plain = rows[0]
        print(f"\n## {prover}: {' + '.join(forms)}; val {sizes[8]} problems, test {sizes[9]}; plain val {count({n for _, n in plain['val']})}, test {count({n for _, n in plain['test']})} (files / families)\n")
        print('| S | Method | val files / fam | test files / fam | test gain | test by form | Controls on test (shuffled, overlap) |')
        print('|--:|---|---|---|---|---|---|')
        for entry in rows:
            front = not any(o is not entry and o['S'] <= entry['S'] and len(o['test']) >= len(entry['test']) and (o['S'] < entry['S'] or len(o['test']) > len(entry['test'])) for o in rows)
            gains, losses = len(entry['test'] - plain['test']), len(plain['test'] - entry['test'])
            gain = '' if entry is plain else f"{(len(entry['test']) - len(plain['test'])) / len(plain['test']):+.1%} (+{gains}/-{losses}, p = {mcnemar(gains, losses):.2g})"
            controls = ''
            if 'shuffled' in entry:
                against = lambda other: f"+{len(entry['test'] - other)}/-{len(other - entry['test'])}, p = {mcnemar(len(entry['test'] - other), len(other - entry['test'])):.2g}"
                controls = f"{len(entry['shuffled'])} ({against(entry['shuffled'])}); {len(entry['overlap'])} ({against(entry['overlap'])})"

            by_form = ', '.join(f'{form} {n}' for form, n in entry['forms'].items())
            star = ' `*`' if front else ''
            print(f"| {entry['S']} | {entry['label']}{star} | {count({n for _, n in entry['val']})} | {count({n for _, n in entry['test']})} | {gain} | {by_form} | {controls} |")

if __name__ == '__main__':
    main()
