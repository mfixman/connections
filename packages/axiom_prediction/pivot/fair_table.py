# Old (9 Nov frontier-gaps run, Sharbaraz5 e20) against Sunday (sine/p8-*) on exactly the problems both ran, part 8.
# Every row is recomputed per problem on that common set, then scaled to TPTP by Z / |common set|, where Z is the
# number of TPTP problems of the kind the common set holds (CNF only for Prover9, SPASS, iProver, E; CNF + FOF for
# SATCoP and Vampire).
import csv
import json
from multiprocessing.pool import ThreadPool
from pathlib import Path
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
old = Path('/rds/user/msf43/hpc-work/frontier-gaps-20261109')
sys.path.insert(0, str(run))
from ladder_table import proves
from sine_pool import groups, load, settings

csv.field_size_limit(10 ** 9)
tptp = {'CNF': 8345, 'all': 8345 + 9091}
sine = [((f'esine-{s}', 0), 2) for s in settings]
three = [(('learned', k), 8) for k in (16, 64, 256)]
new_rows = {'Plain': [], '1-ladder': [(('learned', 16), 8)], 'SInE': sine, '3-ladder': three, '3-ladder + SInE': sine + three}

def proved_csv(path):
    return {r['problem']: r['proved'] in ('True', 'true') for r in csv.DictReader(open(path))}

def vampire_mode(mode):
    root = old / 'vampire/s5'
    def read(name):
        path = root / name / mode / 'result.json'
        return name + '.p', path.exists() and json.loads(path.read_text()).get('solved', False)

    with ThreadPool(16) as pool:
        return dict(pool.map(read, [d.name for d in root.iterdir() if d.is_dir() and d.name[:3].isupper()]))

def satcop_old():
    base = {r['problem']: r for r in csv.DictReader(open('/rds/user/msf43/hpc-work/axiom-goal/satcop-part8-w32/baseline-SatCoP.csv'))}
    tops = {k: {r['problem']: r for r in csv.DictReader(open(old / f'satcop/s5-top{k}-SatCoP.csv'))} for k in (16, 64, 256)}
    win = lambda r, limit: r is not None and r['proved'] == 'true' and float(r['seconds']) <= limit
    def schedule(name, sizes, slice_s):
        b = base[name]
        if win(b, 1):
            return True
        spent = 1 + float(tops[16].get(name, {}).get('prediction_seconds') or 0)
        for k in sizes:
            r = tops[k].get(name)
            if r is None:
                continue
            if win(r, slice_s):
                return True
            spent += min(float(r['seconds']), slice_s)

        return win(b, 60 - spent + 1)

    return {
        'Plain': {n: win(b, 60) for n, b in base.items()},
        '1-ladder': {n: schedule(n, (16,), 2) for n in base},
        '3-ladder': {n: schedule(n, (16, 64, 256), 8) for n in base},
    }

def old_rows(prover):
    if prover == 'Prover9':
        d = old / 'prover9/s5'
        return {'Plain': proved_csv(d / 'results-Baseline.csv'), '1-ladder': proved_csv(d / 'results-ResumeFilter16.csv'), '3-ladder': proved_csv(d / 'results-ResumeLadder.csv')}

    if prover in ('E', 'SPASS', 'iProver'):
        d = old / prover.lower().replace('e', 'e') if prover != 'E' else old / 'e'
        d = old / {'E': 'e', 'SPASS': 'spass', 'iProver': 'iprover'}[prover]
        return {'Plain': proved_csv(d / 's5/results-Baseline.csv'), '1-ladder': proved_csv(old / f'extra/{d.name}/s5/results-ResumeFilter16.csv'), '3-ladder': proved_csv(d / 's5/results-Ladder.csv')}

    if prover == 'Vampire':
        return {'Plain': vampire_mode('baseline'), '1-ladder': vampire_mode('resume-filter16'), '3-ladder': vampire_mode('resume-filter16:8+filter64:8+filter256:8')}

    return satcop_old()

def new_solved(prover):
    out, have = {label: set() for label in new_rows}, set()
    for place in ('sine/p8-cnf' + ('3' if prover == 'SATCoP' else ''), f'sine/p8-{groups.get(prover, "fof3")}'):
        prepared, results = load(run / place, prover)
        for name in prepared:
            if ('plain', 0) in results[name]:
                have.add(name)
                for label, stages in new_rows.items():
                    if proves(results[name], stages, prepared[name]['timings'].get('learned', 0) if 'learned' in str(stages) else 0, 'plain'):
                        out[label].add(name)

    return have, out

def main():
    print('| Prover | common problems | factor | ' + ' | '.join(f'Old {r}' for r in ('Plain', '1-ladder', '3-ladder')) + ' | ' + ' | '.join(new_rows) + ' |')
    print('|---|---|---|' + '---|' * 8)
    for prover in ('Prover9', 'SPASS', 'SATCoP', 'Vampire', 'iProver', 'E'):
        before = old_rows(prover)
        have, after = new_solved(prover)
        common = set(before['Plain']) & set.intersection(*[set(v) for v in before.values()]) & have
        kind = 'CNF' if all('+' not in n for n in common) else 'all'
        factor = tptp[kind] / len(common)
        cells = [sum(before[r].get(n, False) for n in common) for r in ('Plain', '1-ladder', '3-ladder')]
        cells += [len(after[r] & common) for r in new_rows]
        plain_old, plain_new = cells[0], cells[3]
        shown = [f'{c} → {round(c * factor):,}' for c in cells]
        print(f'| {prover} | {len(common)} ({kind}) | {factor:.2f} | ' + ' | '.join(shown) + ' |')

if __name__ == '__main__':
    main()
