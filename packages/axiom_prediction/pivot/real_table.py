# Compares schedules executed end to end (real_schedule.py) with the same schedule rebuilt from separate stage runs.
# usage: real_table.py PART [SCHEDULE]
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from ladder_table import compare, proves
from sine_pool import groups, load, schedules

def main():
    part, schedule = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'SL'
    real = defaultdict(dict)
    for form in ('cnf', 'fof'):
        path = run / f'real/p{part}-{form}/real.jsonl'
        if path.exists():
            for row in map(json.loads, path.open()):
                if row['schedule'] == schedule:
                    real[row['prover']][(form.upper(), row['problem'])] = row

    stages, charged = schedules[schedule]
    print(f'part {part}, schedule {schedule}: executed end to end against rebuilt from stage runs')
    for prover, rows in real.items():
        rebuilt, scoring = set(), {}
        group = groups.get(prover, 'fof3')
        for form, place in (('CNF', f'sine/p{part}-cnf' + ('3' if prover == 'SATCoP' else '')), ('FOF', f'sine/p{part}-{group}')):
            prepared, results = load(run / place, prover)
            for name in prepared:
                scoring[(form, name)] = prepared[name]['timings'].get('learned', 0)
                if ('plain', 0) in results[name] and proves(results[name], stages, scoring[(form, name)] if charged else 0, 'plain'):
                    rebuilt.add((form, name))

        common = set(rows) & set(scoring)
        executed = {key for key in common if rows[key].get('proved')}
        rebuilt &= common
        errors = sum('error' in rows[key] for key in common)
        late = sum(rows[key].get('seconds', 0) > 61 for key in common)
        by = Counter(rows[key]['by'].split('-')[0] or 'none' for key in executed)
        live = sorted(rows[key].get('score_s', 0) for key in common)
        stored = sorted(scoring[key] for key in common)
        print(f'\n{prover}: {len(common)} problems; executed {len(executed)}, rebuilt {len(rebuilt)} ({compare("executed against rebuilt", executed, rebuilt)}); error rows {errors}; runs over 61 s {late}')
        print(f'  proofs by: {dict(by)}; scoring time median / 90th percentile: live {live[len(live) // 2]:.2f} / {live[len(live) * 9 // 10]:.2f} s, in the stage job {stored[len(stored) // 2]:.2f} / {stored[len(stored) * 9 // 10]:.2f} s')
        for form in ('CNF', 'FOF'):
            mine = {key for key in executed if key[0] == form}
            theirs = {key for key in rebuilt if key[0] == form}
            print(f'  {form}: executed {len(mine)}, rebuilt {len(theirs)}, only executed {len(mine - theirs)}, only rebuilt {len(theirs - mine)}')

if __name__ == '__main__':
    main()
