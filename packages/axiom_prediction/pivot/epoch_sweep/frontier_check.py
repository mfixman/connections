# Checks simulate_stages.py on the 2026-11-09 frontier runs: stage outcomes are read from one schedule's run and replayed
# into another schedule, which is compared with that schedule's own run.
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_stages import Problem, Stage, count, resume_row, rows

frontier = Path('/rds/user/msf43/hpc-work/frontier-gaps-20261109')
ladder_mode = 'resume-filter16:8+filter64:8+filter256:8'

def vampire_result(folder, mode):
    path = folder / mode / 'result.json'
    return json.loads(path.read_text()) if path.exists() else None

# Per stage: time to proof if its log proved, else the time it used (elapsed differences between stage ends).
def vampire_stages(result):
    out = {}
    marks = result['stages']
    for index in range(1, len(marks)):
        # The first slice of a resume mode carries the mode's 'resume-' prefix.
        name = marks[index]['stage'].split(':')[0].removeprefix('resume-')
        used = marks[index]['elapsed'] - marks[index - 1]['elapsed']
        log = 'guided.log' if len(marks) == 2 else f'guided-{index - 1}.log'
        out[name] = Stage(log in result['proof_logs'], used)

    return out

def check_vampire(model):
    root = frontier / 'vampire' / model
    problems = {}
    actual = {'Ladder': set(), 'LadderSupport1': set(), 'ResumeSupport1': set()}
    modes = {'Ladder': ladder_mode, 'LadderSupport1': ladder_mode + '+support1:20', 'ResumeSupport1': 'resume-support1'}
    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        base = vampire_result(folder, 'baseline')
        if base is None:
            continue

        p = problems[folder.name] = Problem(folder.name + '.p')
        p.plain = base['elapsed'] if base['solved'] else None
        for row, mode in modes.items():
            result = vampire_result(folder, mode)
            if result and result['solved']:
                actual[row].add(p.name)

        ladder_run = vampire_result(folder, ladder_mode)
        support_run = vampire_result(folder, 'resume-support1')
        if ladder_run:
            p.stages.update({k: v for k, v in vampire_stages(ladder_run).items() if k.startswith('filter')})
        if support_run:
            p.stages.update(vampire_stages(support_run))

        p.stages = {k.replace('filter', 'top').replace('support1', 'support'): v for k, v in p.stages.items()}

    print(f'Vampire {model}: {len(problems)} problems, plain {sum(p.plain is not None for p in problems.values())}')
    for row in ('Ladder', 'ResumeSupport1', 'LadderSupport1'):
        files, families, proved = count(problems, rows['vampire'][row])
        missed, extra = len(actual[row] - set(proved)), len(set(proved) - actual[row])
        print(f'  {row}: simulated {files} vs run {len(actual[row])} (run only {missed}, simulated only {extra})')

def check_iprover(model):
    def table(folder, mode):
        path = frontier / folder / 'iprover' / model / f'results-{mode}.csv' if folder else frontier / 'iprover' / model / f'results-{mode}.csv'
        return {r['problem']: r for r in csv.DictReader(open(path))}

    base, lad = table('', 'Baseline'), table('', 'Ladder')
    rf16 = table('extra', 'ResumeFilter16')
    problems = {}
    for name, b in base.items():
        p = problems[name] = Problem(name)
        p.plain = float(b['seconds']) if b['proved'] == 'True' and b['seconds'] else None
        r = lad.get(name)
        if r and r['score_s']:
            p.score_s = float(r['score_s'])
            p.score_error = bool(r.get('error'))
            start = probe_s = 1 + p.score_s
            # The ladder's CSV keeps only the proving stage and total time; earlier stages are taken to use their whole 8 s.
            for size in (16, 64, 256):
                stage = f'top{size}'
                if r['stage'] == f'{stage}Learned':
                    p.stages[stage] = Stage(True, float(r['seconds']) - start)
                    break

                p.stages[stage] = Stage(False, 8)
                start += 8

    print(f'iProver {model}: plain {sum(p.plain is not None for p in problems.values())}')
    for row, actual in (('Ladder', lad), ('ResumeFilter16', rf16)):
        files, families, proved = count(problems, rows['iprover'][row])
        run = {k for k, r in actual.items() if r['proved'] == 'True'}
        print(f'  {row}: simulated {files} vs run {len(run)} (run only {len(run - set(proved))}, simulated only {len(set(proved) - run)})')

if __name__ == '__main__':
    for model in ('s5', 's15'):
        check_iprover(model)
        check_vampire(model)
