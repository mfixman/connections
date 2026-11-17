# Rebuilds the Prover9, Vampire and iProver schedule rows from per-problem stage runs (epoch_stages.py output).
# Each row follows its runner: Prover9 premise_run.py / half_support.py / p9_sine_gnn.py, Vampire vampire_compare.py,
# iProver gap_compare2.py. A paused plain prover gets back whatever the GNN stages did not use.
import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import re

csv.field_size_limit(10 ** 9)
budget = 60
probe = 1

@dataclass
class Stage:
    proved: bool
    seconds: float
    limit: float | None = None

    def used(self, window):
        # Time the stage took from the schedule: all of its window, unless it stopped early. A stage that was still
        # running at its own limit is taken to keep running without a proof until the window ends.
        if not self.proved and self.limit is not None and self.seconds >= self.limit - 0.5:
            return window

        return min(window, self.seconds)

    def proves(self, window):
        return self.proved and self.seconds <= window

@dataclass
class Problem:
    name: str
    plain: float | None = None
    axioms: int | None = None
    score_s: float | None = None
    score_error: bool = False
    sine_s: float = 0
    sine_runs: bool = True
    stages: dict = field(default_factory = dict)

    def plain_proves(self, seconds):
        return self.plain is not None and self.plain <= seconds

# Resume rows: probe, pause, scoring, stages, resume; steps are (stage, window) pairs, support is the 20 s SoS slice.
def resume_row(p, steps, support = False, score_cap = budget / 4, skip_small = True, sine = False):
    if p.plain_proves(probe):
        return True

    if p.score_s is not None:
        spent = p.score_s + (p.sine_s if sine else 0)
        if p.score_error or spent > score_cap:
            return p.plain_proves(budget - min(spent, score_cap))
    else:
        spent = 0

    for stage, window in steps:
        size = int(stage[3:]) if stage.startswith('top') else 0
        if skip_small and p.axioms is not None and size and p.axioms <= size:
            break

        if stage == 'sine' and not p.sine_runs:
            continue

        remaining = budget - probe - spent
        if remaining <= 0:
            break

        record = p.stages.get(stage)
        if record is None:
            continue

        window = min(window, remaining)
        if record.proves(window):
            return True

        spent += record.used(window)

    if support and 'support' in p.stages:
        window = min(budget / 3, budget - probe - spent)
        if window > 0:
            if p.stages['support'].proves(window):
                return True

            spent += p.stages['support'].used(window)

    return p.plain_proves(budget - spent)

def prover9_support1(p):
    # premise_run.py Support1: small problems run plain; 1 s plain probe, 5 s scoring, SoS for the rest, then a fresh plain run.
    gate = 16
    score_cap = 5
    if p.axioms is not None and p.axioms <= gate:
        return p.plain_proves(budget)

    if p.plain_proves(probe):
        return True

    if p.score_error or p.score_s is None or p.score_s > score_cap:
        return p.plain_proves(budget - probe - min(p.score_s, score_cap))

    window = budget - probe - p.score_s
    record = p.stages.get('support')
    if record is None:
        return p.plain_proves(window)

    if record.proves(window):
        return True

    return p.plain_proves(window - record.used(window))

ladder = [('top16', 8), ('top64', 8), ('top256', 8)]
rows = {
    'prover9': {
        'Plain': lambda p: p.plain_proves(budget),
        'Support1': prover9_support1,
        'ResumeFilter16': lambda p: resume_row(p, [('top16', 2)]) if p.axioms is None or p.axioms > 16 else p.plain_proves(budget),
        'ResumeSupport1': lambda p: resume_row(p, [], support = True),
        'ResumeFilter16Support1': lambda p: resume_row(p, [('top16', 2)], support = True),
        'Ladder': lambda p: resume_row(p, ladder),
        'SineGnn': lambda p: resume_row(p, [('sine', 8), ('top16', 8), ('top64', 8)], sine = True),
        'LadderSupport1': lambda p: resume_row(p, ladder, support = True),
    },
    'iprover': {
        'Plain': lambda p: p.plain_proves(budget),
        'ResumeFilter16': lambda p: resume_row(p, [('top16', 2)]),
        'Ladder': lambda p: resume_row(p, ladder),
        'SineGnn': lambda p: resume_row(p, [('sine', 8), ('top16', 8), ('top64', 8)], sine = True),
    },
    'vampire': {
        'Plain': lambda p: p.plain_proves(budget),
        'Support1': lambda p: 'support' in p.stages and p.stages['support'].proves(budget),
        'ResumeFilter16': lambda p: resume_row(p, [('top16', 20)], skip_small = False),
        'ResumeSupport1': lambda p: resume_row(p, [('support', 20)], skip_small = False),
        'ResumeFilter16Support1': lambda p: resume_row(p, [('top16', 2), ('support', 20)], skip_small = False),
        'Ladder': lambda p: resume_row(p, ladder, skip_small = False),
        'SineGnn': lambda p: resume_row(p, [('sine', 8), ('top16', 8), ('top64', 8)], skip_small = False),
        'LadderSupport1': lambda p: resume_row(p, [*ladder, ('support', 20)], skip_small = False),
    },
}

rows['eprover'] = rows['iprover']

def family(name):
    return re.match(r'[A-Z]{3}\d{3}', name).group(0)

def read(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []

def add_stages(problems, path, rename = None):
    for row in read(path):
        p = problems.setdefault(row['problem'], Problem(row['problem']))
        for stage in json.loads(row['stages'] or '[]'):
            name = rename or stage['stage']
            p.stages[name] = Stage(stage['proved'], stage['seconds'], stage.get('limit'))

        if row.get('axioms') and '/' not in row['axioms']:
            p.axioms = int(row['axioms'])

        # Both jobs of an epoch score each problem; the filter job's time is the one used.
        if row.get('score_s') and rename is None and p.score_s is None:
            p.score_s = float(row['score_s'])
            p.score_error = bool(row.get('score_error'))

        if row.get('error') and rename is None:
            p.score_error = True
            p.score_s = p.score_s or 0

def load(prover, controls, epoch_dir, shuffled = False):
    problems = {}
    for row in read(controls / f'{prover}-plain.csv'):
        stage = json.loads(row['stages'] or '[]')
        p = problems.setdefault(row['problem'], Problem(row['problem']))
        p.plain = stage[0]['seconds'] if stage and stage[0]['proved'] else None

    for row in read(controls / f'{prover}-sine.csv'):
        p = problems.setdefault(row['problem'], Problem(row['problem']))
        stages = json.loads(row['stages'] or '[]')
        p.sine_runs = bool(stages)
        p.sine_s = float(row['score_s'] or 0)
        if stages:
            p.stages['sine'] = Stage(stages[0]['proved'], stages[0]['seconds'], stages[0].get('limit'))

    suffix = '-shuffled' if shuffled else ''
    stage_dir = controls if shuffled else epoch_dir
    add_stages(problems, stage_dir / f'{prover}-filters{suffix}.csv')
    add_stages(problems, stage_dir / f'{prover}-support{suffix}.csv')
    if shuffled:
        # Shuffled stages are model-free; charge them this epoch's scoring time, as the schedules' shuffled controls do.
        for row in read(epoch_dir / f'{prover}-filters.csv'):
            if row['problem'] in problems and row.get('score_s'):
                problems[row['problem']].score_s = float(row['score_s'])

    return problems

def count(problems, row):
    proved = [p.name for p in problems.values() if row(p)]
    return len(proved), len(set(map(family, proved))), proved

def main():
    parser = argparse.ArgumentParser(description = 'rebuild schedule rows from stage runs')
    parser.add_argument('--run', type = Path, default = Path('/rds/user/msf43/hpc-work/epoch-stages-20261111'))
    parser.add_argument('--output', type = Path, required = True)
    parser.add_argument('--models', nargs = '+', default = ['Sharbaraz5', 'Sharbaraz15'])
    parser.add_argument('--exclude', type = Path, help = 'JSON lists of problem paths left out of every row (Lustre target 114)')
    args = parser.parse_args()

    controls = args.run / 'results/controls'
    excluded = {Path(p).name for paths in json.loads(args.exclude.read_text()).values() for p in paths} if args.exclude else set()
    out = []
    for model in args.models:
        for epoch_dir in sorted((args.run / 'results' / model).glob('e*'), key = lambda d: int(d.name[1:])):
            for prover, schedules in rows.items():
                expected = len(read(controls / f'{prover}-plain.csv'))
                stage_files = [epoch_dir / f'{prover}-{kind}.csv' for kind in ('filters', 'support') if kind == 'filters' or prover != 'iprover']
                if any(len(read(path)) != expected for path in stage_files):
                    continue

                for shuffled in (False, True):
                    problems = {k: v for k, v in load(prover, controls, epoch_dir, shuffled).items() if k not in excluded}

                    for schedule, row in schedules.items():
                        if shuffled and schedule == 'Plain':
                            continue

                        files, families, _ = count(problems, row)
                        name = f'Shuffled{schedule}' if shuffled else schedule
                        out.append({'prover': prover, 'model': model, 'epoch': int(epoch_dir.name[1:]), 'schedule': name, 'files': files, 'families': families, 'problems': len(problems)})

    with args.output.open('w', newline = '') as handle:
        writer = csv.DictWriter(handle, fieldnames = list(out[0]))
        writer.writeheader()
        writer.writerows(out)

    print(f'{len(out)} rows to {args.output}')

if __name__ == '__main__':
    main()
