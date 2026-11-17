# Oracle premise transfer: runs a prover on a problem keeping only the clauses that another prover's proof used
# (cross runs, on problems the target did not prove), and on its own proof's clauses (self runs, a minimisation step).
# Phase 1 writes every needed problem once in both input syntaxes; phase 2 filters those files line by line.
import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

checkout = Path('/home/msf43/connections')
sys.path[:0] = [str(checkout / 'packages'), str(checkout / 'src')]

from axiom_prediction.prover_cores.collect_cores import run_prover
from axiom_prediction.prover_cores.provers import Prover, modules
from axiom_prediction.search_workers import supervised_results

subutai = Path('/rds/user/msf43/hpc-work/subutai')
tptp_root = '/rds/user/msf43/hpc-work/TPTP'
# Files on the Lustre target that hung compute nodes on 2026-11-11; the list is kept on the home filesystem.
hanging_files = Path('/home/msf43/tmp/tptp-ost114.txt')
clause_index = {
    'tptp': re.compile(r'^cnf\(c(\d+),'),
    'prover9': re.compile(r'# label\(weighted_input_(\d+)\)\.\s*$'),
}

def syntax(prover):
    return 'prover9' if prover == Prover.Prover9 else 'tptp'

def read_cores(part, provers):
    state = {}
    for prover in provers:
        for line in (subutai / prover / f'p{part}' / 'rows.jsonl').open():
            row = json.loads(line)
            entry = {'proved': bool(row['proved']), 'used': row.get('used', []), 'seconds': row['seconds'], 'error': row.get('error', '')}
            state.setdefault(row['problem'], {'path': row['problem_path'], 'provers': {}})['provers'][str(prover)] = entry

    return state

def plan(state):
    cross, own = [], []
    for name, problem in sorted(state.items()):
        for target, mine in problem['provers'].items():
            if mine['proved'] and mine['used']:
                own.append((name, target, target))
            elif not mine['proved'] and not mine['error']:
                cross += [
                    (name, target, source)
                    for source, other in problem['provers'].items()
                    if source != target and other['proved'] and other['used']
                ]

    return cross + own

def touches_hanging_file(path, hanging):
    if path in hanging:
        return True

    includes = re.findall(r"include\('([^']+)'", Path(path).read_text(errors = 'replace'))
    return any(f'{tptp_root}/{name}' in hanging for name in includes)

# Runs in a supervised worker: one load per problem, written once per syntax.
def prepare(path, scratch):
    from axiom_prediction.tptp import load_tptp_problem

    hanging = set(hanging_files.read_text().split('\n'))
    if touches_hanging_file(path, hanging):
        return {'problem': path, 'skipped': 'file on the hanging Lustre target'}

    problem = load_tptp_problem(path, tptp_root = tptp_root)
    conjectures = set(problem.conjecture_clause_ids)
    fof = '+' in Path(path).name
    stem = Path(scratch) / Path(path).name
    modules[Prover.E].write_input(problem.matrix, conjectures, fof, f'{stem}.tptp', 60)
    modules[Prover.Prover9].write_input(problem.matrix, conjectures, fof, f'{stem}.prover9', 60)
    return {'problem': path, 'conjectures': sorted(conjectures), 'clauses': len(problem.matrix.clauses)}

def run_task(key, full_input, kept, seconds):
    name, target, source = key.split('|')
    prover = Prover(target)
    pattern = clause_index[syntax(prover)]
    kept = set(kept)
    with tempfile.NamedTemporaryFile('w', suffix = '.in') as filtered:
        for line in open(full_input):
            match = pattern.search(line)
            if match is None or int(match.group(1)) in kept:
                filtered.write(line)

        filtered.flush()
        started = time.monotonic()
        output = run_prover(modules[prover].command(filtered.name, seconds), seconds + 5)
        run_s = time.monotonic() - started

    proved = modules[prover].proved(output)
    used = modules[prover].used_indices(output) if proved else []
    return {'problem': key, 'kept': len(kept), 'proved': proved, 'used': used, 'run_s': round(run_s, 3)}

def output_row(result):
    name, target, source = result.pop('problem').split('|')
    row = {'problem': name, 'target': target, 'source': source, **result}
    row['seconds'] = round(row['seconds'], 2)
    if 'outcome' in row:
        row['error'] = row.pop('outcome')

    return row

def main():
    parser = argparse.ArgumentParser(description = 'oracle premise transfer between provers')
    parser.add_argument('--part', type = int, default = 8)
    parser.add_argument('--output', type = Path, required = True)
    parser.add_argument('--scratch', type = Path, required = True)
    parser.add_argument('--seconds', type = float, default = 60)
    parser.add_argument('--workers', type = int, default = 76)
    parser.add_argument('--count', type = int, default = 0, help = 'only the first COUNT tasks, for a check')
    parser.add_argument('--provers', type = Prover, nargs = '+', default = list(Prover))
    args = parser.parse_args()

    args.output.mkdir(parents = True, exist_ok = True)
    args.scratch.mkdir(parents = True, exist_ok = True)
    tempfile.tempdir = '/dev/shm'
    state = read_cores(args.part, args.provers)
    (args.output / 'cores.json').write_text(json.dumps(state))
    results = args.output / 'results.jsonl'
    done = set()
    if results.exists():
        done = {f"{r['problem']}|{r['target']}|{r['source']}" for r in map(json.loads, results.open())}

    tasks = [task for task in plan(state) if '|'.join(task) not in done]
    tasks = tasks[:args.count] if args.count else tasks
    workers = min(args.workers, len(os.sched_getaffinity(0)))
    print(f'part {args.part}: {len(state)} problems, {len(tasks)} tasks to do, {len(done)} done, {workers} workers', flush = True)

    needed = sorted({state[name]['path'] for name, _, _ in tasks})
    prepared = {}
    for result in supervised_results(prepare, [(path, str(args.scratch)) for path in needed], workers = workers, timeout = 200):
        prepared[Path(result['problem']).name] = result

    ready = {name: p for name, p in prepared.items() if 'conjectures' in p}
    print(f'prepared {len(ready)} of {len(needed)} problems', flush = True)
    (args.output / 'prepared.json').write_text(json.dumps(prepared))

    arguments = []
    for name, target, source in tasks:
        if name in ready:
            kept = set(state[name]['provers'][source]['used']) | set(ready[name]['conjectures'])
            full_input = f'{args.scratch}/{name}.{syntax(Prover(target))}'
            arguments.append((f'{name}|{target}|{source}', full_input, sorted(kept), args.seconds))

    counts = {'finished': 0, 'proved': 0, 'errors': 0}
    rows = supervised_results(run_task, arguments, workers = workers, timeout = args.seconds + 30)
    try:
        with results.open('a') as stream:
            for row in map(output_row, rows):
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                counts['finished'] += 1
                counts['proved'] += row['proved']
                counts['errors'] += 'error' in row
                if counts['finished'] % 200 == 0:
                    print(counts, flush = True)
    finally:
        rows.close()

    print(f'finished: {counts}', flush = True)

if __name__ == '__main__':
    main()
