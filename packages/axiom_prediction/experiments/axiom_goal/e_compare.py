# E prover comparison: Baseline, E's own SInE, and a resume ladder (probe, pause, top-k filtered E runs, resume).
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import multiprocessing
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import tempfile
import time

import paths

ladder = ((16, 8), (64, 8), (256, 8))
fields = ['problem', 'mode', 'proved', 'stage', 'seconds', 'score_s', 'error']
settings = {'eprover': str(paths.eprover), 'prover9': str(paths.prover9_selection), 'scratch': '/dev/shm'}
predictor = None

def configure(values):
    # Pool workers are spawned, so each one receives the locations and the TPTP root again.
    settings.update(values)
    os.environ['TPTP_ROOT'] = settings['tptp']
    paths.add_prover9(settings['prover9'])

def tptp_clause(clause):
    from problem import term_text
    variables = {}
    parts = [('' if literal.polarity else '~ ') + term_text(literal.atom, variables) for literal in clause.literals]
    return (' | '.join(parts) or '$false').replace('$T', '$true').replace('$F', '$false')

def proved(output):
    return bool(re.search(r'SZS status (Unsatisfiable|Theorem|ContradictoryAxioms)', output))

def start(path, seconds, cpu, extra = ()):
    command = ['taskset', '-c', str(cpu), settings['eprover'], '--auto', '--tstp-format', '-s', f'--cpu-limit={max(1, int(seconds))}', *extra, str(path)]
    return subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, env = {**os.environ, 'TPTP': os.environ['TPTP_ROOT']})

def finish(process, timeout):
    try:
        output, _ = process.communicate(timeout = timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        output, _ = process.communicate()

    return proved(output)

def score(path):
    from problem import load_problem
    matrix = load_problem(path, os.environ['TPTP_ROOT'])
    weights, info = predictor.predict(matrix)
    return {'weights': weights, **info}

def ranked_axioms(matrix, weights, mode, path):
    conjectures = set(matrix.conjecture_clauses)
    ranked = sorted((i for i in range(len(matrix.clauses)) if i not in conjectures), key = lambda i: (-weights[i], i))
    if mode.startswith('Shuffled'):
        random.Random(path.stem).shuffle(ranked)

    return conjectures, ranked

def write_filtered(matrix, conjectures, kept):
    with tempfile.NamedTemporaryFile('w', suffix = '.p', dir = settings['scratch'], delete = False) as handle:
        for index in kept:
            role = 'negated_conjecture' if index in conjectures else 'axiom'
            handle.write(f'cnf(c{index}, {role}, ({tptp_clause(matrix.clauses[index])})).\n')

    return Path(handle.name)

def climb(matrix, conjectures, ranked, cpu, remaining):
    # Returns the ladder rung that proved the problem, or None when every rung failed or ran out of time.
    for size, seconds in ladder:
        if len(ranked) <= size or remaining() <= 1:
            return None

        filtered = write_filtered(matrix, conjectures, sorted(conjectures | set(ranked[:size])))
        try:
            if finish(start(filtered, seconds, cpu), min(seconds, remaining())):
                return f'top{size}'
        finally:
            os.unlink(filtered)

    return None

def resume_ladder(path, mode, budget, model, cpu, result, started, remaining):
    from predict import Predictor
    from problem import load_problem
    from bounded import bounded
    global predictor
    baseline = start(path, budget, cpu)
    try:
        output, _ = baseline.communicate(timeout = 1)
        result.update(proved = proved(output), stage = 'probe')
        return result
    except subprocess.TimeoutExpired:
        pass

    os.kill(baseline.pid, signal.SIGSTOP)
    if predictor is None:
        predictor = Predictor(model)

    # Parsing and scoring share one bound, like Prover9's schedule; on failure the baseline just resumes.
    with tempfile.NamedTemporaryFile(suffix = '.json', dir = settings['scratch']) as scratch:
        scored = bounded(lambda: score(path), min(15, remaining()), scratch.name)

    result['score_s'] = round(time.monotonic() - started - 1, 2)
    if 'error' in scored:
        os.kill(baseline.pid, signal.SIGCONT)
        result.update(proved = finish(baseline, remaining()), stage = 'resumed-unscored', error = str(scored['error'])[:120])
        return result

    matrix = load_problem(path, os.environ['TPTP_ROOT'])
    conjectures, ranked = ranked_axioms(matrix, scored['weights'], mode, path)
    rung = climb(matrix, conjectures, ranked, cpu, remaining)
    if rung:
        result.update(proved = True, stage = rung)
        baseline.kill()
        baseline.wait()
        return result

    os.kill(baseline.pid, signal.SIGCONT)
    result.update(proved = finish(baseline, remaining()), stage = 'resumed')
    return result

def solve(path, mode, budget, model, cpu):
    started = time.monotonic()
    remaining = lambda: max(0.1, budget - (time.monotonic() - started))
    result = {'problem': path.name, 'mode': mode, 'proved': False, 'stage': ''}
    if mode == 'ESInE':
        result.update(proved = finish(start(path, budget, cpu, ('--sine=Auto',)), remaining()), stage = 'esine')
    elif mode == 'Baseline':
        result.update(proved = finish(start(path, budget, cpu), remaining()), stage = 'baseline')
    else:
        resume_ladder(path, mode, budget, model, cpu, result, started, remaining)

    if result['stage'] != 'probe':
        result['seconds'] = round(time.monotonic() - started, 2)

    return result

def run(task):
    path, mode, budget, model, cpu = task
    try:
        return solve(Path(path), mode, budget, model, cpu)
    except Exception as error:
        return {'problem': Path(path).name, 'mode': mode, 'proved': False, 'stage': 'error', 'error': repr(error)[:200]}

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'compare E prover strategies on a CNF manifest')
    parser.add_argument('--manifest', required = True)
    parser.add_argument('--tptp', required = True)
    parser.add_argument('--output', type = Path, required = True)
    parser.add_argument('--model', default = '')
    parser.add_argument('--modes', nargs = '+', default = ['Baseline'])
    parser.add_argument('--seconds', type = float, default = 60)
    parser.add_argument('--workers', type = int, default = 76)
    parser.add_argument('--count', type = int, default = 0)
    parser.add_argument('--eprover', default = settings['eprover'])
    parser.add_argument('--prover9', default = settings['prover9'], help = "Prover9's axiom_selection directory")
    parser.add_argument('--scratch', default = settings['scratch'], help = 'directory for the filtered problem files')
    return parser.parse_args()

def run_mode(args, mode, problem_paths, cpus):
    # Rows are appended as they finish, so a job that hits its time limit resumes where it stopped.
    target = args.output / f'results-{mode}.csv'
    done = {r['problem'] for r in csv.DictReader(open(target))} if target.exists() else set()
    todo = [p for p in problem_paths if Path(p).name not in done]
    tasks = [(p, mode, args.seconds, args.model, cpus[i % len(cpus)]) for i, p in enumerate(todo)]
    begun = time.monotonic()
    values = {'eprover': args.eprover, 'prover9': args.prover9, 'scratch': args.scratch, 'tptp': args.tptp}
    pool = ProcessPoolExecutor(min(args.workers, len(cpus)), mp_context = multiprocessing.get_context('spawn'), initializer = configure, initargs = (values,))
    with open(target, 'a', newline = '') as handle, pool:
        writer = csv.DictWriter(handle, fieldnames = fields)
        if not done:
            writer.writeheader()

        for finished, future in enumerate(as_completed([pool.submit(run, task) for task in tasks]), 1):
            writer.writerow(future.result())
            handle.flush()
            if finished % 50 == 0:
                print(f'{mode}: {finished}/{len(todo)} in {time.monotonic() - begun:.0f} s', flush = True)

    rows = list(csv.DictReader(open(target)))
    print(mode, sum(r['proved'] == 'True' for r in rows), 'proved of', len(rows), 'errors', sum(r['stage'] == 'error' for r in rows), flush = True)

def main():
    args = parse_arguments()
    configure({'eprover': args.eprover, 'prover9': args.prover9, 'scratch': args.scratch, 'tptp': args.tptp})
    problem_paths = json.load(open(args.manifest))['paths']
    problem_paths = problem_paths[:args.count] if args.count else problem_paths
    args.output.mkdir(parents = True, exist_ok = True)
    cpus = sorted(os.sched_getaffinity(0))
    print(f'{len(cpus)} cpus, {len(problem_paths)} problems', flush = True)
    for mode in args.modes:
        run_mode(args, mode, problem_paths, cpus)

if __name__ == '__main__':
    main()
