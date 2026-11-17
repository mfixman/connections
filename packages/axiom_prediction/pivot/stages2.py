# Runs each schedule stage once per problem (filter sequence, set of support, plain, SInE) and records its time to proof,
# so the Prover9, Vampire and iProver schedules can be rebuilt per epoch by simulation.
import argparse
import csv
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import tempfile
import time

connections = Path('/home/msf43/connections')
prover9_root = connections / 'external/Prover9/axiom_selection'
sys.path[:0] = [str(prover9_root), str(connections / 'packages'), str(connections / 'src')]
csv.field_size_limit(10 ** 9)

kinds = ('filters', 'support', 'plain', 'sine')
fields = ['problem', 'axioms', 'score_s', 'score_error', 'stages', 'error', 'seconds']
predictors = {}

def predictor(path):
    from predict import Predictor
    if path not in predictors:
        predictors[path] = Predictor(path)

    return predictors[path]

def stopped(process, limit):
    started = time.monotonic()
    try:
        output, _ = process.communicate(timeout = limit)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = process.communicate()

    return output, time.monotonic() - started

def proof_found(output):
    return bool(re.search(r'SZS status (Unsatisfiable|Theorem|ContradictoryAxioms)', output))

# E and SPASS reject the parenthesised equations of the earlier runners, so equality literals are printed bare.
def tptp_literal(literal, variables):
    from problem import term_text
    atom = literal.atom
    if atom.symbol in ('=', 'equal___'):
        left, right = (term_text(arg, variables) for arg in atom.args)
        return f'{left} = {right}' if literal.polarity else f'{left} != {right}'

    return ('' if literal.polarity else '~ ') + term_text(atom, variables)

def tptp_clause(clause):
    variables = {}
    parts = [tptp_literal(literal, variables) for literal in clause.literals]
    return (' | '.join(parts) or '$false').replace('$T', '$true').replace('$F', '$false')

class Prover9:
    supports = True

    def __init__(self, path, args):
        from problem import clause_text, load_problem
        self.matrix = load_problem(path, args.tptp)
        self.clauses = [clause_text(c) for c in self.matrix.clauses]
        self.scratch = Path(tempfile.mkdtemp(prefix = 'p9-'))

    def run(self, name, kept, seconds, usable = ()):
        from portfolio import search
        result = search(self.clauses, sorted(kept), seconds, self.scratch / name, usable = sorted(usable))
        return {'proved': result['proved'], 'seconds': round(result['search_s'], 3), 'exit': result['exit']}

    def plain(self, seconds):
        return self.run('plain', range(len(self.clauses)), seconds)

    def support(self, conjectures, top, rest, seconds):
        return self.run('support', conjectures | top, seconds, usable = rest)

class IProver:
    supports = False

    def __init__(self, path, args):
        from problem import load_problem
        self.path, self.tptp = path, args.tptp
        self.binary = str(connections / 'external/iprover/iproveropt')
        self.matrix = load_problem(path, args.tptp)

    def start(self, path, seconds):
        command = [self.binary, '--include_path', self.tptp, '--time_out_real', str(max(1, int(seconds))), str(path)]
        return subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, start_new_session = True)

    def finished(self, process, seconds):
        output, elapsed = stopped(process, seconds + 2)
        return {'proved': proof_found(output), 'seconds': round(elapsed, 3), 'exit': process.returncode}

    def run(self, name, kept, seconds):
        conjectures = set(self.matrix.conjecture_clauses)
        with tempfile.NamedTemporaryFile('w', suffix = '.p', delete = False) as handle:
            for index in sorted(kept):
                role = 'negated_conjecture' if index in conjectures else 'axiom'
                handle.write(f'cnf(c{index}, {role}, ({tptp_clause(self.matrix.clauses[index])})).\n')

        try:
            return self.finished(self.start(handle.name, seconds), seconds)
        finally:
            os.unlink(handle.name)

    def plain(self, seconds):
        return self.finished(self.start(self.path, seconds), seconds)

class EProver(IProver):
    def __init__(self, path, args):
        IProver.__init__(self, path, args)
        self.binary = str(connections / 'external/eprover/bin/eprover')

    def start(self, path, seconds, sine = False):
        options = ['--auto', '--sine=Auto'] if sine else ['--auto']
        command = [self.binary, *options, '--tstp-format', '-s', f'--cpu-limit={max(1, int(seconds))}', str(path)]
        env = {**os.environ, 'TPTP': self.tptp}
        return subprocess.Popen(command, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, start_new_session = True, env = env)

    # E's own SInE on the whole problem, as in the earlier E runs.
    def sine(self, seconds):
        return self.finished(self.start(self.path, seconds, sine = True), seconds)

class Vampire:
    supports = True

    def __init__(self, path, args):
        self.path, self.args = path, args

    def launch(self, seconds, model = None, filter_size = None, support = False, sine = False):
        env = {k: v for k, v in os.environ.items() if not k.startswith('TOROK_')}
        env.update(TPTP = self.args.tptp, OMP_NUM_THREADS = '1', OPENBLAS_NUM_THREADS = '1', MKL_NUM_THREADS = '1')
        command = [
            self.args.vampire, '--mode', 'vampire', '--time_limit', f'{seconds:.1f}s', '--memory_limit', '3072',
            '--random_seed', '1', '--proof', 'tptp', '--saturation_algorithm', 'lrs',
            '--sine_selection', 'axioms' if sine else 'off',
        ]

        if model:
            command += ['--torok_file', model]

        if filter_size:
            env['TOROK_FILTER'] = str(filter_size)

        if support:
            env['TOROK_SUPPORT'] = '1'
            command += ['--sos', 'all']

        process = subprocess.Popen([*command, str(self.path)], stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True, env = env, start_new_session = True)
        output, elapsed = stopped(process, seconds + 2)
        return {
            'proved': proof_found(output),
            'seconds': round(elapsed, 3),
            'exit': process.returncode,
            'status': ','.join(re.findall(r'SZS status (\w+)', output)),
            'reason': ','.join(re.findall(r'Termination reason: ([^\n]+)', output))[:80],
            'score_errors': re.findall(r'Torok inference failed: ([^\n]+)', output)[:1],
        }

def score(prover, model, args):
    from bounded import bounded
    predictor(model)
    with tempfile.NamedTemporaryFile(suffix = '.json') as scratch:
        started = time.monotonic()
        scored = bounded(lambda: {'weights': predictor(model).predict(prover.matrix)[0]}, args.score_cap, scratch.name)
        return scored, round(time.monotonic() - started, 3)

def ranking(prover, weights, shuffled, path):
    conjectures = set(prover.matrix.conjecture_clauses)
    axioms = [i for i in range(len(prover.matrix.clauses)) if i not in conjectures]
    if shuffled:
        random.Random(Path(path).stem).shuffle(axioms)
        return conjectures, axioms

    return conjectures, sorted(axioms, key = lambda i: (-weights[i], i))

# Top-16, then top-64 only if top-16 failed within 8 s, then top-256 only if both failed (the ladder's order).
def filter_sequence(prover, conjectures, ranked, args, row):
    window = 8
    for size in (16, 64, 256):
        if len(ranked) <= size:
            break

        result = prover.run(f'top{size}', conjectures | set(ranked[:size]), window)
        row['stages'].append({'stage': f'top{size}', 'limit': window, **result})
        if result['proved']:
            break

def vampire_filter_sequence(prover, model, row):
    # ResumeFilter16 gives top-16 a 20 s slice, so top-16 runs 20 s; top-64 runs whenever it had not proved by 8 s.
    top16 = prover.launch(20, model, filter_size = 16)
    row['stages'].append({'stage': 'top16', 'limit': 20, **top16})
    if top16['proved'] and top16['seconds'] <= 8:
        return

    for size in (64, 256):
        result = prover.launch(8, model, filter_size = size)
        row['stages'].append({'stage': f'top{size}', 'limit': 8, **result})
        if result['proved']:
            return

def guided(prover, path, args, row):
    if args.prover == 'vampire':
        model = args.random if args.shuffled else args.model
        if args.kind == 'filters':
            vampire_filter_sequence(prover, model, row)
        else:
            row['stages'].append({'stage': 'support', 'limit': args.support_s, **prover.launch(args.support_s, model, support = True)})

        return

    weights = None
    if not args.shuffled:
        scored, row['score_s'] = score(prover, args.model, args)
        if 'error' in scored:
            row['score_error'] = str(scored['error'])[:120]
            return

        weights = scored['weights']

    conjectures, ranked = ranking(prover, weights, args.shuffled, path)
    row['axioms'] = len(ranked)
    if args.kind == 'filters':
        filter_sequence(prover, conjectures, ranked, args, row)
    elif ranked:
        rest = set(ranked[1:])
        row['stages'].append({'stage': 'support', 'limit': args.support_s, **prover.support(conjectures, set(ranked[:1]), rest, args.support_s)})

def sine_stage(prover, path, args, row):
    window = 8
    if args.prover == 'vampire':
        row['stages'].append({'stage': 'sine', 'limit': window, **prover.launch(window, sine = True)})
        return

    if args.prover == 'eprover':
        row['stages'].append({'stage': 'sine', 'limit': window, **prover.sine(window)})
        return

    scored, row['score_s'] = score(prover, args.sine, args)
    if 'error' in scored:
        row['score_error'] = str(scored['error'])[:120]
        return

    conjectures = set(prover.matrix.conjecture_clauses)
    axioms = [i for i in range(len(prover.matrix.clauses)) if i not in conjectures]
    selected = [i for i in axioms if scored['weights'][i] > 0.5]
    row['axioms'] = len(axioms)
    row['selected'] = len(selected)
    if selected and len(selected) < len(axioms):
        row['stages'].append({'stage': 'sine', 'limit': window, **prover.run('sine', conjectures | set(selected), window)})

def solve(path, args):
    started = time.monotonic()
    prover = {'prover9': Prover9, 'iprover': IProver, 'eprover': EProver, 'vampire': Vampire}[args.prover](path, args)
    row = {'problem': Path(path).name, 'stages': []}
    if args.kind == 'plain':
        row['stages'].append({'stage': 'plain', 'limit': 60, **(prover.launch(60) if args.prover == 'vampire' else prover.plain(60))})
    elif args.kind == 'sine':
        sine_stage(prover, path, args, row)
    else:
        guided(prover, path, args, row)

    row['seconds'] = round(time.monotonic() - started, 3)
    return row

def output_row(result):
    if 'stages' not in result:
        return {'problem': Path(result['problem']).name, 'stages': '[]', 'error': result.get('outcome', 'error'), 'seconds': round(result['seconds'], 3)}

    selected = result.pop('selected', None)
    if selected is not None:
        result['score_error'] = result.get('score_error') or ''
        result['axioms'] = f"{result['axioms']}/{selected}"

    result['stages'] = json.dumps(result['stages'])
    return result

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'run schedule stages once per problem and record their times to proof')
    parser.add_argument('--prover', choices = ['prover9', 'iprover', 'eprover', 'vampire'], required = True)
    parser.add_argument('--kind', choices = kinds, required = True)
    parser.add_argument('--manifest', type = Path, required = True)
    parser.add_argument('--output', type = Path, required = True, help = 'CSV, resumed when it exists')
    parser.add_argument('--tptp', default = '/rds/user/msf43/hpc-work/TPTP')
    parser.add_argument('--model', help = 'checkpoint (.ts); for Vampire a chunked export')
    parser.add_argument('--shuffled', action = 'store_true', help = 'uniform random ranking instead of the model')
    parser.add_argument('--random', help = 'random-score .ts for shuffled Vampire stages')
    parser.add_argument('--sine', default = '/rds/user/msf43/hpc-work/frontier-gaps-20261109/models/sine')
    parser.add_argument('--vampire', help = 'Vampire binary')
    parser.add_argument('--support-s', type = float, default = 55)
    parser.add_argument('--score-cap', type = float, default = 60)
    parser.add_argument('--workers', type = int, default = 32)
    parser.add_argument('--count', type = int, default = 0)
    parser.add_argument('--skip', type = Path, help = 'JSON lists of problems not to open (files on a hanging Lustre target)')
    return parser.parse_args()

def main():
    from axiom_prediction.search_workers import supervised_results

    args = parse_arguments()
    manifest = json.loads(args.manifest.read_text())
    paths = manifest.get('paths') or manifest['problems']
    paths = paths[:args.count] if args.count else paths
    done = {r['problem'] for r in csv.DictReader(args.output.open())} if args.output.exists() else set()
    todo = [p for p in paths if Path(p).name not in done]
    skip = {p for paths_ in json.loads(args.skip.read_text()).values() for p in paths_} if args.skip else set()
    skipped = [p for p in todo if p in skip]
    todo = [p for p in todo if p not in skip]
    workers = min(args.workers, len(os.sched_getaffinity(0)))
    print(f'{args.prover} {args.kind}{" shuffled" if args.shuffled else ""}: {len(todo)} to do, {len(done)} done, {workers} workers', flush = True)

    # Covers the longest per-problem sequence (scoring cap plus 20 + 8 + 8 s, or 60 s plain) with room for loading.
    problem_limit_s = args.score_cap + 120
    args.output.parent.mkdir(parents = True, exist_ok = True)
    counts = {'finished': 0, 'proved': 0, 'errors': 0}
    results = supervised_results(solve, [(p, args) for p in todo], workers = workers, timeout = problem_limit_s)
    # Closing the generator stops the workers, so a failed write (RDS went read-only once) ends the run instead of hanging it.
    try:
        with args.output.open('a', newline = '') as handle:
            writer = csv.DictWriter(handle, fieldnames = fields, extrasaction = 'ignore')
            if not done:
                writer.writeheader()

            # Reading these problems hangs on compute nodes (Lustre target 114, 2026-11-11); record them without opening them.
            for path in skipped:
                writer.writerow({'problem': Path(path).name, 'stages': '[]', 'error': 'SkippedOST114', 'seconds': 0})

            for row in map(output_row, results):
                writer.writerow(row)
                handle.flush()
                counts['finished'] += 1
                counts['proved'] += any(s['proved'] for s in json.loads(row['stages']))
                counts['errors'] += bool(row.get('error') or row.get('score_error'))
                if counts['finished'] % 100 == 0:
                    print(counts, flush = True)
    finally:
        results.close()

    print(f'finished: {counts}', flush = True)

if __name__ == '__main__':
    main()
