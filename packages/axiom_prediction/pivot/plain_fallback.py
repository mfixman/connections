# Plain 60 s runs for problems the stage runner could not prepare (loading or scoring too slow, or out of memory).
# Provers that read first-order TPTP get the original file; iProver gets E's clausification; Prover9 gets E's
# clausification printed in LADR syntax. Rows have the library_stages.py format with arm "plain".
# usage: plain_fallback.py PROBLEM_LIST OUTPUT_DIR WORKERS PROVER...
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library_stages import command, flatten, proved
from axiom_prediction.prover_cores.collect_cores import run_prover
from axiom_prediction.prover_cores.provers import Prover, modules
from axiom_prediction.search_workers import supervised_results

tptp = '/rds/user/msf43/hpc-work/TPTP'
eprover = '/home/msf43/connections/external/eprover/bin/eprover'
budget_s = 60

def e_clauses(source, target):
    output = run_prover([eprover, '--cnf', '--tstp-format', '-s', source], 300)
    clauses = [line for line in output.splitlines() if line.startswith('cnf(')]
    if not clauses:
        raise RuntimeError('E produced no clauses')

    Path(target).write_text('\n'.join(clauses) + '\n')

def prover_input(prover, original, stem):
    if prover not in ('Prover9', 'iProver'):
        return original

    clauses = f'{stem}.ecnf.p'
    e_clauses(original, clauses)
    if prover == 'iProver':
        return clauses

    from axiom_prediction.tptp import load_tptp_problem
    problem = load_tptp_problem(clauses, tptp_root = tptp)
    target = f'{stem}.prover9'
    modules[Prover.Prover9].write_input(problem.matrix, set(problem.conjecture_clause_ids), False, target, budget_s)
    return target

def run(key):
    path, prover = key.split('|')
    stem = f'{tempfile.gettempdir()}/fallback-{os.getpid()}-{Path(path).name}'
    original = f'{stem}.fof'
    try:
        flatten(path, tptp, original)
        source = prover_input(prover, original, stem)
        started = time.monotonic()
        output = run_prover(command(prover, source, budget_s), budget_s + 1)
        return {'problem': key, 'proved': proved(prover, output), 'run_s': round(time.monotonic() - started, 3), 'kept': 0, 'status': 'fallback'}
    finally:
        for suffix in ('fof', 'ecnf.p', 'prover9'):
            Path(f'{stem}.{suffix}').unlink(missing_ok = True)

def main():
    paths, output, workers, provers = Path(sys.argv[1]).read_text().split(), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
    output.mkdir(parents = True, exist_ok = True)
    tempfile.tempdir = '/dev/shm'
    results = output / 'results.jsonl'
    done = {(r['problem'], r['prover']) for r in map(json.loads, results.open())} if results.exists() else set()
    tasks = [(f'{path}|{prover}',) for path in paths for prover in provers if (Path(path).name, prover) not in done]
    print(f'{len(tasks)} plain fallback runs, {len(done)} done, {workers} workers', flush = True)
    with results.open('a') as stream:
        for row in supervised_results(run, tasks, workers = workers, timeout = budget_s + 420):
            path, prover = row.pop('problem').split('|')
            row = {'problem': Path(path).name, 'prover': prover, 'arm': 'plain', 'k': 0, **row}
            if 'outcome' in row:
                row['error'] = row.pop('outcome')

            stream.write(json.dumps(row) + '\n')
            stream.flush()

    print('done', flush = True)

if __name__ == '__main__':
    main()
