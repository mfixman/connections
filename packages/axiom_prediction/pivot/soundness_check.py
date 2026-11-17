# Soundness check of the clause sets given to external provers: first-order problems that TPTP marks CounterSatisfiable
# are clausified by Connections and must not be refuted. usage: soundness_check.py LIST SECONDS WORKERS
from multiprocessing import Pool
from pathlib import Path
import re
import sys
import tempfile

sys.path[:0] = ['/home/msf43/connections/packages', '/home/msf43/connections/src']
from axiom_prediction.prover_cores.collect_cores import run_prover
from axiom_prediction.prover_cores.provers import Prover, modules

tptp = '/rds/user/msf43/hpc-work/TPTP'

def check(task):
    from axiom_prediction.tptp import load_tptp_problem

    name, seconds = task
    # Files on the Lustre target that hung compute nodes are not opened, nor problems that include one.
    hanging = set(Path('/home/msf43/tmp/tptp-ost114.txt').read_text().split('\n'))
    path = f'{tptp}/Problems/{name}'
    if path in hanging or any(f'{tptp}/{file}' in hanging for file in re.findall(r"include\('([^']+)'", Path(path).read_text(errors = 'replace'))):
        return name, {'load': 'skipped: hanging Lustre file'}

    try:
        problem = load_tptp_problem(path, tptp_root = tptp)
    except Exception as error:
        return name, {'load': str(error)[:80]}

    out = {'clauses': len(problem.matrix.clauses)}
    with tempfile.NamedTemporaryFile('w', suffix = '.p', dir = '/dev/shm') as handle:
        modules[Prover.E].write_input(problem.matrix, set(problem.conjecture_clause_ids), True, handle.name, seconds)
        for prover in (Prover.E, Prover.Vampire):
            out[str(prover)] = modules[prover].proved(run_prover(modules[prover].command(handle.name, seconds), seconds + 2))

    return name, out

def main():
    names, seconds, workers = Path(sys.argv[1]).read_text().split(), int(sys.argv[2]), int(sys.argv[3])
    refuted, failed = [], 0
    with Pool(workers) as pool:
        for name, out in pool.imap_unordered(check, [(name, seconds) for name in names]):
            failed += 'load' in out
            if out.get('E') or out.get('Vampire'):
                refuted.append((name, out))

    print(f'{len(names)} CounterSatisfiable problems, {failed} not loaded, {len(refuted)} refuted from the Connections clause set')
    print('\n'.join(map(str, refuted[:40])))

if __name__ == '__main__':
    main()
