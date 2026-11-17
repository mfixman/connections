# Clausifies first-order TPTP problems with E (eprover --cnf) into one clause file per problem, so that selection and
# every prover can work on a clause form that does not come from this project.
# usage: eclausify.py PROBLEM_LIST OUTPUT_DIR WORKERS
from multiprocessing import Pool
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library_stages import flatten

tptp = '/rds/user/msf43/hpc-work/TPTP'
eprover = '/home/msf43/connections/external/eprover/bin/eprover'

def clausify(task):
    path, output = task
    target = Path(output) / Path(path).name
    flat = f'/dev/shm/eclausify-{Path(path).name}'
    flatten(path, tptp, flat)
    try:
        done = subprocess.run([eprover, '--cnf', '--tstp-format', '-s', flat], capture_output = True, text = True, timeout = 120)
        clauses = [line for line in done.stdout.splitlines() if line.startswith('cnf(')]
    except subprocess.TimeoutExpired:
        clauses = []
    finally:
        Path(flat).unlink(missing_ok = True)

    if clauses:
        target.write_text('\n'.join(clauses) + '\n')

    return path, len(clauses)

def main():
    paths, output, workers = Path(sys.argv[1]).read_text().split(), Path(sys.argv[2]), int(sys.argv[3])
    output.mkdir(parents = True, exist_ok = True)
    with Pool(workers) as pool:
        counts = dict(pool.imap_unordered(clausify, [(path, str(output)) for path in paths]))

    written = [path for path in paths if counts[path]]
    (output / 'problems.txt').write_text(''.join(f'{output}/{Path(path).name}\n' for path in written))
    print(f'{len(written)} of {len(paths)} problems clausified by E into {output}', flush = True)

if __name__ == '__main__':
    main()
