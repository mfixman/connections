# Health probe for compute nodes: imports torch, loads and scores two TPTP problems, runs one Prover9 search and one Vampire run.
set -u
export TMPDIR=/dev/shm
started=$(date +%s)
R=/rds/user/msf43/hpc-work/epoch-stages-20261111
timeout 600 /home/msf43/venv/bin/python -u -c "
import sys, time; t = time.time()
sys.path[:0] = ['/home/msf43/connections/external/Prover9/axiom_selection', '/home/msf43/connections/packages', '/home/msf43/connections/src']
import torch; print('torch', round(time.time() - t, 1), flush = True)
from problem import load_problem, clause_text
from predict import Predictor
from portfolio import search
p = Predictor('/home/msf43/connections/results/big-train/Sharbaraz15/model/epoch-0010.ts'); print('model', round(time.time() - t, 1), flush = True)
for name in ('ALG/ALG343-1.p', 'BOO/BOO017-2.p'):
    m = load_problem('/rds/user/msf43/hpc-work/TPTP/Problems/' + name, '/rds/user/msf43/hpc-work/TPTP')
    p.predict(m)
    r = search([clause_text(c) for c in m.clauses], range(len(m.clauses)), 2, '/dev/shm/probe-' + str(time.time()))
    print(name, 'loaded, scored, searched', round(time.time() - t, 1), flush = True)
"
echo "python exit $?"
TPTP=/rds/user/msf43/hpc-work/TPTP timeout 120 $R/bin/vampire --mode vampire --time_limit 5s --torok_file $R/models/Sharbaraz15-e10-chunked.ts --sos all /rds/user/msf43/hpc-work/TPTP/Problems/ALG/ALG044+1.p > /dev/null 2>&1
echo "vampire exit $?"
echo "PROBE DONE in $(( $(date +%s) - started )) s"
