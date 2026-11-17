# Compares the rebuilt Prover9 Support1 row for Sharbaraz5 e20 with the frontier run, problem by problem.
import csv
from pathlib import Path
import sys

sys.path.insert(0, '/rds/user/msf43/hpc-work/epoch-stages-20261111')
from simulate_stages import load, prover9_support1

run = Path('/rds/user/msf43/hpc-work/epoch-stages-20261111')
problems = load('prover9', run / 'results/controls', run / 'results/Sharbaraz5/e20')
frontier = {r['problem']: r for r in csv.DictReader(open('/rds/user/msf43/hpc-work/frontier-gaps-20261109/prover9/s5/results-Support1.csv'))}
base = {r['problem']: r for r in csv.DictReader(open('/rds/user/msf43/hpc-work/frontier-gaps-20261109/prover9/s5/results-Baseline.csv'))}
simulated = {p.name for p in problems.values() if prover9_support1(p)}
actual = {k for k, r in frontier.items() if r['proved'] == 'True'}
print('simulated', len(simulated), 'frontier', len(actual), 'sim only', len(simulated - actual), 'frontier only', len(actual - simulated))
for name in sorted(simulated ^ actual):
    p, r = problems[name], frontier[name]
    support = p.stages.get('support')
    print(name, 'sim' if name in simulated else 'run', 'axioms', p.axioms, 'score', p.score_s, 'plain', p.plain, 'frontier plain', base[name]['time_ms'], base[name]['proved'],
          'support', support, '| run', r['result'], r['time_ms'], r['score_ms'], r['error'][:40])
