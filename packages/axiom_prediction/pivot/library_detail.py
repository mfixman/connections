# Details behind library_table.py: stage successes per arm and cut-off, learned against symbol overlap paired, the
# full-graph ranking against the re-ranked prefilter, and the clausifier check. usage: library_detail.py OUTPUT_DIR
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library_table import mcnemar, method_arm, schedules, within

def main():
    output = Path(sys.argv[1])
    prepared = {name: p for name, p in json.loads((output / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (output / 'results.jsonl').open()):
        results[(row['problem'], row['prover'])][(row['arm'], row['k'])] = row

    provers = sorted({prover for _, prover in results})
    print('== proofs within 8 s per stage (problems where the stage ran)')
    for prover in provers:
        cells = []
        for arm, sizes in (('learned', (16, 64, 256, 1024)), ('rerank', (16, 64, 256)), ('overlap', (16, 64, 256, 1024)), ('shuffled', (16, 64, 256)), ('sine', (0,)), ('clausified', (0,))):
            for k in sizes:
                runs = [rows[(arm, k)] for (name, p), rows in results.items() if p == prover and (arm, k) in rows]
                cells.append(f'{arm}{k or ""} {sum(within(run, 8) for run in runs)}/{len(runs)}')

        print(f'{prover:8s} ' + '  '.join(cells))

    print('\n== learned (method) against symbol overlap, paired: ladder at 60 s, and top-k stages alone')
    for prover in provers:
        ladder, stages = [0, 0], [0, 0]
        for name in prepared:
            out = schedules(results[(name, prover)], prepared[name], 60)
            ladder[0] += out['ladder-method'] and not out['ladder-overlap']
            ladder[1] += out['ladder-overlap'] and not out['ladder-method']
            stages[0] += out['stages-method'] and not out['stages-overlap']
            stages[1] += out['stages-overlap'] and not out['stages-method']

        print(f'{prover:8s} ladder: learned only {ladder[0]}, overlap only {ladder[1]} (p = {mcnemar(*ladder):.3g}); stages: learned only {stages[0]}, overlap only {stages[1]} (p = {mcnemar(*stages):.3g})')

    both = [name for name, p in prepared.items() if 'learned' in p['rankings'] and 'rerank' in p['rankings']]
    whole = [name for name, p in prepared.items() if 'learned' in p['rankings']]
    print(f'\n== {len(whole)} problems with at most 2048 axiom clauses (network on the whole problem), 60 s')
    for prover in provers:
        counts = defaultdict(int)
        for name in whole:
            rows = results[(name, prover)]
            plain = rows.get(('plain', 0))
            counts['plain'] += within(plain, 60)
            for arm in ('learned', 'overlap', 'shuffled'):
                stage_hit = any(within(rows.get((arm, k)), 8) for k in (16, 64, 256))
                counts[f'stages-{arm}'] += stage_hit
                counts[f'new-{arm}'] += stage_hit and not within(plain, 60)

        print(f'{prover:8s} ' + ', '.join(f'{key} {value}' for key, value in counts.items()))

    print(f'\n== {len(both)} problems with 1025-2048 axiom clauses: whole-problem ranking against the re-ranked prefilter (any top-k stage)')
    for prover in provers:
        full = sum(any(within(results[(name, prover)].get(('learned', k)), 8) for k in (16, 64, 256)) for name in both)
        pre = sum(any(within(results[(name, prover)].get(('rerank', k)), 8) for k in (16, 64, 256)) for name in both)
        print(f'{prover:8s} whole problem {full}, re-ranked prefilter {pre}')

    print('\n== any prover, 60 s')
    for label in ('plain', 'ladder-method', 'ladder-overlap', 'ladder-shuffled', 'sine+plain', 'bushy'):
        print(f'{label}: {sum(any(schedules(results[(name, prover)], prepared[name], 60)[label] for prover in provers) for name in prepared)}')

    print('\n== clausifier check, 10 s: original FOF against the Connections clause set (TPTP provers)')
    for prover in provers:
        if prover != 'Prover9':
            fof = sum(within(results[(name, prover)].get(('plain', 0)), 10) for name in prepared)
            cnf = sum(within(results[(name, prover)].get(('clausified', 0)), 10) for name in prepared)
            print(f'{prover:8s} original {fof}, Connections clauses {cnf}')

if __name__ == '__main__':
    main()
