# Which short learned ladders do as well as the 13-step one? Part 8 only, rebuilt from the sine/p8-* stage runs.
from itertools import combinations
from pathlib import Path
import sys

run = Path('/rds/user/msf43/hpc-work/pivot-20261114')
sys.path.insert(0, str(run))
from ladder_table import dense, proves
from sine_pool import groups, load

def main():
    data = {}
    for prover, group in groups.items():
        data[prover] = [load(run / place, prover) for place in ('sine/p8-cnf', f'sine/p8-{group}')]

    def count(prover, cutoffs, slice_s):
        stages = [(('learned', k), slice_s) for k in cutoffs]
        return sum(
            proves(results[name], stages, prepared[name]['timings'].get('learned', 0), 'plain')
            for prepared, results in data[prover]
            for name in prepared
            if ('plain', 0) in results[name]
        )

    provers = list(groups)
    fixed = {'plain': ((), 8), '16/64/256 x 8 s': ((16, 64, 256), 8), '13 cut-offs x 2 s': (dense, 2)}
    print('part 8, 1,266 problems; ' + ' / '.join(provers))
    for label, (cutoffs, slice_s) in fixed.items():
        print(f'{label:28s}', [count(p, cutoffs, slice_s) for p in provers])

    for size, slices in ((3, (4, 8)), (4, (4, 6)), (5, (3, 5))):
        best = []
        for cutoffs in combinations(dense, size):
            for slice_s in slices:
                totals = [count(p, cutoffs, slice_s) for p in provers]
                best.append((sum(totals), cutoffs, slice_s, totals))

        for total, cutoffs, slice_s, totals in sorted(best, reverse = True)[:3]:
            print(f'{size} cut-offs {cutoffs} x {slice_s} s: {totals} (sum {total})')

if __name__ == '__main__':
    main()
