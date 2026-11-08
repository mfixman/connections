# Part-9 Prover9 table: files, families, paired gains/losses and exact McNemar p against the baseline.
import argparse
from pathlib import Path

import paths
import results

table = [
    ('part9-ladder-s5e20', 'ResumeLadder', 'Ladder, learned'),
    ('part9-ladder-s5e20', 'ShuffledResumeLadder', 'Ladder, shuffled'),
    ('part9-ladder-overlap', 'ResumeLadder', 'Ladder, overlap'),
    ('part9-ladder-sine', 'ResumeLadder', 'Ladder, SInE'),
    ('part9-ladder-s5e20', 'ResumeLadderSupport1', 'LadderSupport1, learned'),
    ('part9-ladder-s5e20', 'ShuffledResumeLadderSupport1', 'LadderSupport1, shuffled'),
    ('part9-ladder-overlap', 'ResumeLadderSupport1', 'LadderSupport1, overlap'),
    ('part9-ladder-sine', 'ResumeLadderSupport1', 'LadderSupport1, SInE'),
    ('part9-sharbaraz5e20', 'ResumeFilter16Support1', 'F16S1, learned (earlier)'),
]

def file_name(row):
    return Path(row['problem']).name

def load(prover9, folder, mode):
    path = prover9 / folder / f'results-{mode}.csv'
    return results.proved(path, file_name), len(results.rows(path))

def main():
    parser = argparse.ArgumentParser(description = 'Prover9 part-9 results table with McNemar tests')
    parser.add_argument('--prover9', type = Path, default = paths.goal / 'prover9', help = 'folder of the Prover9 result directories')
    args = parser.parse_args()
    base, n = load(args.prover9, 'part9-sharbaraz5e20', 'Baseline')
    base_families = len(results.families(base))
    print(f'baseline {len(base)} files, {base_families} families, {n} problems')
    for folder, mode, label in table:
        s, _ = load(args.prover9, folder, mode)
        g, l = len(s - base), len(base - s)
        counts = f'files {len(s)} ({results.pct(len(s), len(base))}) fam {len(results.families(s))} ({results.pct(len(results.families(s)), base_families)})'
        print(f'{label:28s} {counts} gains/losses {g}/{l} McNemar p={results.mcnemar(g, l):.4f}')

    learned, _ = load(args.prover9, 'part9-ladder-s5e20', 'ResumeLadder')
    shuffled, _ = load(args.prover9, 'part9-ladder-s5e20', 'ShuffledResumeLadder')
    overlap, _ = load(args.prover9, 'part9-ladder-overlap', 'ResumeLadder')
    sine, _ = load(args.prover9, 'part9-ladder-sine', 'ResumeLadder')
    p = results.mcnemar(len(learned - shuffled), len(shuffled - learned))
    print('ladder learned vs shuffled', len(learned - shuffled), len(shuffled - learned), f'p={p:.4f}', '| vs overlap', len(learned - overlap), len(overlap - learned), '| vs SInE', len(learned - sine), len(sine - learned))

main()
