# Paired comparisons of the dense ladders over the given provers together (default Prover9 and E), for the rules
# fixed in NOTES.md. usage: dense_pairs.py OUTPUT_DIR [PROVER...]
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_table import ladders, proves
from library_table import mcnemar

def main():
    output, provers = Path(sys.argv[1]), sys.argv[2:] or ['Prover9', 'E']
    prepared = {name: p for name, p in json.loads((output / 'prepared.json').read_text()).items() if 'rankings' in p}
    results = defaultdict(dict)
    for row in map(json.loads, (output / 'results.jsonl').open()):
        results[(row['problem'], row['prover'])][(row['arm'], row['k'])] = row

    def solved(arm, label):
        rungs, slice_s = ladders[label]
        return {
            (prover, name)
            for prover in provers
            for name in prepared
            if proves(results[(name, prover)], arm, rungs, slice_s, prepared[name]['timings'].get('learned', 0))
        }

    standard = solved('learned', 'standard 16/64/256 x 8 s')
    for label in ladders:
        learned = solved('learned', label)
        cells = [f'{label:26s} learned {len(learned)}, against the standard ladder +{len(learned - standard)}/-{len(standard - learned)}']
        for other in ('shuffled', 'overlap'):
            control = solved(other, label)
            gains, losses = len(learned - control), len(control - learned)
            cells.append(f'against {other} +{gains}/-{losses} (p = {mcnemar(gains, losses):.2g})')

        print('; '.join(cells))

if __name__ == '__main__':
    main()
