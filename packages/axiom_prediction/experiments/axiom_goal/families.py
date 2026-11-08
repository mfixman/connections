# Per-file and per-family (TPTP stem, e.g. SYN003) proof counts for result CSVs; the first CSV is the baseline.
import argparse
from pathlib import Path

import results

def file_name(row):
    return Path(row['problem']).name

def main():
    parser = argparse.ArgumentParser(description = 'file and family proof counts per result CSV')
    parser.add_argument('csv', nargs = '+', help = 'the first CSV is the baseline')
    args = parser.parse_args()
    base = results.proved(args.csv[0], file_name)
    base_families = results.families(base)
    total = len(results.families(map(file_name, results.rows(args.csv[0]))))
    for path in args.csv:
        files = results.proved(path, file_name)
        fams = results.families(files)
        counts = f'files {len(files)} ({results.pct(len(files), len(base))}), families {len(fams)}/{total} ({results.pct(len(fams), len(base_families))})'
        print(f'{Path(path).stem}: {counts}, new families {len(fams - base_families)}, lost {len(base_families - fams)}')

main()
