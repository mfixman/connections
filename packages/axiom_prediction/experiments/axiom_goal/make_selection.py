# Writes the refutable TPTP problems of one split-10 part, as the validation backfill selects them.
import argparse
import json
from pathlib import Path

import paths

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'write the SATCoP problem selection of one split-10 part')
    parser.add_argument('part', type = int)
    parser.add_argument('output')
    parser.add_argument('--connections', type = Path, default = paths.connections)
    parser.add_argument('--tptp', type = Path, default = paths.tptp)
    return parser.parse_args()

def main():
    args = parse_arguments()
    paths.add_connections(args.connections)
    from axiom_prediction.split import ProblemSplit
    from axiom_prediction.tptp import tptp_problems
    from axiom_prediction.validation_run import refutable

    split = ProblemSplit(split = 10, parts = [args.part], scheme = 5)
    problems = sorted(filter(refutable, split.select(tptp_problems(tptp_root = args.tptp))))
    json.dump({'held_out': len(problems), 'problems': problems, 'split': split.describe()}, open(args.output, 'w'))
    print(len(problems), 'problems')

main()
