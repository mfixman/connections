# Runs `axiom_predictor.py run` on the problems of a selection file, so arms pair with the backfill's results.
# Usage: run_selection.py SELECTION CHECKOUT [run arguments...]; the interpreter running this script is reused.
import json
import os
import sys

def main():
    selection, checkout, *rest = sys.argv[1:]
    problems = json.load(open(selection))['problems']
    python = sys.executable
    os.execv(python, [python, '-u', f'{checkout}/axiom_predictor.py', 'run', *rest, *problems])

main()
