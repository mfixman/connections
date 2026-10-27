import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', type = Path)
    parser.add_argument('--prover9', type = Path, required = True)
    parser.add_argument('--tptp', type = Path, required = True)
    parser.add_argument('--output', type = Path, required = True)
    parser.add_argument('--seconds', type = float, required = True)
    parser.add_argument('--memory-mb', type = int, default = 512)
    args = parser.parse_args()
    started = time.monotonic()
    sys.path.insert(0, str(args.prover9 / 'axiom_selection'))
    from problem import load_problem, clause_text
    from portfolio import native_input

    try:
        matrix = load_problem(args.problem, args.tptp)
        clauses = list(map(clause_text, matrix.clauses))
        remaining = max(.001, args.seconds - (time.monotonic() - started))
        text = native_input(clauses, list(range(len(clauses))), remaining)
        text = text.replace('assign(max_megs,512).', f'assign(max_megs,{args.memory_mb}).')
        source = args.output / 'native.in'
        source.write_text(text)
        command = [str(args.prover9 / 'axiom_selection/build/native-search'),
                   str(args.prover9 / 'provers.src/prover9'), str(source),
                   str(args.output / 'native.log'), str(remaining)]
        result = subprocess.run(command, text = True, capture_output = True, check = True)
        value = json.loads(result.stdout)
        value['clauses'] = len(clauses)
    except Exception as error:
        value = dict(proved = False, error = repr(error))

    value['elapsed_s'] = time.monotonic() - started
    temporary = args.output / 'native-result.tmp'
    temporary.write_text(json.dumps(value, indent = 2) + '\n')
    temporary.replace(args.output / 'native-result.json')

if __name__ == '__main__':
    main()
