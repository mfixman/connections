import sys
import os 
import subprocess
import signal
import traceback
from os.path import dirname, abspath

import random
import logging

from connections.env import *
from parse_args import parse_pycop_args
from problem_inputs import resolved_problem_paths, split_problem_paths

from pathlib import Path

from strategies import *

def default_logic_for_strategy(strategy_name: str) -> Logic:
    if strategy_name.startswith('SATSolve') or strategy_name == 'SATCoPRust':
        return Logic.ClassicalSAT
    if strategy_name == 'FirstAction':
        return Logic.Classical
    return Logic.ClassicalCut

def resolve_logic_for_strategy(strategy_name: str, logic_override: Logic | None) -> Logic:
    if logic_override is not None:
        return logic_override
    return default_logic_for_strategy(strategy_name)

def translate_logic(file: str, logic: Logic) -> str:
    translator_path = Path('translation') / str(logic) / 'translate.sh'

    problem = os.path.basename(os.path.normpath(file))
    with subprocess.Popen([translator_path, file, problem], preexec_fn = os.setsid) as process:
        try:
            output, errors = process.communicate(timeout = 1)
        except subprocess.TimeoutExpired as err:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    return str(output)

def _final_closed_tableau(state) -> str | None:
    tableau = getattr(state, 'tableau', None)
    if tableau is None:
        return None

    if getattr(tableau, 'children', None):
        return '\n'.join(str(child) for child in tableau.children)

    if getattr(tableau, 'literal', None) is not None:
        return str(tableau)

    return None

def _format_debug_info(state, info, debug: bool):
    if not debug or info != 'Theorem':
        return info

    closed_tableau = _final_closed_tableau(state)
    if not closed_tableau:
        return info

    return f'Theorem\n{closed_tableau}'

def _print_result(result: dict, problem: str | None = None) -> None:
    payload = ({'problem': problem} | result) if problem is not None else result
    info = payload.get('info')

    if not isinstance(info, str) or '\n' not in info:
        print(payload)
        return

    summary = dict(payload)
    del summary['info']
    print(summary)
    print(info)

def _run_problem(path: str, args) -> dict:
    logic = resolve_logic_for_strategy(args.strategy, args.logic)
    env = ConnectionEnv(
        path,
        Settings(
            logic = logic,
            domain = args.domain,
            sat_solver = args.sat_solver,
            debug = args.debug,
        ),
    )
    strategy = Strategy.strategies()[args.strategy](env)
    observation = env.reset()

    done, info = env.state.initial_status()

    steps = 0
    while not done:
        if steps >= args.max_steps:
            info = {'Solution': 'Unknown'}
            break

        action = strategy.step()

        if args.print_ratio > 0 and random.randint(0, args.print_ratio - 1) == 0:
            print(action)
            if info:
                print(info)

        try:
            observation, reward, done, info = env.step(action)
        except RecursionError:
            logging.error('Recursion error.')
            steps = args.max_steps - 1

        steps += 1

    info = _format_debug_info(env.state, info, args.debug)

    if info is None:
        info = {}
    elif type(info) is str:
        info = {'info': info}

    return info | {'steps': steps}

def main():
    args = parse_pycop_args()
    logging.basicConfig(level = logging.INFO if args.verbose else logging.WARN, format = '[%(relativeCreated)d] %(message)s')

    try:
        with resolved_problem_paths(args.file) as problem_paths:
            problem_paths = split_problem_paths(problem_paths, args.split, args.part)
            multiple = len(problem_paths) > 1
            for path in problem_paths:
                result = _run_problem(path, args)
                _print_result(result, os.path.basename(path) if multiple else None)
    except FileNotFoundError as exc:
        sys.exit(f'error: {exc}')

if __name__ == '__main__':
    main()
