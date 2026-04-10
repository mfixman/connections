import sys
import os 
import subprocess
import signal
import traceback
from os.path import dirname, abspath

import random
import logging

from connections.env import *
from parse_args import parse_args
from problem_inputs import resolved_problem_paths

from pathlib import Path
import ipdb

from strategies import *

def translate_logic(file: str, logic: Logic) -> str:
    translator_path = Path('translation') / str(logic) / 'translate.sh'

    problem = os.path.basename(os.path.normpath(file))
    with subprocess.Popen([translator_path, file, problem], preexec_fn = os.setsid) as process:
        try:
            output, errors = process.communicate(timeout = 1)
        except subprocess.TimeoutExpired as err:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    return str(output)
def _run_problem(path: str, args) -> dict:
    env = ConnectionEnv(
        path,
        Settings(logic = args.logic, domain = args.domain, sat_solver = args.sat_solver),
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

    if info is None:
        info = {}
    elif type(info) is str:
        info = {'info': info}

    return info | {'steps': steps}

# @ipdb.launch_ipdb_on_exception()
def main():
    args = parse_args()
    logging.basicConfig(level = logging.INFO if args.verbose else logging.WARN, format = '[%(relativeCreated)d] %(message)s')

    try:
        with resolved_problem_paths(args.file) as problem_paths:
            multiple = len(problem_paths) > 1
            for path in problem_paths:
                result = _run_problem(path, args)
                if multiple:
                    print({'problem': os.path.basename(path)} | result)
                else:
                    print(result)
    except FileNotFoundError as exc:
        sys.exit(f'error: {exc}')

if __name__ == '__main__':
    main()
