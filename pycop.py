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

@ipdb.launch_ipdb_on_exception()
def main():
    args = parse_args()
    logging.basicConfig(level = logging.INFO if args.verbose else logging.WARN, format = '[%(relativeCreated)d] %(message)s')

    env = ConnectionEnv(args.file, Settings(logic = args.logic, domain = args.domain))
    strategy = Strategy.strategies()[args.strategy](env)
    observation = env.reset()

    done = False
    info = None

    steps = 0
    while not done:
        if steps >= args.max_steps:
            info = {'Solution': 'Unknown'}
            break

        action = strategy.step()

        if args.print_ratio > 0 and random.randint(0, args.print_ratio - 1) == 1:
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

    print(info | {'steps': steps})

if __name__ == '__main__':
    main()
