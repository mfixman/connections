import sys
import os 
import subprocess
import signal
import traceback
from os.path import dirname, abspath

import random
import logging

from connections.env import *
import argparse

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description = 'Python equivalent of version 1.0f of leanCoP, ileanCoP, and mleanCoP')
    parser.add_argument('--logic', default = Logic.Classical, choices = list(Logic), type = Logic, help = "Which logic")
    parser.add_argument('--domain', default = Domain.Constant, choices = list(Domain), type = Domain, help = "Which domain")
    parser.add_argument('--translate', action = 'store_true', help = 'Whether to translate the logic with Prolog.')
    parser.add_argument('--print-ratio', '-pr', type = int, default = 1, help = 'Ratio of messages to be printed')
    parser.add_argument('--max-steps', default = 100000, type = int, help = 'Maximum amount of steps before breaking.')
    parser.add_argument('--verbose', '-v', action = 'store_true', help = 'Verbose messages.')
    parser.add_argument("file", help = "The conjecture you want to prove")
    return parser.parse_args()

def translate_logic(file: str, logic: Logic) -> str:
    translator_path = Path('translation') / str(logic) / 'translate.sh'

    problem = os.path.basename(os.path.normpath(args.file))
    with subprocess.Popen([translator_path, args.file, problem], preexec_fn = os.setsid) as process:
        try:
            output, errors = process.communicate(timeout = 1)
        except subprocess.TimeoutExpired as err:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

    return output

def main():
    args = parse_args()
    logging.basicConfig(level = logging.INFO if args.verbose else logging.WARN, format = '[%(relativeCreated)d] %(message)s')

    env = ConnectionEnv(args.file, Settings(logic = args.logic, domain = args.domain))
    observation = env.reset()

    done = False
    info = None

    steps = 0
    while not done:
        if args.max_steps is not None and steps >= args.max_steps:
            info = {'Solution': 'Unknown'}
            break

        action = env.action_space[0]
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

    print(info | {'steps': steps})

if __name__ == '__main__':
    main()
