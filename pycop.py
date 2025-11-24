import sys
import os 
import subprocess
import signal
import traceback
from os.path import dirname, abspath

from connections.env import *
import argparse

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description = 'Python equivalent of version 1.0f of leanCoP, ileanCoP, and mleanCoP')
    parser.add_argument('--logic', default = 'classical', type = Logic, help = "Which logic")
    parser.add_argument('--domain', default = 'constant', help = "Which domain")
    parser.add_argument('--translate', action = 'store_true', help = 'Whether to translate the logic with Prolog.')
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

    print(errors, file = sys.stderr)
    return output

def main():
    args = parse_args()

    env = ConnectionEnv(args.file, Settings(logic = args.logic, domain = args.domain))

    observation = env.reset()

    done = False
    while not done:
        action = env.action_space[0]
        print(action)
        observation, reward, done, info = env.step(action)

    print(info)

if __name__ == '__main__':
    main()
