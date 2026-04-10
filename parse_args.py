import argparse
from strategies import *
from connections.env import *

def parse_args():
    parser = argparse.ArgumentParser(description = 'Python equivalent of version 1.0f of leanCoP, ileanCoP, and mleanCoP')
    parser.add_argument('--logic', default = Logic.Classical, choices = list(Logic), type = Logic, help = "Which logic")
    parser.add_argument('--domain', default = Domain.Constant, choices = list(Domain), type = Domain, help = "Which domain")
    parser.add_argument(
        '--sat-solver',
        default = 'pycosat',
        choices = SAT_SOLVER_CHOICES,
        help = 'SAT backend for ClassicalSAT/SearchSAT variants',
    )
    parser.add_argument('--translate', action = 'store_true', help = 'Whether to translate the logic with Prolog.')
    parser.add_argument('--print-ratio', '-pr', type = int, default = 1, help = 'Ratio of messages to be printed')
    parser.add_argument('--max-steps', default = 100000, type = int, help = 'Maximum amount of steps before breaking.')
    parser.add_argument('--verbose', '-v', action = 'store_true', help = 'Verbose messages.')
    parser.add_argument('--debug', action = 'store_true', help = 'Print extra debug information.')
    parser.add_argument('--strategy', default = 'FirstAction', choices = Strategy.strategies().keys(), help = 'Strategy to use')
    parser.add_argument(
        "file",
        help = "Problem input: local file, directory of .p files, TPTP codename/path, or raw quoted TPTP text",
    )
    return parser.parse_args()
