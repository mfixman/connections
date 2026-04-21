import argparse
from pathlib import Path

from connections.env import Domain, Logic, SATSolver
from problem_inputs import add_problem_split_arguments, validate_problem_split_arguments
from strategies import Strategy
from strategies.groups import expand_strategy_groups, strategy_choices, strategy_group_names


def _add_common_prover_arguments(parser: argparse.ArgumentParser, *, max_steps_default: int) -> None:
    parser.add_argument('--logic', default=None, choices=list(Logic), type=Logic, help='Which logic')
    parser.add_argument(
        '--sat-solver',
        default=None,
        choices=list(SATSolver),
        type=SATSolver,
        help='SAT backend for ClassicalSAT',
    )
    parser.add_argument('--max-steps', default=max_steps_default, type=int, help='Maximum amount of steps before breaking.')


def _add_pycop_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_prover_arguments(parser, max_steps_default=100000)
    parser.add_argument('--domain', default=Domain.Constant, choices=list(Domain), type=Domain, help='Which domain')
    parser.add_argument('--translate', action='store_true', help='Whether to translate the logic with Prolog.')
    parser.add_argument('--print-ratio', '-pr', type=int, default=0, help='Ratio of messages to be printed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose messages.')
    parser.add_argument('--debug', action='store_true', help='Print extra debug information.')
    parser.add_argument('--strategy', default='FirstAction', choices=Strategy.strategies().keys(), help='Strategy to use')
    add_problem_split_arguments(parser)
    parser.add_argument(
        'file',
        help='Problem input: local file, directory of .p files, TPTP codename/path, or raw quoted TPTP text',
    )


def _input_basename(inputs: list[str]) -> str:
    if not inputs:
        return 'all'
    return Path(inputs[0]).name


def _named_result_stem(inputs: list[str], split: int, part: int) -> str:
    suffix = '' if split == 1 else f'_{part}'
    return f'{_input_basename(inputs)}{suffix}'


def _default_partial_file(inputs: list[str], split: int, part: int) -> str:
    return str(Path('partial') / _named_result_stem(inputs, split, part))


def _named_partial_file(name: str, inputs: list[str], split: int, part: int) -> str:
    stem = _named_result_stem(inputs, split, part)
    return str(Path('partial') / f'{name}_{stem}')


def _named_csv_file(name: str, inputs: list[str], split: int, part: int) -> str:
    return str(Path('results') / name / f'{_named_result_stem(inputs, split, part)}.csv')


def _add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    available_strategies = list(Strategy.strategies().keys())
    default_strategies = [available_strategies[0], available_strategies[-1]]
    available_choices = strategy_choices(available_strategies)

    parser.add_argument(
        'inputs', nargs='*', metavar='DIR_OR_FILE',
        help='directories and/or problem files/codenames (default: all $TPTP/Problems/*/*.p)',
    )

    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument(
        '--strategies', nargs='+',
        choices=available_choices, metavar='STRATEGY_OR_GROUP',
        help=f'strategies or groups to compare (default: {" ".join(default_strategies)})',
    )
    strategy_group.add_argument(
        '--all-strategies', action='store_true',
        help='compare all available strategies',
    )
    _add_common_prover_arguments(parser, max_steps_default=50000)
    parser.add_argument(
        '--timeout', type=float, default=30.0, metavar='SEC',
        help='timeout per problem in seconds (default: 30)',
    )
    parser.add_argument(
        '--steps', default=True, action=argparse.BooleanOptionalAction,
        help='show step counts in results (default: on)',
    )
    parser.add_argument(
        '--time', default=True, action=argparse.BooleanOptionalAction,
        help='show wall-clock time in results (default: on)',
    )
    parser.add_argument(
        '--csv', default=None, metavar='PATH',
        help='write final results as CSV to PATH (required unless --name is provided)',
    )
    parser.add_argument(
        '--partial-file', metavar='PATH',
        help='append report output to PATH and resume from any completed rows already present '
             '(default: partial/INPUT_PART)',
    )
    parser.add_argument(
        '--name', metavar='NAME',
        help='derive default --partial-file and --csv paths from NAME and the input basename',
    )
    parser.add_argument(
        '--color', '--colour', dest='color', action='store_true',
        help='preserve ANSI colour codes in --partial-file output (default: off)',
    )
    add_problem_split_arguments(parser)


def _compare_epilog() -> str:
    available_strategies = list(Strategy.strategies().keys())
    return (
        f'available strategies: {", ".join(available_strategies)}\n'
        f'available strategy groups: {", ".join(strategy_group_names())}\n'
        f'available logics:     {", ".join(str(logic) for logic in Logic)}\n'
        '\n'
        'examples:\n'
        '  %(prog)s --csv results.csv\n'
        '  %(prog)s tests/cnf_problems --csv results.csv\n'
        '  %(prog)s comparisons/classical --timeout 60 --max-steps 100000 --csv results.csv\n'
        '  %(prog)s tests --strategies FirstAction SmartCut --csv results.csv\n'
        '  %(prog)s tests --all-strategies --csv results.csv\n'
        '  %(prog)s tests --no-time --csv results.csv\n'
        '  %(prog)s tests --no-steps --csv results.csv\n'
        '  %(prog)s tests --split 4 --part 0 --csv results.csv\n'
        '  %(prog)s tests --name run1 --split 4 --part 2\n'
    )


def _build_pycop_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Python equivalent of version 1.0f of leanCoP, ileanCoP, and mleanCoP')
    _add_pycop_arguments(parser)
    return parser


def _build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='compare_strategies.py',
        description='Compare proof strategies across a set of problems.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_compare_epilog(),
    )
    _add_compare_arguments(parser)
    return parser


def _finalize_compare_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    available_strategies = list(Strategy.strategies().keys())
    default_strategies = [available_strategies[0], available_strategies[-1]]

    if args.all_strategies:
        args.strategies = available_strategies
    elif args.strategies is None:
        args.strategies = default_strategies
    else:
        args.strategies = expand_strategy_groups(args.strategies)

    if args.name is not None:
        if args.partial_file is None:
            args.partial_file = _named_partial_file(args.name, args.inputs, args.split, args.part)
        if args.csv is None:
            args.csv = _named_csv_file(args.name, args.inputs, args.split, args.part)
    else:
        if args.partial_file is None:
            args.partial_file = _default_partial_file(args.inputs, args.split, args.part)
        if args.csv is None:
            parser.error('--csv results file is required unless --name is provided')


def parse_pycop_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_pycop_parser()
    args = parser.parse_args(argv)
    validate_problem_split_arguments(args, parser)
    if args.sat_solver is None:
        strategy = Strategy.strategies().get(args.strategy)
        args.sat_solver = SATSolver(getattr(strategy, 'sat_solver_backend', SATSolver.CaDiCaL))
    return args


def parse_compare_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_compare_parser()
    args = parser.parse_args(argv)
    validate_problem_split_arguments(args, parser)
    _finalize_compare_args(args, parser)
    return args
