"""Shared CLI and problem-input helpers for the retained prover surface."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence, TypeVar

from connections.env import Logic, SATSolver
from connections.utils.classical_problem_parsing import problem_metadata
from strategies import Strategy
from strategies.groups import expand_strategy_groups, strategy_choices, strategy_group_names

T = TypeVar('T')


def default_logic_for_strategy(strategy_name: str) -> Logic:
    if strategy_name == 'FirstAction':
        return Logic.Classical
    strategy = Strategy.strategies().get(strategy_name)
    if strategy is not None and strategy.__module__ == 'strategies.classical':
        return Logic.ClassicalCut
    return Logic.ClassicalSAT


def resolve_logic_for_strategy(strategy_name: str, logic_override: Logic | None) -> Logic:
    return default_logic_for_strategy(strategy_name)


def default_sat_solver_for_strategy(strategy_name: str) -> SATSolver:
    strategy = Strategy.strategies().get(strategy_name)
    if strategy is None:
        return SATSolver.CaDiCaL
    return SATSolver(getattr(strategy, 'sat_solver_backend', SATSolver.CaDiCaL))


def resolve_sat_solver_for_strategy(
    strategy_name: str,
    sat_solver_override: SATSolver | None,
) -> SATSolver:
    if sat_solver_override is not None:
        return sat_solver_override
    return default_sat_solver_for_strategy(strategy_name)


def add_problem_split_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--split',
        default=None,
        type=int,
        metavar='N',
        help='split the ordered problem list into N parts (default: 1)',
    )
    parser.add_argument(
        '--part',
        default=None,
        type=int,
        metavar='M',
        help='run only problems whose ordered index modulo --split is M (default: 0)',
    )


def validate_problem_split(split: int, part: int) -> None:
    if split <= 0:
        raise ValueError('--split must be greater than 0')
    if part < 0:
        raise ValueError('--part must be greater than or equal to 0')
    if part >= split:
        raise ValueError('--part must be less than --split')


def validate_problem_split_arguments(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if args.split is None and args.part is None:
        slurm_split = os.getenv('SLURM_ARRAY_TASK_COUNT')
        slurm_part = os.getenv('SLURM_ARRAY_TASK_ID')
        if slurm_split is not None and slurm_part is not None:
            try:
                args.split = int(slurm_split)
                args.part = int(slurm_part)
            except ValueError:
                parser.error(
                    'SLURM_ARRAY_TASK_COUNT and SLURM_ARRAY_TASK_ID must be integers '
                    'when used as --split/--part defaults'
                )
            print(
                'warning: using '
                f'SLURM_ARRAY_TASK_COUNT={args.split} as --split and '
                f'SLURM_ARRAY_TASK_ID={args.part} as --part',
                file=sys.stderr,
            )
        else:
            args.split = 1
            args.part = 0
    elif args.split is None:
        args.split = 1
    elif args.part is None:
        args.part = 0

    try:
        validate_problem_split(args.split, args.part)
    except ValueError as exc:
        parser.error(str(exc))


def split_problem_paths(paths: Sequence[T], split: int, part: int) -> list[T]:
    validate_problem_split(split, part)
    return [path for index, path in enumerate(paths) if index % split == part]


def configure_tptp_root(tptp_root: str | None) -> None:
    """Apply an explicit TPTP root override for this process."""
    if tptp_root is None:
        return

    root = Path(tptp_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'--tptp does not point to a directory: {root}')

    os.environ['TPTP'] = str(root)


def resolve_problem_path(path: str) -> str:
    problem_path = Path(path)

    for candidate in _candidate_paths(problem_path):
        if candidate.exists():
            return str(candidate)

    tptp_root = os.getenv('TPTP')
    if tptp_root is None:
        raise FileNotFoundError(
            f'could not find problem {path!r}; pass --tptp ROOT or set the TPTP environment variable'
        )

    matches = _find_tptp_matches(problem_path, Path(tptp_root))
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(f'could not find problem {path!r} under {tptp_root}')

    formatted = ', '.join(str(match) for match in matches[:5])
    if len(matches) > 5:
        formatted += ', ...'
    raise FileNotFoundError(f'problem {path!r} is ambiguous under {tptp_root}: {formatted}')


def expand_problem_inputs(inputs: list[str]) -> list[Path]:
    if not inputs:
        return _all_tptp_problem_paths()

    problems: list[Path] = []
    for raw in inputs:
        input_path = Path(raw)
        if input_path.exists() and input_path.is_dir():
            dir_problems = sorted(
                path for path in input_path.glob('*.p')
                if _is_supported_tptp_problem_path(path) and not problem_metadata(path).typed
            )
            if not dir_problems:
                raise FileNotFoundError(f'no supported .p files found in {raw!r}')
            problems.extend(dir_problems)
        else:
            problems.append(Path(resolve_problem_path(raw)))

    return _unique_paths_in_order([problem.resolve() for problem in problems])


@contextmanager
def resolved_problem_paths(raw_input: str):
    raw_suffix = _raw_tptp_suffix(raw_input)
    if raw_suffix is not None and not Path(raw_input).exists():
        with tempfile.TemporaryDirectory(prefix = 'raw-problem-') as temp_dir:
            temp_path = Path(temp_dir) / f'problem{raw_suffix}'
            temp_path.write_text(raw_input, encoding = 'utf-8')
            yield [str(temp_path)]
        return

    yield [str(path) for path in expand_problem_inputs([raw_input])]


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def _add_common_prover_arguments(parser: argparse.ArgumentParser, *, max_steps_default: int) -> None:
    parser.add_argument(
        '--logic',
        default=None,
        choices=list(Logic),
        type=Logic,
        help='deprecated override; effective logic is determined by the selected strategy',
    )
    parser.add_argument(
        '--sat-solver',
        default=None,
        choices=list(SATSolver),
        type=SATSolver,
        help='SAT backend for ClassicalSAT',
    )
    parser.add_argument(
        '--tptp',
        default=None,
        metavar='ROOT',
        help='TPTP root directory; if omitted, use the TPTP environment variable',
    )
    parser.add_argument('--max-steps', default=max_steps_default, type=int, help='Maximum amount of steps before breaking.')


def _add_pycop_arguments(parser: argparse.ArgumentParser) -> None:
    _add_common_prover_arguments(parser, max_steps_default=100000)
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
    _add_common_prover_arguments(parser, max_steps_default=1000000)
    parser.add_argument(
        '--timeout', type=float, default=90.0, metavar='SEC',
        help='timeout per problem in seconds (default: 90)',
    )
    parser.add_argument(
        '--num-workers', type=_positive_int, default=None, metavar='N',
        help='override worker process count (default: CPU count, capped by tasks)',
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
        '  %(prog)s tests --strategies FirstAction SATSolveLiteral --csv results.csv\n'
        '  %(prog)s tests --all-strategies --csv results.csv\n'
        '  %(prog)s tests --no-time --csv results.csv\n'
        '  %(prog)s tests --no-steps --csv results.csv\n'
        '  %(prog)s tests --split 4 --part 0 --csv results.csv\n'
        '  %(prog)s tests --name run1 --split 4 --part 2\n'
    )


def _build_pycop_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Python connection prover for classical, ClassicalCut, and ClassicalSAT search')
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
        args.sat_solver = default_sat_solver_for_strategy(args.strategy)
    return args


def parse_compare_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_compare_parser()
    args = parser.parse_args(argv)
    validate_problem_split_arguments(args, parser)
    _finalize_compare_args(args, parser)
    return args


def _strip_tptp_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('%'):
            continue
        lines.append(line)
    return '\n'.join(lines).strip()


def _raw_tptp_suffix(text: str) -> str | None:
    normalized = _strip_tptp_comments(text).lower()
    if not normalized:
        return None
    if 'fof(' in normalized:
        return '+1.p'
    if any(marker in normalized for marker in ('cnf(', 'include(')):
        return '-1.p'
    return None


def _candidate_paths(path: Path) -> list[Path]:
    candidates = [path]
    if path.suffix:
        return candidates

    candidates.append(path.with_suffix('.p'))
    return candidates


def _find_tptp_matches(path: Path, tptp_root: Path) -> list[Path]:
    matches = []

    for candidate in _candidate_paths(path):
        rooted = (tptp_root / candidate).resolve()
        if rooted.exists():
            matches.append(rooted)

    if matches:
        return _unique_paths(matches)

    for candidate in _candidate_paths(path):
        matches.extend(found.resolve() for found in tptp_root.rglob(candidate.name))
    return _unique_paths(matches)


def _unique_paths(paths: list[Path]) -> list[Path]:
    return sorted(set(paths))


def _unique_paths_in_order(paths: list[Path]) -> list[Path]:
    return list(dict.fromkeys(paths))


def _problem_priority_key(problem: Path) -> tuple[int, str, str]:
    if problem.name.startswith('SYN'):
        priority = 0
    elif problem.name.startswith('PUZ'):
        priority = 1
    else:
        priority = 2
    return (priority, problem.name, str(problem))


def _all_tptp_problem_paths() -> list[Path]:
    tptp_root_raw = os.getenv('TPTP')
    if tptp_root_raw is None:
        raise FileNotFoundError(
            'no input problems provided and no TPTP root is configured; '
            'pass --tptp ROOT or set the TPTP environment variable'
        )

    problems_root = Path(tptp_root_raw) / 'Problems'
    if not problems_root.is_dir():
        raise FileNotFoundError(f'could not find TPTP problems directory at {problems_root}')

    problems = sorted(
        (
            path.resolve()
            for path in problems_root.glob('*/*.p')
            if _is_supported_tptp_problem_path(path) and not problem_metadata(path).typed
        ),
        key=_problem_priority_key,
    )
    if not problems:
        raise FileNotFoundError(f'no supported .p files found under {problems_root}')

    return problems


def _is_supported_tptp_problem_path(path: Path) -> bool:
    if ''.join(path.suffixes) not in {'.p', '.ax'}:
        return False
    return '+' in path.name or '-' in path.name
