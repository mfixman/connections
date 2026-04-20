"""Shared problem-input resolution helpers for CLI entrypoints."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Sequence, TypeVar


T = TypeVar('T')


def add_problem_split_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared problem-list partitioning arguments to a CLI parser."""
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
    """Validate problem-list partitioning arguments."""
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
    """Resolve and validate shared split arguments from argparse."""
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
    """Return the selected partition of an ordered problem list."""
    validate_problem_split(split, part)
    return [path for index, path in enumerate(paths) if index % split == part]


def resolve_problem_path(path: str) -> str:
    """Resolve a problem from the local path or from the `TPTP` tree."""
    problem_path = Path(path)

    for candidate in _candidate_paths(problem_path):
        if candidate.exists():
            return str(candidate)

    tptp_root = os.getenv('TPTP')
    if tptp_root is None:
        raise FileNotFoundError(f'could not find problem {path!r} and TPTP is not set')

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
    """Expand directories and problem names into a deduplicated list of files."""
    if not inputs:
        return _all_tptp_problem_paths()

    problems: list[Path] = []
    for raw in inputs:
        input_path = Path(raw)
        if input_path.exists() and input_path.is_dir():
            dir_problems = sorted(
                path for path in input_path.glob('*.p')
                if _is_supported_tptp_problem_path(path)
            )
            if not dir_problems:
                raise FileNotFoundError(f'no supported .p files found in {raw!r}')
            problems.extend(dir_problems)
        else:
            problems.append(Path(resolve_problem_path(raw)))

    return _unique_paths_in_order([problem.resolve() for problem in problems])


@contextmanager
def resolved_problem_paths(raw_input: str):
    """Resolve one CLI input into one or more concrete problem paths.

    Accepts local files, directories, TPTP codenames/paths, and raw
    quoted TPTP problem text.
    """
    raw_suffix = _raw_tptp_suffix(raw_input)
    if raw_suffix is not None and not Path(raw_input).exists():
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=raw_suffix,
            delete=False,
            encoding='utf-8',
        ) as tmp:
            tmp.write(raw_input)
            temp_path = Path(tmp.name)
        try:
            yield [str(temp_path)]
        finally:
            temp_path.unlink(missing_ok=True)
        return

    yield [str(path) for path in expand_problem_inputs([raw_input])]


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
    """Return the exact name and the optional `.p` variant to try."""
    candidates = [path]
    if path.suffix:
        return candidates

    candidates.append(path.with_suffix('.p'))
    return candidates


def _find_tptp_matches(path: Path, tptp_root: Path) -> list[Path]:
    """Search the TPTP tree for an exact filename or bare codename."""
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
    """Deduplicate and sort resolved paths for stable lookup errors."""
    return sorted(set(paths))


def _unique_paths_in_order(paths: list[Path]) -> list[Path]:
    """Deduplicate while preserving first-seen order."""
    return list(dict.fromkeys(paths))


def _problem_priority_key(problem: Path) -> tuple[int, str, str]:
    """Prefer SYN, then PUZ, then the rest for bulk TPTP runs."""
    if problem.name.startswith('SYN'):
        priority = 0
    elif problem.name.startswith('PUZ'):
        priority = 1
    else:
        priority = 2
    return (priority, problem.name, str(problem))


def _all_tptp_problem_paths() -> list[Path]:
    """Return all TPTP `.p` problems, prioritising SYN and PUZ first."""
    tptp_root_raw = os.getenv('TPTP')
    if tptp_root_raw is None:
        raise FileNotFoundError('no input problems provided and TPTP is not set')

    problems_root = Path(tptp_root_raw) / 'Problems'
    if not problems_root.is_dir():
        raise FileNotFoundError(f'could not find TPTP problems directory at {problems_root}')

    problems = sorted(
        (
            path.resolve()
            for path in problems_root.glob('*/*.p')
            if _is_supported_tptp_problem_path(path)
        ),
        key=_problem_priority_key,
    )
    if not problems:
        raise FileNotFoundError(f'no supported .p files found under {problems_root}')

    return problems


def _is_supported_tptp_problem_path(path: Path) -> bool:
    """Return whether the filename shape is one our parser supports."""
    if ''.join(path.suffixes) not in {'.p', '.ax'}:
        return False
    return '+' in path.name or '-' in path.name
