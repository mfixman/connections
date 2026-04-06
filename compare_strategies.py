#!/usr/bin/env python3
"""Compare proof strategies across a set of problems."""

import argparse
import contextlib
import csv
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
import logging
from collections import deque
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from pathlib import Path
from typing import Optional

from connections.env import ConnectionEnv, Logic, Settings
from strategies import Strategy

def green(s: str) -> str: return f'\033[32m{s}\033[0m'
def red(s: str) -> str:   return f'\033[31m{s}\033[0m'
def bold(s: str) -> str:  return f'\033[1;33m{s}\033[0m'
def dim(s: str) -> str:   return f'\033[2m{s}\033[0m'

_ANSI = re.compile(r'\033\[[0-9;]*m')

def _colour(code: str, s: str, enabled: bool = True) -> str:
    if not enabled:
        return s
    return f'\033[{code}m{s}\033[0m'

def green(s: str, enabled: bool = True) -> str: return _colour('32', s, enabled)
def red(s: str, enabled: bool = True) -> str:   return _colour('31', s, enabled)
def bold(s: str, enabled: bool = True) -> str:  return _colour('1;33', s, enabled)
def dim(s: str, enabled: bool = True) -> str:   return _colour('2', s, enabled)

def strip_ansi(s: str) -> str:
    return _ANSI.sub('', s)

def vlen(s: str) -> int:
    """Visible length of a string, ignoring ANSI escape codes."""
    return len(strip_ansi(s))

def rpad(s: str, width: int) -> str:
    """Right-pad a string to visible width, ignoring ANSI escape codes."""
    return s + ' ' * max(0, width - vlen(s))

def show_progress(s: str, width: int) -> None:
    """Show an in-place progress label without changing final table output."""
    if not sys.stdout.isatty():
        return
    sys.stdout.write('\033[2K\r' + rpad(s, width))
    sys.stdout.flush()

def clear_progress() -> None:
    """Clear the in-place progress label before printing a completed row."""
    if not sys.stdout.isatty():
        return
    sys.stdout.write('\033[2K\r')
    sys.stdout.flush()

def _status_label(result: Optional[dict]) -> str:
    if result is None:
        return 'timeout'
    return result.get('status', 'error')

def _recv_result(conn: Connection, proc: mp.Process) -> dict:
    try:
        return conn.recv()
    except EOFError:
        return {
            'status': 'error',
            'solved': False,
            'error': f'worker exited with code {proc.exitcode}',
        }

def _cleanup_task_process(conn: Connection, proc: mp.Process) -> dict:
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {'status': 'timeout', 'solved': False}
    if conn.poll():
        return _recv_result(conn, proc)
    return {
        'status': 'error',
        'solved': False,
        'error': f'worker exited with code {proc.exitcode}',
    }

def default_logic_for_strategy(strategy_name: str) -> str:
    if strategy_name in ('SATSolve', 'SATSolveLiteral'):
        return str(Logic.ClassicalSAT)
    return str(Logic.ClassicalCut)

def resolve_logic_for_strategy(strategy_name: str, logic_override: Optional[str]) -> str:
    if logic_override is not None:
        return logic_override
    return default_logic_for_strategy(strategy_name)

def _run_problem(
    conn: Connection,
    path: str,
    strategy_name: str,
    logic: str,
    max_steps: int,
) -> None:
    """Worker function: run one strategy on one problem and send back a result."""
    try:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                logging.disable(logging.CRITICAL)
                try:
                    env = ConnectionEnv(path, Settings(logic=logic))
                    strategy = Strategy.strategies()[strategy_name](env)
                    env.reset()

                    done = False
                    info = None
                    steps = 0
                    hit_max_steps = False
                    start = time.perf_counter()

                    while not done:
                        if steps >= max_steps:
                            hit_max_steps = True
                            break
                        action = strategy.step()
                        try:
                            _, _, done, info = env.step(action)
                        except RecursionError:
                            steps = max_steps - 1
                        steps += 1

                    elapsed = time.perf_counter() - start

                    status_info = info.get('status') if isinstance(info, dict) else info
                    solved = status_info in {'Theorem', 'Theorem (SAT)'}
                    status = 'solved'
                    if not solved:
                        status = 'max_steps' if hit_max_steps else 'unsolved'

                    result = {
                        'status': status,
                        'solved': solved,
                        'steps': steps,
                        'time': elapsed,
                    }
                    if status_info is not None:
                        result['info'] = str(status_info)

                    conn.send(result)
                except Exception as exc:
                    conn.send({
                        'status': 'error',
                        'solved': False,
                        'error': f'{type(exc).__name__}: {exc}',
                        'traceback': traceback.format_exc(),
                    })
    finally:
        conn.close()

def _parse_slurm_cpus_per_task(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None

def _process_cpu_count() -> Optional[int]:
    process_cpu_count = getattr(os, 'process_cpu_count', None)
    if process_cpu_count is None:
        return None
    return process_cpu_count()

def determine_worker_count(total_tasks: int) -> int:
    slurm_raw = os.getenv('SLURM_CPUS_PER_TASK')
    slurm_cpus = _parse_slurm_cpus_per_task(slurm_raw)
    process_cpus = _process_cpu_count()

    logging.info(f'SLURM_CPUS_PER_TASK={slurm_raw!r}')
    logging.info(f'os.process_cpu_count()={process_cpus!r}')

    candidates = [count for count in (slurm_cpus, process_cpus) if count is not None]
    if not candidates:
        workers = 1
    else:
        workers = min(candidates)

    workers = max(1, min(workers, total_tasks))
    worker_label = 'worker process' if workers == 1 else 'worker processes'
    task_label = 'task' if total_tasks == 1 else 'tasks'
    logging.info(f'using {workers} {worker_label} for {total_tasks} {task_label}')
    return workers

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
        (path.resolve() for path in problems_root.glob('*/*.p')),
        key=_problem_priority_key,
    )
    if not problems:
        raise FileNotFoundError(f'no .p files found under {problems_root}')

    return problems

def expand_problem_inputs(inputs: list[str]) -> list[Path]:
    """Expand directories and problem names into a deduplicated list of files."""
    if not inputs:
        return _all_tptp_problem_paths()

    problems: list[Path] = []
    for raw in inputs:
        input_path = Path(raw)
        if input_path.exists() and input_path.is_dir():
            dir_problems = sorted(input_path.glob('*.p'))
            if not dir_problems:
                raise FileNotFoundError(f'no .p files found in {raw!r}')
            problems.extend(dir_problems)
        else:
            problems.append(Path(resolve_problem_path(raw)))

    return _unique_paths_in_order([problem.resolve() for problem in problems])

def fmt_result(r: Optional[dict], show_steps: bool, show_time: bool, use_color: bool = True) -> str:
    if r is None:
        return red('timeout', use_color)
    status = r.get('status')
    if status == 'timeout':
        return red('timeout', use_color)
    if status == 'max_steps':
        return bold('max-steps', use_color)
    if status == 'unsolved':
        info = r.get('info', 'unsolved')
        if str(info).startswith('Non-Theorem'):
            return dim('non-theorem', use_color)
        return dim('unsolved', use_color)
    if status == 'error':
        return red('error', use_color)
    if not r['solved']:
        return dim('unsolved', use_color)
    parts = []
    if show_steps:
        parts.append(f'{r["steps"]:,}')
    if show_time:
        parts.append(f'{r["time"] * 1000:.0f}ms')
    return green(' / '.join(parts) if parts else 'solved', use_color)

def fmt_delta(
    r1: Optional[dict], r2: Optional[dict],
    show_steps: bool, show_time: bool,
    name1: str, name2: str,
    use_color: bool = True,
) -> str:
    s1 = r1 is not None and r1['solved']
    s2 = r2 is not None and r2['solved']

    if not s1 and not s2:
        label1 = _status_label(r1)
        label2 = _status_label(r2)
        if label1 == label2:
            return dim(f'both {label1}', use_color)
        return dim(f'{label1} / {label2}', use_color)
    if s1 and not s2:
        return red(f'{name1} only', use_color)
    if s2 and not s1:
        return bold(f'{name2} only', use_color)

    # Both solved
    parts = []
    if show_steps:
        pct = (r2['steps'] - r1['steps']) * 100 / max(r1['steps'], 1)
        if pct < 0:
            parts.append(bold(f'{pct:+.0f}% steps', use_color))
        elif pct > 0:
            parts.append(red(f'{pct:+.0f}% steps', use_color))
        else:
            parts.append('=steps')
    if show_time:
        pct = (r2['time'] - r1['time']) * 100 / max(r1['time'], 1e-9)
        if pct < -5:
            parts.append(bold(f'{pct:+.0f}% time', use_color))
        elif pct > 5:
            parts.append(red(f'{pct:+.0f}% time', use_color))
        else:
            parts.append('≈time')

    return ', '.join(parts) if parts else dim('tie', use_color)

def _problem_display_name(problem: Path) -> str:
    return problem.name

def _problem_csv_name(problem: Path) -> str:
    return problem.name

def report_layout(problems: list[Path], strategies: list[str]) -> tuple[int, int, str, str]:
    problem_width = max(max(len(_problem_display_name(problem)) for problem in problems), len('Problem'))
    strategy_width = max(max(len(strategy) for strategy in strategies), len('Strategy'))
    header = (
        f'{rpad("Problem", problem_width)}  '
        f'{rpad("Strategy", strategy_width)}  '
        'Result'
    )
    sep = '-' * (problem_width + strategy_width + len('Result') + 4)
    return problem_width, strategy_width, header, sep

def format_task_result(
    problem: Path,
    strategy_name: str,
    result: dict,
    show_steps: bool,
    show_time: bool,
    problem_width: int,
    strategy_width: int,
) -> str:
    return (
        f'{rpad(_problem_display_name(problem), problem_width)}  '
        f'{rpad(strategy_name, strategy_width)}  '
        f'{fmt_result(result, show_steps, show_time)}'
    )

def print_task_error(problem: Path, strategy_name: str, result: dict) -> None:
    error = result.get('error', 'unknown error')
    print(
        f'error: {_problem_display_name(problem)} [{strategy_name}]: {error}',
        file=sys.stderr,
        flush=True,
    )

def parse_result_text(text: str) -> Optional[dict]:
    stripped = text.strip()
    if not stripped:
        return None

    simple_statuses = {
        'timeout': {'status': 'timeout', 'solved': False},
        'max-steps': {'status': 'max_steps', 'solved': False},
        'unsolved': {'status': 'unsolved', 'solved': False},
        'non-theorem': {'status': 'unsolved', 'solved': False, 'info': 'Non-Theorem'},
        'error': {'status': 'error', 'solved': False},
        'solved': {'status': 'solved', 'solved': True},
    }
    if stripped in simple_statuses:
        return dict(simple_statuses[stripped])

    result = {'status': 'solved', 'solved': True}
    for token in re.split(r'\s*/\s*', stripped):
        if re.fullmatch(r'[0-9][0-9,]*', token):
            result['steps'] = int(token.replace(',', ''))
            continue
        if re.fullmatch(r'[0-9]+(?:\.[0-9]+)?ms', token):
            result['time'] = float(token[:-2]) / 1000
            continue
        return None
    return result

@dataclass
class PartialReportState:
    results: dict[Path, dict[str, dict]]
    has_header: bool
    has_summary: bool
    has_csv_line: bool
    last_line: Optional[str]

def load_partial_report(
    partial_path: Optional[Path],
    problems: list[Path],
    strategies: list[str],
) -> PartialReportState:
    results: dict[Path, dict[str, dict]] = {problem: {} for problem in problems}
    if partial_path is None or not partial_path.exists():
        return PartialReportState(results, False, False, False, None)

    problem_by_name = {_problem_display_name(problem): problem for problem in problems}
    strategy_names = set(strategies)
    has_header = False
    saw_summary = False
    saw_csv_line = False
    last_line = None

    for raw_line in partial_path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = strip_ansi(raw_line).rstrip()
        if not line:
            continue

        last_line = line
        parts = re.split(r' {2,}', line, maxsplit=2)
        if parts == ['Problem', 'Strategy', 'Result']:
            has_header = True
            continue
        if line.startswith('Solved: '):
            saw_summary = True
            continue
        if line.startswith('CSV: '):
            saw_csv_line = True
            continue
        if len(parts) != 3:
            continue

        problem_name, strategy_name, result_text = parts
        problem = problem_by_name.get(problem_name)
        if problem is None or strategy_name not in strategy_names:
            continue

        result = parse_result_text(result_text)
        if result is None:
            continue
        results[problem][strategy_name] = result

    completed_tasks = sum(len(result_map) for result_map in results.values())
    total_tasks = len(problems) * len(strategies)
    if completed_tasks != total_tasks:
        saw_summary = False
        saw_csv_line = False

    return PartialReportState(results, has_header, saw_summary, saw_csv_line, last_line)

class ReportWriter:
    def __init__(
        self,
        partial_path: Optional[Path],
        partial_color: bool,
        last_line: Optional[str] = None,
    ) -> None:
        self.partial_path = partial_path
        self.partial_color = partial_color
        self.last_line = last_line
        self._partial_file = None
        self._needs_leading_newline = False

        if partial_path is not None and partial_path.exists() and partial_path.stat().st_size > 0:
            with open(partial_path, 'rb') as existing:
                existing.seek(-1, os.SEEK_END)
                self._needs_leading_newline = existing.read(1) != b'\n'

    def _open_partial(self):
        if self.partial_path is None:
            return None
        if self._partial_file is None:
            self._partial_file = open(self.partial_path, 'a', encoding='utf-8')
        return self._partial_file

    def emit(self, line: str) -> None:
        print(line, flush=True)

        partial = self._open_partial()
        if partial is not None:
            if self._needs_leading_newline:
                partial.write('\n')
                self._needs_leading_newline = False
            partial.write(line if self.partial_color else strip_ansi(line))
            partial.write('\n')
            partial.flush()

        self.last_line = strip_ansi(line)

    def close(self) -> None:
        if self._partial_file is not None:
            self._partial_file.close()
            self._partial_file = None

def run_tasks(
    problems: list[Path],
    strategies: list[str],
    logic_override: Optional[str],
    max_steps: int,
    timeout: float,
    max_workers: int,
    show_steps: bool,
    show_time: bool,
    reporter: ReportWriter,
    initial_results: Optional[dict[Path, dict[str, dict]]] = None,
) -> tuple[dict[Path, dict[str, dict]], list[tuple[str, str, str]]]:
    results: dict[Path, dict[str, dict]] = {
        problem: dict((initial_results or {}).get(problem, {}))
        for problem in problems
    }
    task_queue = deque(
        (problem, strategy)
        for problem in problems
        for strategy in strategies
        if strategy not in results[problem]
    )
    errors: list[tuple[str, str, str]] = []

    problem_width, strategy_width, _, _ = report_layout(problems, strategies)

    active: dict[Connection, tuple[Path, str, mp.Process, float]] = {}

    def start_task(problem: Path, strategy_name: str) -> None:
        logic = resolve_logic_for_strategy(strategy_name, logic_override)
        parent_conn, child_conn = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=_run_problem,
            args=(child_conn, str(problem), strategy_name, logic, max_steps),
            daemon=True,
        )
        proc.start()
        child_conn.close()
        active[parent_conn] = (problem, strategy_name, proc, time.monotonic() + timeout)

    while task_queue and len(active) < max_workers:
        start_task(*task_queue.popleft())

    while active:
        now = time.monotonic()
        expired = [
            conn for conn, (_, _, _, deadline) in active.items()
            if deadline <= now
        ]
        for conn in expired:
            problem, strategy_name, proc, _ = active.pop(conn)
            result = _cleanup_task_process(conn, proc)
            conn.close()
            results[problem][strategy_name] = result
            reporter.emit(format_task_result(
                problem, strategy_name, result,
                show_steps, show_time, problem_width, strategy_width,
            ))
            if result.get('status') == 'error':
                error = result.get('error', 'unknown error')
                errors.append((_problem_display_name(problem), strategy_name, error))
                print_task_error(problem, strategy_name, result)

        while task_queue and len(active) < max_workers:
            start_task(*task_queue.popleft())

        if not active:
            break

        next_deadline = min(deadline for _, _, _, deadline in active.values())
        timeout_remaining = max(0.0, next_deadline - time.monotonic())
        ready = wait(list(active), timeout=timeout_remaining)
        for conn in ready:
            problem, strategy_name, proc, _ = active.pop(conn)
            result = _recv_result(conn, proc)
            proc.join()
            conn.close()
            results[problem][strategy_name] = result
            reporter.emit(format_task_result(
                problem, strategy_name, result,
                show_steps, show_time, problem_width, strategy_width,
            ))
            if result.get('status') == 'error':
                error = result.get('error', 'unknown error')
                errors.append((_problem_display_name(problem), strategy_name, error))
                print_task_error(problem, strategy_name, result)

        while task_queue and len(active) < max_workers:
            start_task(*task_queue.popleft())

    return results, errors

def write_results_csv(
    csv_path: str,
    problems: list[Path],
    strategies: list[str],
    results: dict[Path, dict[str, dict]],
) -> None:
    fieldnames = [
        'problem',
        'strategy',
        'status',
        'solved',
        'steps',
        'time_ms',
        'info',
        'error',
    ]
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for problem in problems:
            for strategy_name in strategies:
                result = results[problem].get(strategy_name, {})
                writer.writerow({
                    'problem': _problem_csv_name(problem),
                    'strategy': strategy_name,
                    'status': result.get('status'),
                    'solved': result.get('solved'),
                    'steps': result.get('steps'),
                    'time_ms': (
                        None if result.get('time') is None else round(result['time'] * 1000)
                    ),
                    'info': result.get('info'),
                    'error': result.get('error'),
                })

def parse_args() -> argparse.Namespace:
    available_strategies = list(Strategy.strategies().keys())
    default_strategies = [available_strategies[0], available_strategies[-1]]
    available_logics = [str(l) for l in Logic]

    parser = argparse.ArgumentParser(
        prog='compare_strategies.py',
        description='Compare proof strategies across a set of problems.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            f'available strategies: {", ".join(available_strategies)}\n'
            f'available logics:     {", ".join(available_logics)}\n'
            '\n'
            'examples:\n'
            '  %(prog)s\n'
            '  %(prog)s tests/cnf_problems\n'
            '  %(prog)s comparisons/classical --timeout 60 --max-steps 100000\n'
            '  %(prog)s tests --strategies FirstAction SmartCut\n'
            '  %(prog)s tests --all-strategies\n'
            '  %(prog)s tests --no-time\n'
            '  %(prog)s tests --no-steps\n'
        ),
    )
    parser.add_argument(
        'inputs', nargs='*', metavar='DIR_OR_FILE',
        help='directories and/or problem files/codenames (default: all $TPTP/Problems/*/*.p)',
    )

    strategy_group = parser.add_mutually_exclusive_group()
    strategy_group.add_argument(
        '--strategies', nargs='+',
        choices=available_strategies, metavar='STRATEGY',
        help=f'strategies to compare (default: {" ".join(default_strategies)})',
    )
    strategy_group.add_argument(
        '--all-strategies', action='store_true',
        help='compare all available strategies',
    )
    parser.add_argument(
        '--logic', choices=available_logics, metavar='LOGIC',
        help='logic to use for all strategies (default: strategy-dependent)',
    )
    parser.add_argument(
        '--max-steps', type=int, default=50000, metavar='N',
        help='max steps per run (default: 50000)',
    )
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
        '--csv', default='results.csv', metavar='PATH',
        help='write final results as CSV to PATH (default: results.csv)',
    )
    parser.add_argument(
        '--partial-file', metavar='PATH',
        help='append report output to PATH and resume from any completed rows already present',
    )
    parser.add_argument(
        '--color', '--colour', dest='color', action='store_true',
        help='preserve ANSI colour codes in --partial-file output (default: off)',
    )
    args = parser.parse_args()

    if args.all_strategies:
        args.strategies = available_strategies
    elif args.strategies is None:
        args.strategies = default_strategies

    return args

def main() -> None:
    logging.basicConfig(level = logging.INFO, format = '[%(relativeCreated)d] %(message)s')
    args = parse_args()

    try:
        problems = expand_problem_inputs(args.inputs)
    except FileNotFoundError as exc:
        sys.exit(f'error: {exc}')

    strategies = args.strategies
    two = len(strategies) == 2
    total = len(problems)
    total_tasks = total * len(strategies)
    max_workers = determine_worker_count(total_tasks)
    partial_path = None if args.partial_file is None else Path(args.partial_file)
    partial_state = load_partial_report(partial_path, problems, strategies)
    reporter = ReportWriter(partial_path, partial_color=args.color, last_line=partial_state.last_line)

    _, _, header, sep = report_layout(problems, strategies)
    if not partial_state.has_header:
        reporter.emit(header)
        reporter.emit(sep)

    try:
        results, errors = run_tasks(
            problems, strategies, args.logic, args.max_steps, args.timeout, max_workers,
            args.steps, args.time, reporter, partial_state.results,
        )

        solved_counts = {s: 0 for s in strategies}
        step_wins = {s: 0 for s in strategies}
        ties = 0

        for problem in problems:
            for strategy_name in strategies:
                result = results[problem].get(strategy_name)
                if result and result.get('solved'):
                    solved_counts[strategy_name] += 1

            if not two:
                continue

            r1, r2 = results[problem].get(strategies[0]), results[problem].get(strategies[1])
            s1 = r1 is not None and r1.get('solved')
            s2 = r2 is not None and r2.get('solved')
            if s1 and s2:
                steps1 = r1.get('steps')
                steps2 = r2.get('steps')
                if steps1 is None or steps2 is None:
                    ties += 1
                elif steps1 < steps2:
                    step_wins[strategies[0]] += 1
                elif steps2 < steps1:
                    step_wins[strategies[1]] += 1
                else:
                    ties += 1
            elif s1:
                step_wins[strategies[0]] += 1
            elif s2:
                step_wins[strategies[1]] += 1

        solved_str = '  '.join(f'{s}={solved_counts[s]}/{total}' for s in strategies)
        if two:
            wins_str = (
                f'  |  step wins: {strategies[0]}={step_wins[strategies[0]]}'
                f'  {strategies[1]}={step_wins[strategies[1]]}'
                f'  ties={ties}'
            )
        else:
            wins_str = ''
        summary_line = f'Solved: {solved_str}{wins_str}'

        if not partial_state.has_summary:
            if reporter.last_line != sep:
                reporter.emit(sep)
            reporter.emit(summary_line)
        else:
            print(summary_line, flush=True)

        if errors:
            print(f'Errors: {len(errors)}', file=sys.stderr, flush=True)

        write_results_csv(args.csv, problems, strategies, results)
        csv_line = f'CSV: {args.csv}'
        if not partial_state.has_csv_line:
            reporter.emit(csv_line)
        else:
            print(csv_line, flush=True)
    finally:
        reporter.close()

if __name__ == '__main__':
    main()
