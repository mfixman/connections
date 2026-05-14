"""Compare proof strategies across a set of problems."""

import contextlib
import csv
import json
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

from cli import (
    configure_tptp_root,
    default_logic_for_strategy,
    default_sat_solver_for_strategy,
    expand_problem_inputs,
    parse_compare_args,
    resolve_sat_solver_for_strategy,
    split_problem_paths,
)
from connections.env import ConnectionEnv, Logic, SATSolver, Settings
from connections.utils.classical_problem_parsing import problem_metadata
from strategies import Strategy

_ANSI = re.compile(r'\033\[[0-9;]*m')
_PARTIAL_RESULT_PREFIX = 'RESULT_JSON\t'
_INLINE_PARTIAL_SEPARATOR = '  |  '

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

def _is_success_info(status_info: object) -> bool:
    if status_info == 'Theorem':
        return True
    if isinstance(status_info, str) and status_info.startswith('Theorem (SAT)'):
        return True
    return False

def _final_strategy_info(strategy: object) -> Optional[str]:
    final_info = getattr(strategy, 'final_info', None)
    if not callable(final_info):
        return None
    info = final_info()
    if info is None:
        return None
    return str(info)

def _combine_info(*parts: object) -> Optional[str]:
    present = [str(part) for part in parts if part is not None and str(part)]
    if not present:
        return None
    return '; '.join(present)

def _terminal_result_label(status_info: object) -> Optional[str]:
    if not isinstance(status_info, str) or not status_info:
        return None
    if status_info == 'Theorem':
        return status_info
    if status_info.startswith('Theorem (SAT)'):
        return 'Theorem (SAT)'
    if status_info.startswith('Non-Theorem'):
        return status_info
    return None

def _info_without_terminal_result(status_info: object) -> Optional[str]:
    if not isinstance(status_info, str) or not status_info:
        return None
    label = _terminal_result_label(status_info)
    if label is None:
        return status_info
    if status_info == label:
        return None
    if status_info.startswith(label):
        suffix = status_info[len(label):].lstrip()
        if suffix.startswith(';'):
            suffix = suffix[1:].strip()
        return suffix or None
    return None

def _problem_result_metadata(path: str | Path) -> dict:
    problem_path = Path(path)
    if not problem_path.exists():
        return {}

    metadata = problem_metadata(problem_path)
    return {
        'problem_spc_tags': sorted(metadata.spc_tags),
        'problem_typed': metadata.typed,
        'problem_declared_status': metadata.declared_status,
        'problem_declared_status_kind': metadata.declared_status_kind,
    }

def _contradiction_warning(result_label: Optional[str], path: str | Path) -> Optional[str]:
    if result_label is None:
        return None
    if result_label not in {'Theorem', 'Theorem (SAT)'}:
        return None

    metadata = problem_metadata(path)
    if metadata.declared_status_kind in {None, 'unsat'}:
        return None

    declared = metadata.declared_status or 'unknown'
    return f'WARNING: declared status={declared} but prover found {result_label}'

def _normalize_result(problem: Path, result: dict) -> dict:
    normalized = dict(result)
    label = normalized.get('result')
    if not isinstance(label, str) or not label:
        label = _terminal_result_label(normalized.get('info'))
        if label is not None:
            normalized['result'] = label

    stripped_info = _info_without_terminal_result(normalized.get('info'))
    if stripped_info is None:
        normalized.pop('info', None)
    else:
        normalized['info'] = stripped_info

    for key, value in _problem_result_metadata(problem).items():
        normalized.setdefault(key, value)
    return normalized

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
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=1.0)
        return {'status': 'timeout', 'solved': False}
    if conn.poll():
        return _recv_result(conn, proc)
    return {
        'status': 'error',
        'solved': False,
        'error': f'worker exited with code {proc.exitcode}',
    }

def _run_problem(
    conn: Connection,
    path: str,
    strategy_name: str,
    sat_solver: Optional[SATSolver],
    max_steps: int,
    timeout: Optional[float] = None,
) -> None:
    """Worker function: run one strategy on one problem and send back a result."""
    try:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                logging.disable(logging.CRITICAL)
                try:
                    logic = default_logic_for_strategy(strategy_name)
                    sat_solver = resolve_sat_solver_for_strategy(strategy_name, sat_solver)
                    env = ConnectionEnv(path, Settings(logic=logic, sat_solver=sat_solver))
                    strategy = Strategy.strategies()[strategy_name](env)
                    env.reset()

                    done, info = env.state.initial_status()
                    steps = 0
                    hit_max_steps = False
                    hit_timeout = False
                    start = time.perf_counter()

                    while not done:
                        if steps >= max_steps:
                            hit_max_steps = True
                            break
                        if timeout is not None and time.perf_counter() - start >= timeout:
                            hit_timeout = True
                            break
                        action = strategy.step()
                        try:
                            _, _, done, info = env.step(action)
                        except RecursionError:
                            steps = max_steps - 1
                        steps += 1

                    elapsed = time.perf_counter() - start

                    status_info = info.get('status') if isinstance(info, dict) else info
                    solved = _is_success_info(status_info)
                    status = 'solved'
                    if not solved:
                        if hit_timeout:
                            status = 'timeout'
                        elif hit_max_steps:
                            status = 'max_steps'
                        else:
                            status = 'unsolved'

                    result = {
                        'status': status,
                        'solved': solved,
                        'steps': steps,
                        'time': elapsed,
                    }
                    result_label = _terminal_result_label(status_info)
                    if result_label is not None:
                        result['result'] = result_label
                    final_strategy_info = _final_strategy_info(strategy)
                    final_info = _combine_info(
                        _info_without_terminal_result(status_info),
                        final_strategy_info,
                        _contradiction_warning(result_label, path),
                    )
                    if final_info is not None:
                        result['info'] = final_info
                    result.update(_problem_result_metadata(path))

                    conn.send(result)
                except Exception as exc:
                    result = {
                        'status': 'error',
                        'solved': False,
                        'error': f'{type(exc).__name__}: {exc}',
                        'traceback': traceback.format_exc(),
                    }
                    result.update(_problem_result_metadata(path))
                    conn.send(result)
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

def determine_worker_count(total_tasks: int, requested_workers: Optional[int] = None) -> int:
    if total_tasks <= 0:
        logging.info(f'using 0 worker processes for {total_tasks} tasks')
        return 0

    if requested_workers is not None:
        workers = requested_workers
        logging.info(f'--num-workers={requested_workers}')
    else:
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

def fmt_result(r: Optional[dict], show_steps: bool, show_time: bool, use_color: bool = True) -> str:
    if r is None:
        return red('timeout', use_color)
    status = r.get('status')
    if status == 'timeout':
        return red('timeout', use_color)
    if status == 'max_steps':
        return bold('max-steps', use_color)
    if status == 'unsolved':
        result_label = r.get('result')
        if isinstance(result_label, str) and result_label.startswith('Non-Theorem'):
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
    problem_width = max(
        max((len(_problem_display_name(problem)) for problem in problems), default=0),
        len('Problem'),
    )
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
        'non-theorem': {'status': 'unsolved', 'solved': False, 'result': 'Non-Theorem'},
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

def _partial_result_line(problem: Path, strategy_name: str, result: dict) -> str:
    return (
        f'{_PARTIAL_RESULT_PREFIX}'
        f'{_problem_display_name(problem)}\t{strategy_name}\t'
        f'{json.dumps(result, sort_keys=True, ensure_ascii=True)}'
    )

def _task_result_with_inline_metadata(problem: Path, strategy_name: str, result: dict, line: str) -> str:
    return f'{line}{_INLINE_PARTIAL_SEPARATOR}{_partial_result_line(problem, strategy_name, result)}'

def _parse_partial_result_line(
    line: str,
    problem_by_name: dict[str, Path],
    strategy_names: set[str],
) -> Optional[tuple[Path, str, dict]]:
    if not line.startswith(_PARTIAL_RESULT_PREFIX):
        return None

    try:
        problem_name, strategy_name, payload = line[len(_PARTIAL_RESULT_PREFIX):].split('\t', 2)
    except ValueError:
        return None

    problem = problem_by_name.get(problem_name)
    if problem is None or strategy_name not in strategy_names:
        return None

    try:
        result = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict):
        return None

    return problem, strategy_name, result

def _split_inline_partial_result(
    line: str,
    problem_by_name: dict[str, Path],
    strategy_names: set[str],
) -> tuple[str, Optional[tuple[Path, str, dict]]]:
    idx = line.find(_PARTIAL_RESULT_PREFIX)
    if idx < 0:
        return line, None

    visible = line[:idx].rstrip()
    if visible.endswith(_INLINE_PARTIAL_SEPARATOR.rstrip()):
        visible = visible[: -len(_INLINE_PARTIAL_SEPARATOR.rstrip())].rstrip()
    partial_result = _parse_partial_result_line(line[idx:], problem_by_name, strategy_names)
    return visible, partial_result

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
    row_results: dict[Path, dict[str, dict]] = {problem: {} for problem in problems}
    metadata_results: dict[Path, dict[str, dict]] = {problem: {} for problem in problems}
    if partial_path is None or not partial_path.exists():
        return PartialReportState(row_results, False, False, False, None)

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

        line, partial_result = _split_inline_partial_result(line, problem_by_name, strategy_names)
        if partial_result is not None:
            problem, strategy_name, result = partial_result
            metadata_results[problem][strategy_name] = _normalize_result(problem, result)
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
        row_results[problem][strategy_name] = _normalize_result(problem, result)

    results = {
        problem: {
            strategy_name: (
                metadata_results[problem][strategy_name]
                if strategy_name in metadata_results[problem]
                else _normalize_result(problem, row_results[problem][strategy_name])
            )
            for strategy_name in row_results[problem].keys() | metadata_results[problem].keys()
        }
        for problem in problems
    }

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
            self.partial_path.parent.mkdir(parents=True, exist_ok=True)
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

    def emit_task_result(self, problem: Path, strategy_name: str, result: dict, line: str) -> None:
        print(line, flush=True)

        partial = self._open_partial()
        if partial is not None:
            text = line if self.partial_color else strip_ansi(line)
            payload = _task_result_with_inline_metadata(problem, strategy_name, result, text) + '\n'
            if self._needs_leading_newline:
                payload = '\n' + payload
                self._needs_leading_newline = False
            partial.write(payload)
            partial.flush()

        self.last_line = strip_ansi(line)

    def close(self) -> None:
        if self._partial_file is not None:
            self._partial_file.close()
            self._partial_file = None

def run_tasks(
    problems: list[Path],
    strategies: list[str],
    sat_solver: Optional[SATSolver],
    max_steps: int,
    timeout: float,
    max_workers: int,
    show_steps: bool,
    show_time: bool,
    reporter: ReportWriter,
    initial_results: Optional[dict[Path, dict[str, dict]]] = None,
) -> tuple[dict[Path, dict[str, dict]], list[tuple[str, str, str]]]:
    results: dict[Path, dict[str, dict]] = {
        problem: (initial_results or {}).get(problem, {}).copy()
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
        parent_conn, child_conn = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=_run_problem,
            args=(child_conn, str(problem), strategy_name, sat_solver, max_steps, timeout),
            daemon=True,
        )
        proc.start()
        child_conn.close()
        active[parent_conn] = (problem, strategy_name, proc, time.monotonic() + timeout + 5.0)

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
            result = _normalize_result(problem, _cleanup_task_process(conn, proc))
            conn.close()
            results[problem][strategy_name] = result
            reporter.emit_task_result(
                problem,
                strategy_name,
                result,
                format_task_result(
                    problem, strategy_name, result,
                    show_steps, show_time, problem_width, strategy_width,
                ),
            )
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
            result = _normalize_result(problem, _recv_result(conn, proc))
            proc.join()
            conn.close()
            results[problem][strategy_name] = result
            reporter.emit_task_result(
                problem,
                strategy_name,
                result,
                format_task_result(
                    problem, strategy_name, result,
                    show_steps, show_time, problem_width, strategy_width,
                ),
            )
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
        'result',
        'solved',
        'steps',
        'time_ms',
        'problem_spc_tags',
        'problem_typed',
        'problem_declared_status',
        'problem_declared_status_kind',
        'info',
        'error',
    ]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for problem in problems:
            for strategy_name in strategies:
                result = _normalize_result(problem, results[problem].get(strategy_name, {}))
                writer.writerow({
                    'problem': _problem_csv_name(problem),
                    'strategy': strategy_name,
                    'status': result.get('status'),
                    'result': result.get('result'),
                    'solved': result.get('solved'),
                    'steps': result.get('steps'),
                    'time_ms': (
                        None if result.get('time') is None else round(result['time'] * 1000)
                    ),
                    'problem_spc_tags': ' '.join(result.get('problem_spc_tags', [])),
                    'problem_typed': result.get('problem_typed'),
                    'problem_declared_status': result.get('problem_declared_status'),
                    'problem_declared_status_kind': result.get('problem_declared_status_kind'),
                    'info': result.get('info'),
                    'error': 'error' if result.get('status') == 'error' else None,
                })

def parse_args(argv: list[str] | None = None):
    return parse_compare_args(argv)

def _completed_task_count(results: dict[Path, dict[str, dict]]) -> int:
    return sum(len(result_map) for result_map in results.values())

def print_run_status(
    partial_path: Optional[Path],
    csv_path: str,
    total_tasks: int,
    completed_tasks: int,
    max_workers: int,
    timeout: float,
) -> None:
    remaining_tasks = total_tasks - completed_tasks
    print(
        'Run: '
        f'partial={partial_path if partial_path is not None else "-"}  '
        f'csv={csv_path}  '
        f'tasks={total_tasks}  '
        f'completed={completed_tasks}  '
        f'remaining={remaining_tasks}  '
        f'workers={max_workers}  '
        f'timeout={timeout:g}s',
        file=sys.stderr,
        flush=True,
    )

def main() -> None:
    logging.basicConfig(level = logging.INFO, format = '[%(relativeCreated)d] %(message)s')
    args = parse_args()

    try:
        configure_tptp_root(args.tptp)
        problems = split_problem_paths(
            expand_problem_inputs(args.inputs),
            args.split,
            args.part,
        )
    except FileNotFoundError as exc:
        sys.exit(f'error: {exc}')

    strategies = args.strategies
    two = len(strategies) == 2
    total = len(problems)
    total_tasks = total * len(strategies)
    max_workers = determine_worker_count(total_tasks, args.num_workers)
    partial_path = None if args.partial_file is None else Path(args.partial_file)
    partial_state = load_partial_report(partial_path, problems, strategies)
    reporter = ReportWriter(partial_path, partial_color=args.color, last_line=partial_state.last_line)
    print_run_status(
        partial_path,
        args.csv,
        total_tasks,
        _completed_task_count(partial_state.results),
        max_workers,
        args.timeout,
    )

    _, _, header, sep = report_layout(problems, strategies)
    if not partial_state.has_header:
        reporter.emit(header)
        reporter.emit(sep)

    try:
        results, errors = run_tasks(
            problems, strategies, args.sat_solver, args.max_steps, args.timeout, max_workers,
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
