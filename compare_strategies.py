#!/usr/bin/env python3
"""Compare proof strategies across a set of problems."""

import argparse
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path
from typing import Optional

from connections.env import ConnectionEnv, Logic, Settings
from strategies import Strategy


# ── ANSI colour helpers ───────────────────────────────────────────────────────

def green(s: str) -> str: return f'\033[32m{s}\033[0m'
def red(s: str) -> str:   return f'\033[31m{s}\033[0m'
def bold(s: str) -> str:  return f'\033[1;33m{s}\033[0m'
def dim(s: str) -> str:   return f'\033[2m{s}\033[0m'

_ANSI = re.compile(r'\033\[[0-9;]*m')

def vlen(s: str) -> int:
    """Visible length of a string, ignoring ANSI escape codes."""
    return len(_ANSI.sub('', s))

def rpad(s: str, width: int) -> str:
    """Right-pad a string to visible width, ignoring ANSI escape codes."""
    return s + ' ' * max(0, width - vlen(s))


# ── Worker ────────────────────────────────────────────────────────────────────

def _run_problem(queue: mp.Queue, path: str, strategy_name: str, logic: str, max_steps: int) -> None:
    """Worker function: run one strategy on one problem and push result to queue."""
    try:
        env = ConnectionEnv(path, Settings(logic=logic))
        strategy = Strategy.strategies()[strategy_name](env)
        env.reset()

        done = False
        info = None
        steps = 0
        start = time.perf_counter()

        while not done:
            if steps >= max_steps:
                info = {'Solution': 'Unknown'}
                break
            action = strategy.step()
            try:
                _, _, done, info = env.step(action)
            except RecursionError:
                steps = max_steps - 1
            steps += 1

        elapsed = time.perf_counter() - start

        solved = (info == 'Theorem')

        queue.put({
            'solved': solved,
            'steps': steps,
            'time': elapsed,
        })
    except Exception as exc:
        queue.put({'error': str(exc)})


def run_strategies(
    path: str,
    strategies: list[str],
    logic: str,
    max_steps: int,
    timeout: float,
) -> dict[str, Optional[dict]]:
    """Run all strategies on a problem in parallel, with a shared timeout deadline."""
    queues: dict[str, mp.Queue] = {s: mp.Queue() for s in strategies}
    procs: dict[str, mp.Process] = {
        s: mp.Process(
            target=_run_problem,
            args=(queues[s], path, s, logic, max_steps),
            daemon=True,
        )
        for s in strategies
    }

    for p in procs.values():
        p.start()

    deadline = time.monotonic() + timeout
    results: dict[str, Optional[dict]] = {}
    for s in strategies:
        remaining = max(0.0, deadline - time.monotonic())
        procs[s].join(remaining)
        if procs[s].is_alive():
            procs[s].terminate()
            procs[s].join()
            results[s] = None
        else:
            r = queues[s].get() if not queues[s].empty() else None
            results[s] = None if (r is None or 'error' in r) else r

    return results


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_result(r: Optional[dict], show_steps: bool, show_time: bool) -> str:
    if r is None or not r['solved']:
        return red('timeout')
    parts = []
    if show_steps:
        parts.append(f'{r["steps"]:,}')
    if show_time:
        parts.append(f'{r["time"] * 1000:.0f}ms')
    return green(' / '.join(parts) if parts else 'solved')


def fmt_delta(
    r1: Optional[dict], r2: Optional[dict],
    show_steps: bool, show_time: bool,
    name1: str, name2: str,
) -> str:
    s1 = r1 is not None and r1['solved']
    s2 = r2 is not None and r2['solved']

    if not s1 and not s2:
        return dim('both timeout')
    if s1 and not s2:
        return red(f'{name1} only')
    if s2 and not s1:
        return bold(f'{name2} only')

    # Both solved
    parts = []
    if show_steps:
        pct = (r2['steps'] - r1['steps']) * 100 / max(r1['steps'], 1)
        if pct < 0:
            parts.append(bold(f'{pct:+.0f}% steps'))
        elif pct > 0:
            parts.append(red(f'{pct:+.0f}% steps'))
        else:
            parts.append('=steps')
    if show_time:
        pct = (r2['time'] - r1['time']) * 100 / max(r1['time'], 1e-9)
        if pct < -5:
            parts.append(bold(f'{pct:+.0f}% time'))
        elif pct > 5:
            parts.append(red(f'{pct:+.0f}% time'))
        else:
            parts.append('≈time')

    return ', '.join(parts) if parts else dim('tie')


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    available_strategies = list(Strategy.strategies().keys())
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
            '  %(prog)s tests --no-time\n'
            '  %(prog)s tests --no-steps\n'
        ),
    )
    parser.add_argument(
        'dir', nargs='?', default='tests/cnf_problems', metavar='DIR',
        help='directory of .p problem files (default: tests/cnf_problems)',
    )
    parser.add_argument(
        '--strategies', nargs='+',
        default=available_strategies[:2],
        choices=available_strategies, metavar='STRATEGY',
        help=f'strategies to compare (default: {" ".join(available_strategies[:2])})',
    )
    parser.add_argument(
        '--logic', default='ClassicalCut', choices=available_logics, metavar='LOGIC',
        help='logic to use (default: ClassicalCut)',
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
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    problems = sorted(Path(args.dir).glob('*.p'))
    if not problems:
        sys.exit(f'error: no .p files found in {args.dir!r}')

    strategies = args.strategies
    two = len(strategies) == 2

    name_w = max(max(len(p.name) for p in problems), len('Problem'))
    col_w = 24

    # Header
    sep = '-' * (name_w + (col_w + 2) * len(strategies) + (18 if two else 0))
    header = rpad('Problem', name_w)
    for s in strategies:
        header += '  ' + rpad(s, col_w)
    if two:
        header += '  ' + f'Δ ({strategies[1]} vs {strategies[0]})'
    print(header)
    print(sep)

    solved_counts = {s: 0 for s in strategies}
    step_wins = {s: 0 for s in strategies}
    ties = 0
    total = 0

    for problem in problems:
        total += 1
        results = run_strategies(
            str(problem), strategies, args.logic, args.max_steps, args.timeout,
        )

        for s in strategies:
            if results[s] and results[s]['solved']:
                solved_counts[s] += 1

        row = rpad(problem.name, name_w)
        for s in strategies:
            row += '  ' + rpad(fmt_result(results[s], args.steps, args.time), col_w)

        if two:
            r1, r2 = results[strategies[0]], results[strategies[1]]
            row += '  ' + fmt_delta(r1, r2, args.steps, args.time, strategies[0], strategies[1])

            s1 = r1 is not None and r1['solved']
            s2 = r2 is not None and r2['solved']
            if s1 and s2:
                if r1['steps'] < r2['steps']:
                    step_wins[strategies[0]] += 1
                elif r2['steps'] < r1['steps']:
                    step_wins[strategies[1]] += 1
                else:
                    ties += 1
            elif s1:
                step_wins[strategies[0]] += 1
            elif s2:
                step_wins[strategies[1]] += 1

        print(row)

    print(sep)
    solved_str = '  '.join(f'{s}={solved_counts[s]}/{total}' for s in strategies)
    if two:
        wins_str = (
            f'  |  step wins: {strategies[0]}={step_wins[strategies[0]]}'
            f'  {strategies[1]}={step_wins[strategies[1]]}'
            f'  ties={ties}'
        )
    else:
        wins_str = ''
    print(f'Solved: {solved_str}{wins_str}')


if __name__ == '__main__':
    main()
