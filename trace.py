"""trace.py — print the action sequence used by a strategy on a problem.

Usage:
    python trace.py FILE STRATEGY [MAX_STEPS] [--color]

Example:
    python trace.py tests/cnf_problems/SET027+3.p FirstAction
    python trace.py tests/cnf_problems/SET027+3.p SmartCut
    python trace.py tests/cnf_problems/SET027+3.p SmartCut 10000 --color | less -R
"""

import os
import sys
from pathlib import Path
from connections.env import ConnectionEnv, Settings
from connections.calculi.classicalcut import (
    Start, Reduction, Extension, Backtrack, Reset,
)
from strategies import Strategy

_color = False

def _c(code: str) -> str:
    return code if _color else ''

def BOLD():   return _c('\033[1m')
def DIM():    return _c('\033[2m')
def GREEN():  return _c('\033[32m')
def CYAN():   return _c('\033[36m')
def YELLOW(): return _c('\033[33m')
def RED():    return _c('\033[31m')
def RESET():  return _c('\033[0m')

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

def fmt_action(action, step: int, highlight_special: bool) -> str:
    """Format one action for display."""
    prefix = f'{DIM()}{step:4d}.{RESET()} '

    def bold(s):
        return f'{BOLD()}{s}{RESET()}' if highlight_special else s

    # Layout: action name (9) + cut tag (6) + separator (1) + content
    # "Backtrack" is the longest name at 9 chars; [CUT] is 5 + 1 leading space.
    NAME = 9
    TAG  = 6   # ' [CUT]' or '      '

    def header(name: str, cut: bool | None = None) -> str:
        """Padded action name + optional [CUT] column."""
        n = f'{name:<{NAME}}'
        if cut is True:
            t = f' {BOLD()}{YELLOW()}[CUT]{RESET()}'
        elif cut is False:
            t = ' ' * TAG
        else:
            t = ' ' * TAG
        return n + t + ' '

    match action:
        case Start(clause_copy=clause):
            lits = ', '.join(str(l) for l in clause)
            return prefix + f'{GREEN()}{header("Start")}[{lits}]{RESET()}'

        case Reduction(principle_node=node, path_lit=path, cut=cut):
            line = f'{CYAN()}{header("Reduce", cut)}{RESET()}{node.literal} <- {path}'
            return prefix + (bold(line) if cut else line)

        case Extension(principle_node=node, clause_copy=clause, lit_idx=idx, cut=cut):
            lits = ', '.join(
                f'{BOLD()}{str(l)}{RESET()}' if i == idx else str(l)
                for i, l in enumerate(clause)
            )
            line = f'{CYAN()}{header("Extend", cut)}{RESET()}{node.literal} -> [{lits}]'
            return prefix + (bold(line) if cut else line)

        case Backtrack():
            line = f'{RED()}{header("Backtrack")}{RESET()}'
            return prefix + (bold(line) if highlight_special else line)

        case Reset(depth=depth, reason=reason):
            d = str(depth) if depth is not None else '?'
            suffix = f'  ({reason})' if reason else ''
            line = f'{YELLOW()}{header("Reset")}-> depth {d}{suffix}{RESET()}'
            return prefix + (bold(line) if highlight_special else line)

    return prefix + repr(action)

def run_trace(path: str, strategy_name: str, max_steps: int = 10000):
    env = ConnectionEnv(path, Settings(logic='ClassicalCut'))
    strategy = Strategy.strategies()[strategy_name](env)
    env.reset()

    sep = '-' * 60
    print(f'\n{sep}')
    print(f'  Strategy: {BOLD()}{strategy_name}{RESET()}   Problem: {path}')
    print(sep)

    highlight = (strategy_name != 'FirstAction')
    done, info = env.state.initial_status()
    step = 0

    while not done and step < max_steps:
        action = strategy.step()
        print(fmt_action(action, step + 1, highlight))
        _, _, done, info = env.step(action)
        step += 1

    print(sep)
    if done:
        label = info if isinstance(info, str) and info else 'Theorem'
        print(f'  {GREEN()}{BOLD()}{label}{RESET()}   ({step} steps)')
    else:
        print(f'  {RED()}Unknown (limit {max_steps}){RESET()}')
    print(f'{sep}\n')

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = {a for a in sys.argv[1:] if a.startswith('--')}

    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    if '--color' in flags:
        _color = True

    path = resolve_problem_path(args[0])
    strategy = args[1]
    max_steps = int(args[2]) if len(args) > 2 else 10000

    run_trace(path, strategy, max_steps)
