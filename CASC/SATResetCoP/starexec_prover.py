#!/usr/bin/env python3
"""SZS/CASC front-end for SATResetCoP on StarExec.

StarExec invokes ``bin/starexec_run_<config> <benchmark>`` and enforces a
wall-clock / CPU limit externally with ``runsolver``.  This wrapper runs the
SATResetCoP connection prover on a single TPTP problem and prints its verdict
in the SZS ontology that CASC scores, e.g.::

    % SZS status Theorem for PUZ001+1

It differs from ``pycop.py`` in three ways that matter for the competition:

* output is the SZS status line (and, with ``--proof``, an output block)
  instead of a Python ``dict`` repr;
* it self-imposes a deadline and emits ``% SZS status Timeout`` gracefully
  before ``runsolver`` kills the process; and
* it defaults to the portable PySAT SAT backend (``python-sat`` wheels) rather
  than the native ``pydical`` binding, so the StarExec bundle is relocatable.

The SAT backend swap only changes which satisfying assignments are produced,
never the UNSAT/SAT verdict, so proofs remain sound.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
from pathlib import Path

# Make the bundled prover importable regardless of the current directory.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / 'src'
sys.path.insert(0, str(_SRC))

# Connection search is deeply recursive; give it plenty of Python stack.
sys.setrecursionlimit(1_000_000)

from connections.env import ConnectionEnv, Logic, Settings  # noqa: E402
from connections.calculi.classicalsat import SATSolver  # noqa: E402
from cli import configure_tptp_root, default_logic_for_strategy, _raw_tptp_suffix  # noqa: E402
from strategies import Strategy  # noqa: E402

import tempfile  # noqa: E402

_COUNTED_STEP_ACTIONS = {'Start', 'Reduction', 'Extension'}

# TPTP roles whose presence means "prove the conjecture" -> SZS Theorem.
# Their absence (pure clause set / negated goal) means we refute -> SZS
# Unsatisfiable.  Header ``% Status`` comments are deliberately ignored so we
# never read the expected answer out of the problem file.
_CONJECTURE_RE = re.compile(
    r'\b(?:fof|tff|thf|tcf)\s*\(\s*[^,]+,\s*conjecture\b', re.IGNORECASE
)


def _selectable_strategies() -> dict:
    try:
        return Strategy.strategies(include_hidden=True)
    except TypeError:
        return Strategy.strategies()


def _counts_as_step(action) -> bool:
    return action.__class__.__name__ in _COUNTED_STEP_ACTIONS


def _problem_name(path: str) -> str:
    return Path(path).stem


def _has_conjecture(path: str) -> bool:
    try:
        text = Path(path).read_text(errors='replace')
    except OSError:
        return False
    return bool(_CONJECTURE_RE.search(text))


def _normalized_problem_path(path: str, tmpdir: str) -> str:
    """Return a path the connection parser accepts.

    StarExec's containerised backend mounts the benchmark under the fixed name
    ``/starexec/input/benchmark`` — no ``.p`` suffix and none of the ``+``/``-``
    the parser uses to tell FOF from CNF. When the given filename is not usable,
    sniff the content and materialise a correctly-named copy so the intended
    path runs (FOF -> swipl clausification, CNF -> direct parse).
    """
    p = Path(path)
    usable = ''.join(p.suffixes) in {'.p', '.ax'} and ('+' in p.name or '-' in p.name)
    if usable:
        return path
    try:
        text = p.read_text(errors='replace')
    except OSError:
        return path
    suffix = _raw_tptp_suffix(text)  # '+1.p' (FOF) | '-1.p' (CNF) | None
    if suffix is None:
        return path  # unrecognised content: let the parser reject it -> GaveUp
    target = Path(tmpdir) / f'problem{suffix}'
    target.write_text(text, encoding='utf-8')
    return str(target)


class _SzsPrinter:
    """Prints exactly one SZS status line for the problem, at most once."""

    def __init__(self, name: str, proved_status: str) -> None:
        self.name = name
        self.proved_status = proved_status
        self._emitted = False

    def emit(self, status: str) -> None:
        if self._emitted:
            return
        self._emitted = True
        print(f'% SZS status {status} for {self.name}', flush=True)

    def emit_proof(self, witness: str) -> None:
        block = 'CNFRefutation'
        print(f'% SZS output start {block} for {self.name}', flush=True)
        for line in witness.splitlines():
            print(f'% {line}', flush=True)
        print(f'% SZS output end {block} for {self.name}', flush=True)


def _map_status(info: str | None, printer: _SzsPrinter) -> str:
    """Translate the prover's internal ``info`` string to an SZS status."""
    if not info:
        return 'GaveUp'
    head = info.strip()
    if head.startswith('Theorem'):
        # Proved: refutation of axioms + negated conjecture.
        return printer.proved_status
    if head.startswith('Non-Theorem'):
        # The connection search is a refutation procedure; a "Non-Theorem"
        # signal (e.g. no positive start clause) is not a sound completeness
        # claim in general, so we report GaveUp rather than risk a wrong
        # (Counter)Satisfiable answer.
        return 'GaveUp'
    return 'GaveUp'


def _install_deadline(printer: _SzsPrinter, seconds: float) -> None:
    """Emit a graceful Timeout status just before runsolver kills us."""
    if seconds <= 0:
        return

    def _handler(signum, frame):  # noqa: ANN001
        printer.emit('Timeout')
        os._exit(0)

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def _resolve_time_limit(args) -> float:
    if args.cpu_limit is not None:
        limit = args.cpu_limit
    else:
        env = os.environ
        raw = (
            env.get('STAREXEC_WALLCLOCK_LIMIT')
            or env.get('STAREXEC_CPU_LIMIT')
            or env.get('STAREXEC_MAX_TIMEOUT')
        )
        if raw is not None:
            try:
                limit = float(raw)
            except ValueError:
                limit = float(args.default_limit)
        else:
            limit = float(args.default_limit)
    # Leave a margin so our own Timeout status wins the race with runsolver.
    margin = max(1.0, min(5.0, limit * 0.05))
    return max(0.0, limit - margin)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='SZS/CASC front-end for the SATResetCoP connection prover.'
    )
    parser.add_argument('problem', help='TPTP problem file (FOF or CNF)')
    parser.add_argument(
        '--strategy', default='SATResetCoP',
        help='prover strategy (default: SATResetCoP)',
    )
    parser.add_argument(
        '--sat-solver', default=SATSolver.PySAT, type=SATSolver,
        choices=list(SATSolver),
        help='SAT backend (default: PySAT, the portable python-sat backend)',
    )
    parser.add_argument(
        '--cpu-limit', type=float, default=None, metavar='SEC',
        help='wall-clock limit in seconds (overrides STAREXEC_* env vars)',
    )
    parser.add_argument(
        '--default-limit', type=float, default=300.0, metavar='SEC',
        help='fallback time limit when none is provided (default: 300)',
    )
    parser.add_argument(
        '--tptp', default=None, metavar='ROOT',
        help='TPTP root for include() resolution (else the TPTP env var)',
    )
    parser.add_argument(
        '--max-steps', type=int, default=10_000_000_000,
        help='hard cap on inference steps (time is the real limit)',
    )
    parser.add_argument(
        '--proof', action='store_true',
        help='emit an SZS output block with the refutation witness',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    name = _problem_name(args.problem)
    proved_status = 'Theorem' if _has_conjecture(args.problem) else 'Unsatisfiable'
    printer = _SzsPrinter(name, proved_status)

    _install_deadline(printer, _resolve_time_limit(args))

    try:
        configure_tptp_root(args.tptp)
    except FileNotFoundError as exc:
        print(f'% SZS status Error for {name}', flush=True)
        print(f'% {exc}', file=sys.stderr, flush=True)
        return 0

    try:
        logic = default_logic_for_strategy(args.strategy)
        parse_path = _normalized_problem_path(args.problem, tempfile.mkdtemp(prefix='satresetcop-'))
        env = ConnectionEnv(
            parse_path,
            Settings(
                logic=logic,
                sat_solver=args.sat_solver,
                debug=args.proof,
            ),
        )
        strategy = _selectable_strategies()[args.strategy](env)
    except Exception as exc:  # parse / translation / unsupported problem
        printer.emit('GaveUp')
        print(f'% {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 0

    env.reset()
    done, info = env.state.initial_status()

    steps = 0
    try:
        while not done:
            if steps >= args.max_steps:
                info = None
                break
            action = strategy.step()
            _observation, _reward, done, info = env.step(action)
            if _counts_as_step(action):
                steps += 1
    except RecursionError:
        printer.emit('GaveUp')
        print('% RecursionError: search stack exhausted', file=sys.stderr, flush=True)
        return 0
    except Exception as exc:
        printer.emit('GaveUp')
        print(f'% {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        return 0

    status = _map_status(info if isinstance(info, str) else None, printer)
    printer.emit(status)

    if args.proof and status in {'Theorem', 'Unsatisfiable'} and isinstance(info, str):
        printer.emit_proof(info)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
