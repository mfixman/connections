"""Propagation and assumption-probe SAT variants."""

from __future__ import annotations

from typing import NamedTuple

from connections.calculi.classicalsat import ConnectionAction, Extension, Reduction, SATSolver, Start

from .. import sat_solve as sat_solve_module
from ..sat_solve import SATSolveLiteral
from .sat_solve_ties import _SATSolveLiteralTieMixin


class _PropagationSignals(NamedTuple):
    conflict: int
    implied_count: int
    lookahead_match: int
    evsids: float
    fallback_kind: int
    fallback_size: int
    mapped: int
    backend: str


class _SATSolvePropagationMixin(_SATSolveLiteralTieMixin):
    """SAT propagation probe helpers for forward-action selection."""

    def _init_propagation_probe(self) -> None:
        self._pysat_solver = None
        self._pysat_clauses = set()
        self._pysat_unavailable = False
        self._propagation_backend = 'none'
        self._propagation_decisions = 0
        self._propagation_candidates = 0
        self._propagation_mapped = 0
        self._propagation_conflicts = 0
        self._propagation_implied_nonzero = 0
        self._propagation_agreed_literal = 0
        self._propagation_changed_literal = 0

    def _choose_by_propagation(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        if not actions:
            return None

        lookahead_lit = self._lookahead_literal()
        scored_actions = [
            (action, self._propagation_signals(action, lookahead_lit))
            for action in actions
        ]
        self._record_propagation_signals(signal for _, signal in scored_actions)
        chosen = max(scored_actions, key = lambda item: self._propagation_score(item[1]))[0]
        literal_chosen = min(actions, key = self.score, default = None)
        if chosen is literal_chosen:
            self._propagation_agreed_literal += 1
        else:
            self._propagation_changed_literal += 1
        return chosen

    def _propagation_score(self, signals: _PropagationSignals) -> tuple[int, int, int, float, int, int]:
        return (
            signals.conflict,
            signals.implied_count,
            signals.lookahead_match,
            signals.evsids,
            -signals.fallback_kind,
            -signals.fallback_size,
        )

    def _propagation_signals(self, action: ConnectionAction, lookahead_lit: int | None) -> _PropagationSignals:
        sat_lit = self._selected_sat_literal(action)
        fallback_kind, fallback_size = self._connection_tie_key(action)
        if sat_lit is None:
            return _PropagationSignals(0, 0, 0, 0.0, fallback_kind, fallback_size, 0, 'none')

        backend, conflict, implied_count = self._propagate_assumption(sat_lit)
        solver = self.env.state.shadow.solver
        evsids = 0.0
        if solver is not None and hasattr(solver, 'score'):
            try:
                evsids = solver.score(abs(sat_lit))
            except Exception:
                evsids = 0.0

        lookahead_match = int(
            lookahead_lit is not None
            and (sat_lit == lookahead_lit or abs(sat_lit) == abs(lookahead_lit))
        )
        return _PropagationSignals(
            conflict,
            implied_count,
            lookahead_match,
            evsids,
            fallback_kind,
            fallback_size,
            1,
            backend,
        )

    def _propagate_assumption(self, sat_lit: int) -> tuple[str, int, int]:
        pysat_result = self._pysat_propagation(sat_lit)
        if pysat_result is not None:
            return pysat_result
        return self._cadical_assumption_probe(sat_lit)

    def _pysat_propagation(self, sat_lit: int) -> tuple[str, int, int] | None:
        solver = self._pysat_probe_solver()
        if solver is None:
            return None

        try:
            status, propagated = solver.propagate(assumptions = [sat_lit])
        except Exception:
            return None

        propagated_vars = {
            abs(literal)
            for literal in propagated
            if abs(literal) != abs(sat_lit)
        }
        return ('pysat', int(status is False), len(propagated_vars))

    def _pysat_probe_solver(self):
        if self._pysat_unavailable:
            return None

        try:
            from pysat.solvers import Solver as PySATSolver
        except ModuleNotFoundError:
            self._pysat_unavailable = True
            return None

        if self._pysat_solver is None:
            try:
                self._pysat_solver = PySATSolver(name = 'm22')
            except TypeError:
                self._pysat_solver = PySATSolver()

        current_clauses = set(self.env.state.shadow.ground_clauses)
        for clause in sorted(current_clauses - self._pysat_clauses):
            try:
                self._pysat_solver.add_clause(list(clause))
            except Exception:
                self._pysat_unavailable = True
                self._pysat_solver = None
                return None
            self._pysat_clauses.add(clause)

        return self._pysat_solver

    def _cadical_assumption_probe(self, sat_lit: int) -> tuple[str, int, int]:
        solver = self.env.state.shadow.solver
        if (
            solver is None
            or not hasattr(solver, 'assume')
            or not hasattr(solver, 'solve')
            or not hasattr(solver, 'reset_assumptions')
        ):
            return ('none', 0, 0)

        solver.assume(sat_lit)
        try:
            conflict = int(
                sat_solve_module.pydical is not None
                and solver.solve() == sat_solve_module.pydical.UNSATISFIABLE
            )
        except Exception:
            conflict = 0
        finally:
            solver.reset_assumptions()
        return ('cadical-fallback', conflict, 0)

    def _lookahead_literal(self) -> int | None:
        solver = self.env.state.shadow.solver
        if solver is None or not hasattr(solver, 'lookahead'):
            return None
        try:
            lookahead_lit = solver.lookahead()
        except Exception:
            return None
        return lookahead_lit if isinstance(lookahead_lit, int) and lookahead_lit != 0 else None

    def _record_propagation_signals(self, signals) -> None:
        signals = list(signals)
        self._propagation_decisions += 1
        self._propagation_candidates += len(signals)
        self._propagation_mapped += sum(signal.mapped for signal in signals)
        self._propagation_conflicts += sum(signal.conflict for signal in signals)
        self._propagation_implied_nonzero += sum(
            1
            for signal in signals
            if signal.implied_count > 0
        )
        for signal in signals:
            if signal.backend != 'none':
                self._propagation_backend = signal.backend
                break

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        if self._propagation_decisions:
            parts.append(
                'prop='
                f'backend:{self._propagation_backend},'
                f'decisions:{self._propagation_decisions},'
                f'candidates:{self._propagation_candidates},'
                f'mapped:{self._propagation_mapped},'
                f'conflicts:{self._propagation_conflicts},'
                f'implied_nonzero:{self._propagation_implied_nonzero},'
                f'agreed_literal:{self._propagation_agreed_literal},'
                f'changed_literal:{self._propagation_changed_literal}'
            )
        return '; '.join(parts) if parts else None


class SATSolvePropagation(_SATSolvePropagationMixin, SATSolveLiteral):
    """Rank forward SATCoP actions by SAT propagation data.

    Start actions are still tried before the tableau has a branch.  Once
    reductions and extensions are available, candidate head literals are
    mapped into the current shadow CNF and probed as SAT assumptions.
    PySAT is used when installed because its wrappers expose
    ``propagate``; otherwise the strategy falls back to CaDiCaL live
    assumptions and records only failed-assumption conflicts.
    """

    sat_solver_backend = SATSolver.CaDiCaL

    def __init__(self, env):
        super().__init__(env)
        self._init_propagation_probe()

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        starts = [action for action in actions if isinstance(action, Start)]
        if starts:
            return starts[0]

        forward = [action for action in actions if isinstance(action, (Reduction, Extension))]
        if forward:
            return self._choose_by_propagation(forward)

        return min(actions, key = self.score, default = None)


class SATSolveLiteralPropagation(_SATSolvePropagationMixin, SATSolveLiteral):
    """SATSolveLiteral with propagation data as the tie-breaker."""

    sat_solver_backend = SATSolver.CaDiCaL

    def __init__(self, env):
        super().__init__(env)
        self._init_propagation_probe()

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        tied = self._best_score_tied_actions(actions)
        return self._choose_by_propagation(tied)
