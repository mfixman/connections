"""Tie-breaking variants layered on top of ``SATSolveLiteral``."""

from __future__ import annotations

import random

from connections.calculi.classicalsat import ConnectionAction, Extension, Reduction, Start

from ..sat_solve import SATSolveLiteral


class _SATSolveLiteralTieMixin:
    """Common helpers for variants that keep SATSolveLiteral's primary score."""

    def _best_score_tied_actions(self, actions: list[ConnectionAction]) -> list[ConnectionAction]:
        if not actions:
            return []
        best_score = min(self.score(action) for action in actions)
        return [action for action in actions if self.score(action) == best_score]

    def _connection_tie_key(self, action: ConnectionAction) -> tuple[int, int]:
        match action:
            case Reduction():
                return (0, 0)
            case Extension(clause_copy = clause_copy):
                return (1, len(clause_copy) - 1)
            case Start():
                return (2, 0)
        return (3, 0)


class SATSolveLiteralRandomTie(_SATSolveLiteralTieMixin, SATSolveLiteral):
    """SATSolveLiteral, but choose randomly among actions tied on model value."""

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        tied = self._best_score_tied_actions(actions)
        if not tied:
            return None

        rng = getattr(self.env.state, 'random', random)
        return rng.choice(tied)


class SATSolveLiteralShortTie(_SATSolveLiteralTieMixin, SATSolveLiteral):
    """SATSolveLiteral with reduction/short-extension tie-breaking."""

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        tied = self._best_score_tied_actions(actions)
        return min(tied, key = self._connection_tie_key, default = None)


class SATSolveLiteralLookaheadTie(_SATSolveLiteralTieMixin, SATSolveLiteral):
    """SATSolveLiteral with CaDiCaL lookahead/EVSIDS tie-breaking.

    The SAT model value remains the primary score.  CaDiCaL data is only
    used inside the best model-value bucket, so this variant measures
    whether SAT branching data adds anything beyond SATCoP literal
    selection.
    """

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        tied = self._best_score_tied_actions(actions)
        if not tied:
            return None

        lookahead_lit = self._lookahead_literal()
        return max(tied, key = lambda action: self._lookahead_tie_key(action, lookahead_lit))

    def _lookahead_literal(self) -> int | None:
        solver = self.env.state.shadow.solver
        if solver is None or not hasattr(solver, 'lookahead'):
            return None
        try:
            lookahead_lit = solver.lookahead()
        except Exception:
            return None
        return lookahead_lit if isinstance(lookahead_lit, int) and lookahead_lit != 0 else None

    def _lookahead_tie_key(self, action: ConnectionAction, lookahead_lit: int | None) -> tuple[int, int, float, int, int]:
        sat_lit = self._selected_sat_literal(action)
        solver = self.env.state.shadow.solver
        evsids = 0.0
        if solver is not None and sat_lit is not None and hasattr(solver, 'score'):
            try:
                evsids = solver.score(abs(sat_lit))
            except Exception:
                evsids = 0.0

        exact_lookahead = int(sat_lit is not None and sat_lit == lookahead_lit)
        variable_lookahead = int(
            sat_lit is not None
            and lookahead_lit is not None
            and abs(sat_lit) == abs(lookahead_lit)
        )
        fallback_kind, fallback_size = self._connection_tie_key(action)
        return (exact_lookahead, variable_lookahead, evsids, -fallback_kind, -fallback_size)


class _SATSolveLiteralFrequencyTieMixin(_SATSolveLiteralTieMixin):
    """Tie-break unknown SAT literals by shadow-CNF atom occurrence count."""

    _prefer_high_frequency = True
    _frequency_signed = False
    _frequency_opposite_signed = False

    def _init_frequency_tie_breaker(self) -> None:
        self._frequency_clauses = set()
        self._frequency_atom_counts = {}
        self._frequency_signed_counts = {}
        self._frequency_decisions = 0
        self._frequency_candidates = 0
        self._frequency_mapped = 0
        self._frequency_unmapped = 0
        self._frequency_changed_default = 0
        self._frequency_agreed_default = 0
        self._frequency_zero_scores = 0
        self._frequency_total_score = 0

    def _sync_frequency_counts(self) -> None:
        current_clauses = set(self.env.state.shadow.ground_clauses)
        if not self._frequency_clauses.issubset(current_clauses):
            self._frequency_clauses = set()
            self._frequency_atom_counts = {}
            self._frequency_signed_counts = {}

        for clause in current_clauses - self._frequency_clauses:
            for sat_lit in clause:
                atom_id = abs(sat_lit)
                self._frequency_atom_counts[atom_id] = self._frequency_atom_counts.get(atom_id, 0) + 1
                self._frequency_signed_counts[sat_lit] = self._frequency_signed_counts.get(sat_lit, 0) + 1
            self._frequency_clauses.add(clause)

    def _atom_frequency(self, action: ConnectionAction) -> int | None:
        sat_lit = self._selected_sat_literal(action)
        if sat_lit is None:
            return None
        if self._frequency_signed:
            key = -sat_lit if self._frequency_opposite_signed else sat_lit
            return self._frequency_signed_counts.get(key, 0)
        return self._frequency_atom_counts.get(abs(sat_lit), 0)

    def _frequency_score(self, action: ConnectionAction) -> tuple[int, int, int]:
        frequency = self._atom_frequency(action)
        fallback_kind, fallback_size = self._connection_tie_key(action)
        if frequency is None:
            return (-1, -fallback_kind, -fallback_size)
        if not self._prefer_high_frequency:
            frequency = -frequency
        return (frequency, -fallback_kind, -fallback_size)

    def _choose_frequency_tie(self, tied: list[ConnectionAction]) -> ConnectionAction | None:
        if not tied:
            return None

        self._sync_frequency_counts()
        default_choice = tied[0]
        mapped = sum(1 for action in tied if self._selected_sat_literal(action) is not None)
        self._frequency_decisions += 1
        self._frequency_candidates += len(tied)
        self._frequency_mapped += mapped
        self._frequency_unmapped += len(tied) - mapped

        scored = [(action, self._frequency_score(action)) for action in tied]
        self._frequency_zero_scores += sum(
            1
            for action, score in scored
            if self._selected_sat_literal(action) is not None and score[0] == 0
        )
        self._frequency_total_score += sum(
            abs(score[0])
            for action, score in scored
            if self._selected_sat_literal(action) is not None
        )
        chosen = max(scored, key = lambda item: item[1])[0]
        if chosen is default_choice:
            self._frequency_agreed_default += 1
        else:
            self._frequency_changed_default += 1
        return chosen

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        if self._frequency_decisions:
            parts.append(
                'freq='
                f'mode:{"high" if self._prefer_high_frequency else "low"},'
                f'kind:{self._frequency_kind()},'
                f'decisions:{self._frequency_decisions},'
                f'candidates:{self._frequency_candidates},'
                f'mapped:{self._frequency_mapped},'
                f'unmapped:{self._frequency_unmapped},'
                f'zero_scores:{self._frequency_zero_scores},'
                f'total_score:{self._frequency_total_score},'
                f'agreed_default:{self._frequency_agreed_default},'
                f'changed_default:{self._frequency_changed_default}'
            )
        return '; '.join(parts) if parts else None

    def _frequency_kind(self) -> str:
        if not self._frequency_signed:
            return 'atom'
        if self._frequency_opposite_signed:
            return 'opposite_signed'
        return 'signed'


class SATSolveLiteralFrequencyTie(_SATSolveLiteralFrequencyTieMixin, SATSolveLiteral):
    """SATSolveLiteral with SAT atom-frequency tie-breaking.

    This keeps the SAT model truth value as the primary score.  It only
    intervenes when the best bucket is SAT-unknown, which diagnostics
    showed to be the common underdetermined case.  Within that bucket it
    prefers candidate literals whose SAT atom appears more often in the
    current shadow CNF, falling back to the standard connection tie key.
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_frequency_tie_breaker()

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        tied = self._best_score_tied_actions(actions)
        if not tied:
            return None
        if self.score(tied[0]) != 1 or len(tied) == 1:
            return tied[0]
        return self._choose_frequency_tie(tied)


class SATSolveLiteralLowFrequencyTie(SATSolveLiteralFrequencyTie):
    """SATSolveLiteral tie-breaker that prefers rarer SAT atoms."""

    _prefer_high_frequency = False


class SATSolveLiteralOppositeSignedFrequencyTie(SATSolveLiteralFrequencyTie):
    """SATSolveLiteral tie-breaker using frequency of the opposite SAT literal."""

    _frequency_signed = True
    _frequency_opposite_signed = True
