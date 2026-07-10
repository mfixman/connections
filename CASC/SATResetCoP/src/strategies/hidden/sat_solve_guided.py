"""SAT-guided extension-choice variants."""

from __future__ import annotations

from typing import NamedTuple

from connections.calculi.classicalsat import Backtrack, Extension, Reduction, Reset, Start
from connections.utils.primitives import Literal

from .. import sat_solve as sat_solve_module
from ..sat_solve import SATSolve


class _ExtensionSignals(NamedTuple):
    probe_conflicts: int
    forced_true: int
    evsids: float
    subgoal_count: int


class _GuidedExtensionChoice:
    """Shared SAT-shadow extension-choice policy for guided variants."""

    _use_probe_conflicts = True
    _use_forced_true = True
    _use_evsids = True

    def step(self):
        actions = self.env.legal_actions()

        self._sync_shadow_solver()
        self._record_nonzero_atom_scores()

        for action in actions:
            if isinstance(action, (Reduction, Extension)):
                action.cut = True

        for action in actions:
            if isinstance(action, Start):
                return action
            if isinstance(action, Reduction):
                return action
            if isinstance(action, Extension):
                extensions = [a for a in actions if isinstance(a, Extension)]
                if len(extensions) <= 1:
                    return action
                return max(extensions, key = self._extension_score)

        if (
            self.env.state.goal.parent is None
            and any(isinstance(action, Reset) for action in actions)
            and self.env.state.shadow.new_ground_clauses
        ):
            return Reset(id = 'rs', depth = self.env.state.max_depth, reason = 'ground shadow still growing')

        if action := next((a for a in actions if isinstance(a, Backtrack)), None):
            return action

        if action := next((a for a in actions if isinstance(a, Reset)), None):
            return action

        raise LookupError('Expected at least one action to be Start, Reduction, Extension, Backtrack, or Reset.')

    def _extension_score(self, action: Extension) -> tuple[int, int, float, int]:
        signals = self._extension_signals(action)
        return self.extension_signal_score(signals)

    def extension_signal_score(self, signals: _ExtensionSignals) -> tuple[int, int, float, int]:
        return (
            signals.probe_conflicts,
            signals.forced_true,
            signals.evsids,
            -signals.subgoal_count,
        )

    def _extension_signals(self, action: Extension) -> _ExtensionSignals:
        state = self.env.state
        subgoals = self._remaining_subgoals(action)

        if not (self._use_probe_conflicts or self._use_forced_true or self._use_evsids):
            return _ExtensionSignals(0, 0, 0.0, len(subgoals))

        state.substitution.update(action.sub_updates)
        try:
            sat_literals = self._sat_literals(subgoals)
        finally:
            state.substitution.backtrack()

        solver = state.shadow.solver
        probe_conflicts = 0
        forced_true = 0
        evsids = 0.0

        for sat_lit in sat_literals:
            if self._use_forced_true and solver.fixed(sat_lit) > 0:
                forced_true += 1

            if self._use_evsids:
                evsids += solver.score(abs(sat_lit))

            if not self._use_probe_conflicts:
                continue

            solver.assume(-sat_lit)
            try:
                if solver.solve() == sat_solve_module.pydical.UNSATISFIABLE:
                    probe_conflicts += 1
            finally:
                solver.reset_assumptions()

        return _ExtensionSignals(probe_conflicts, forced_true, evsids, len(subgoals))

    def _remaining_subgoals(self, action: Extension) -> list[Literal]:
        return [
            literal
            for idx, literal in enumerate(action.clause_copy)
            if idx != action.lit_idx
        ]

    def _sat_literals(self, literals: list[Literal]) -> list[int]:
        state = self.env.state
        result = []

        for literal in literals:
            grounded_literal = state.substitution(literal)
            atom_id = state.shadow.atom_map.get(state._atom_key(grounded_literal))
            if atom_id is None:
                continue
            result.append(-atom_id if grounded_literal.neg else atom_id)

        return result


class SATSolveGuided(_GuidedExtensionChoice, SATSolve):
    """SATCoP control with SAT-shadow scoring for extension choice.

    This keeps ``SATSolve``'s control policy, including paper mode,
    restricted backtracking, root resets, and backtracking order.  The
    only difference is at extension choice points: when ``SATSolve``
    would pick an extension and more than one extension is available,
    choose the extension with the best score from the current SAT
    shadow.

    Sources and scope:
        This is not the Rawson/Reger literal-selection rule; that is
        already represented by ``SATSolveLiteral``.  This strategy is an
        experiment in using SAT-solver-internal properties to guide the
        first-order connection prover.

        The SAT-side ingredients are standard CDCL signals rather than
        new prover machinery:
        - EVSIDS-style variable activity, introduced by the Chaff SAT
          solver (Moskewicz et al., 2001), exposed here by CaDiCaL as
          ``score(var)``.
        - Forced assignments / failed assumptions, a standard
          assumption-based incremental SAT interface; PySAT documents
          the same assumption-call style for its solver wrappers.
        - Variable involvement in propagation and conflict analysis as
          a useful importance signal for SAT search, e.g. Liu et al.
          (2024), "NeuroSelect: Learning to Select Clauses in SAT
          Solvers", §3.1-3.2.

        I did not find an existing paper in ``papers/`` that applies
        these SAT-internal signals to guide extension choice in a
        connection prover.  The originality here is intentionally small:
        map candidate extension subgoals to existing SAT atoms and ask
        CaDiCaL which atoms currently look important.  The score is a
        lexicographic tuple, not a weighted sum: first prefer assumption
        probes that refute the current shadow CNF, then forced literals,
        then larger EVSIDS activity, then fewer remaining subgoals.
    """


class SATSolveGuidedProbeConflicts(SATSolveGuided):
    """Guided extension choice using only failed-assumption conflicts."""

    _use_forced_true = False
    _use_evsids = False

    def extension_signal_score(self, signals: _ExtensionSignals) -> tuple[int]:
        return (signals.probe_conflicts,)


class SATSolveGuidedForcedTrue(SATSolveGuided):
    """Guided extension choice using only forced-true SAT literals."""

    _use_probe_conflicts = False
    _use_evsids = False

    def extension_signal_score(self, signals: _ExtensionSignals) -> tuple[int]:
        return (signals.forced_true,)


class SATSolveGuidedEVSIDS(SATSolveGuided):
    """Guided extension choice using only summed CaDiCaL EVSIDS scores."""

    _use_probe_conflicts = False
    _use_forced_true = False

    def extension_signal_score(self, signals: _ExtensionSignals) -> tuple[float]:
        return (signals.evsids,)


class SATSolveGuidedShort(SATSolveGuided):
    """Guided extension choice using only fewer remaining subgoals."""

    _use_probe_conflicts = False
    _use_forced_true = False
    _use_evsids = False

    def extension_signal_score(self, signals: _ExtensionSignals) -> tuple[int]:
        return (-signals.subgoal_count,)


class SATSolveGuidedFast(SATSolveGuided):
    """Guided extension choice using only cheap non-probing signals."""

    _use_probe_conflicts = False

    def extension_signal_score(self, signals: _ExtensionSignals) -> tuple[int, float, int]:
        return (signals.forced_true, signals.evsids, -signals.subgoal_count)
