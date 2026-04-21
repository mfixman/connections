"""Strategy for the SAT-aware classical connection calculus."""

from __future__ import annotations

import random
from typing import NamedTuple

from connections.calculi.classicalsat import (
    Backtrack,
    ConnectionAction,
    Extension,
    Reduction,
    Reset,
    SATSolver,
    Start,
    pydical,
)
from connections.utils.primitives import Literal
from .base import Strategy

class _ExtensionSignals(NamedTuple):
    probe_conflicts: int
    forced_true: int
    evsids: float
    subgoal_count: int

class _PropagationSignals(NamedTuple):
    conflict: int
    implied_count: int
    lookahead_match: int
    evsids: float
    fallback_kind: int
    fallback_size: int
    mapped: int
    backend: str

class _SATScoreTracker:
    """Track the first solver atom score that becomes non-zero."""

    def _init_sat_score_tracker(self) -> None:
        self._sat_nonzero_atom_score = None

    def _sync_shadow_solver(self) -> None:
        state = self.env.state
        state._flush_new_ground_clauses()
        state._solve_ground_clauses()

    def _record_nonzero_atom_scores(self) -> None:
        if self._sat_nonzero_atom_score is not None:
            return

        state = self.env.state
        solver = state.shadow.solver
        if solver is None:
            return

        for atom_id in state.shadow.atom_map.values():
            try:
                atom_score = solver.score(atom_id)
            except Exception:
                continue
            if atom_score not in (0, None):
                self._sat_nonzero_atom_score = atom_score
                return

    def final_info(self) -> str | None:
        if self._sat_nonzero_atom_score is None:
            self._sync_shadow_solver()
            self._record_nonzero_atom_scores()
        if self._sat_nonzero_atom_score is not None:
            return f'score={self._sat_nonzero_atom_score}'
        return None

class SATSolve(_SATScoreTracker, Strategy):
    """Paper-faithful SATCoP control for ``ClassicalSAT``.

    This mirrors Rawson and Reger (2021), Algorithm 2:
    randomised forward selection from the calculus, restricted
    backtracking on successful reduction/extension steps, depth control
    based on whether new grounded clauses were found, and model-based
    lemmata handled by the state machine.
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_sat_score_tracker()
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('SATSolve requires the ClassicalSAT calculus')
        self.env.state.enable_paper_mode(True)

    def step(self):
        actions = self.env.legal_actions()

        self._sync_shadow_solver()
        self._record_nonzero_atom_scores()

        for action in actions:
            if isinstance(action, (Reduction, Extension)):
                action.cut = True

            if isinstance(action, (Start, Reduction, Extension)):
                return action

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
        return self._extension_signal_score(signals)

    def _extension_signal_score(self, signals: _ExtensionSignals) -> tuple[int, int, float, int]:
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
                if solver.solve() == pydical.UNSATISFIABLE:
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

    def _extension_signal_score(self, signals: _ExtensionSignals) -> tuple[int]:
        return (signals.probe_conflicts,)

class SATSolveGuidedForcedTrue(SATSolveGuided):
    """Guided extension choice using only forced-true SAT literals."""

    _use_probe_conflicts = False
    _use_evsids = False

    def _extension_signal_score(self, signals: _ExtensionSignals) -> tuple[int]:
        return (signals.forced_true,)

class SATSolveGuidedEVSIDS(SATSolveGuided):
    """Guided extension choice using only summed CaDiCaL EVSIDS scores."""

    _use_probe_conflicts = False
    _use_forced_true = False

    def _extension_signal_score(self, signals: _ExtensionSignals) -> tuple[float]:
        return (signals.evsids,)

class SATSolveGuidedShort(SATSolveGuided):
    """Guided extension choice using only fewer remaining subgoals."""

    _use_probe_conflicts = False
    _use_forced_true = False
    _use_evsids = False

    def _extension_signal_score(self, signals: _ExtensionSignals) -> tuple[int]:
        return (-signals.subgoal_count,)

class SATSolveGuidedFast(SATSolveGuided):
    """Guided extension choice using only cheap non-probing signals."""

    _use_probe_conflicts = False

    def _extension_signal_score(self, signals: _ExtensionSignals) -> tuple[int, float, int]:
        return (signals.forced_true, signals.evsids, -signals.subgoal_count)

class SATSolveLiteral(_SATScoreTracker, Strategy):
    """Strict SATCoP with §9.1 literal-selection refinement.

    Implements Rawson and Reger (2021), Algorithm 2 with the
    "literal selection" extension from Sect. 9.1: when several forward
    actions are available, prefer those whose head literal is assigned
    ``True`` in the current SAT model (i.e. the first goal literal the
    model would close), falling back to the randomised order from the
    calculus when the model has nothing to say.

    Like ``SATSolve``, this strategy:
    - Enables ``paper_mode`` on the calculus, which gives:
      * initial grounding of start clauses (§4),
      * model-based lemmata that skip goals already assigned false
        in the SAT model (§6),
      * refusal to accept a tableau-level proof — acceptance comes
        only when the SAT solver reports UNSAT (§4.1).
    - Applies restricted backtracking (``cut=True``) on successful
      reduction/extension steps, matching Algorithm 2.
    - Controls iterative deepening via the calculus: stays at the
      same depth while new ground clauses are being discovered, only
      incrementing when they stop (§5).

    Differences from ``SATSolve``:
    - Forward actions are scored by SAT truth value of their head
      literal, so reductions/extensions the SAT model already believes
      are closest to closing fire first (§9.1 "literal selection").
    - When no SAT information is available for the candidate literals
      (e.g. first iteration, or all literals still non-ground), the
      behaviour degenerates to ``SATSolve``'s "first shuffled action".

    Known differences from the paper that are shared with SATSolve:
    - The SAT backend is CaDiCaL rather than the custom CDCL
      from §4.3.
    - Start clauses are taken from the positive+negative pool rather
      than strictly "clauses derived from the conjecture".
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_sat_score_tracker()
        self._init_literal_choice_diagnostics()
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('SATSolveLiteral requires the ClassicalSAT calculus')
        self.env.state.enable_paper_mode(True)

    def score(self, action: ConnectionAction):
        match action:
            case Start():
                return 2

            case Reduction(path_lit = path_lit):
                value = self.env.state.sat_literal_value(path_lit)
                if value is True:
                    return 0
                elif value is None:
                    return 1
                else:
                    return 3

            case Extension(clause_copy = clause_copy, lit_idx = lit_idx):
                value = None
                if 0 <= lit_idx < len(clause_copy):
                    value = self.env.state.sat_literal_value(clause_copy[lit_idx])
                if value is True:
                    return 0
                elif value is None:
                    return 1
                else:
                    return 3

        return 4

    def _init_literal_choice_diagnostics(self) -> None:
        self._literal_choice_points = 0
        self._literal_forward_actions = 0
        self._literal_unique_best = 0
        self._literal_ties = 0
        self._literal_tied_actions = 0
        self._literal_best_true = 0
        self._literal_best_unknown = 0
        self._literal_best_start = 0
        self._literal_best_false = 0
        self._literal_best_other = 0
        self._literal_tied_mappable = 0
        self._literal_tied_mapped = 0
        self._literal_chosen_start = 0
        self._literal_chosen_reduction = 0
        self._literal_chosen_extension = 0

    def _selected_literal(self, action: ConnectionAction) -> Literal | None:
        match action:
            case Reduction(path_lit = path_lit):
                return path_lit
            case Extension(clause_copy = clause_copy, lit_idx = lit_idx):
                if 0 <= lit_idx < len(clause_copy):
                    return clause_copy[lit_idx]
        return None

    def _selected_sat_literal(self, action: ConnectionAction) -> int | None:
        literal = self._selected_literal(action)
        if literal is None:
            return None

        state = self.env.state
        if not hasattr(state, '_atom_key'):
            return None

        substitution = getattr(state, 'substitution', None)
        grounded_literal = substitution(literal) if callable(substitution) else literal
        atom_id = state.shadow.atom_map.get(state._atom_key(grounded_literal))
        if atom_id is None:
            return None
        return -atom_id if grounded_literal.neg else atom_id

    def _record_literal_choice_diagnostics(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> None:
        forward = [
            action
            for action in actions
            if isinstance(action, (Start, Reduction, Extension))
        ]
        if not forward:
            return

        scored = [(action, self.score(action)) for action in forward]
        best_score = min(score for _, score in scored)
        tied = [action for action, score in scored if score == best_score]

        self._literal_choice_points += 1
        self._literal_forward_actions += len(forward)
        if len(tied) == 1:
            self._literal_unique_best += 1
        else:
            self._literal_ties += 1
            self._literal_tied_actions += len(tied)

        match best_score:
            case 0:
                self._literal_best_true += 1
            case 1:
                self._literal_best_unknown += 1
            case 2:
                self._literal_best_start += 1
            case 3:
                self._literal_best_false += 1
            case _:
                self._literal_best_other += 1

        tied_with_selected_literals = [
            action
            for action in tied
            if self._selected_literal(action) is not None
        ]
        self._literal_tied_mappable += len(tied_with_selected_literals)
        self._literal_tied_mapped += sum(
            1
            for action in tied_with_selected_literals
            if self._selected_sat_literal(action) is not None
        )

        if isinstance(chosen, Start):
            self._literal_chosen_start += 1
        elif isinstance(chosen, Reduction):
            self._literal_chosen_reduction += 1
        elif isinstance(chosen, Extension):
            self._literal_chosen_extension += 1

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        return min(actions, key = self.score, default = None)

    def step(self):
        state = self.env.state
        actions = self.env.legal_actions()
        self._record_nonzero_atom_scores()
        action = self._choose_action(actions)
        self._record_literal_choice_diagnostics(actions, action)

        if isinstance(action, (Reduction, Extension)):
            action.cut = True

        if isinstance(action, (Start, Reduction, Extension)):
            return action

        if (
            state.goal.parent is None
            and any(isinstance(action, Reset) for action in actions)
            and state.shadow.new_ground_clauses
        ):
            return Reset(id = 'rs', depth = state.max_depth)

        for action in actions:
            if isinstance(action, Backtrack):
                return action

        for action in actions:
            if isinstance(action, Reset):
                return action

        raise LookupError('expected Backtrack or Reset action')

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        if self._literal_choice_points:
            avg_tie = (
                self._literal_tied_actions / self._literal_ties
                if self._literal_ties
                else 0.0
            )
            parts.append(
                'literal='
                f'choices:{self._literal_choice_points},'
                f'forward:{self._literal_forward_actions},'
                f'unique:{self._literal_unique_best},'
                f'ties:{self._literal_ties},'
                f'avg_tie:{avg_tie:.2f},'
                f'best_true:{self._literal_best_true},'
                f'best_unknown:{self._literal_best_unknown},'
                f'best_start:{self._literal_best_start},'
                f'best_false:{self._literal_best_false},'
                f'best_other:{self._literal_best_other},'
                f'tied_mappable:{self._literal_tied_mappable},'
                f'tied_mapped:{self._literal_tied_mapped},'
                f'chosen_start:{self._literal_chosen_start},'
                f'chosen_reduction:{self._literal_chosen_reduction},'
                f'chosen_extension:{self._literal_chosen_extension}'
            )
        return '; '.join(parts) if parts else None

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
            conflict = int(pydical is not None and solver.solve() == pydical.UNSATISFIABLE)
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

class SATSolveLiteralRandomTie(_SATSolveLiteralTieMixin, SATSolveLiteral):
    """SATSolveLiteral, but choose randomly among actions tied on model value."""

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        tied = self._best_score_tied_actions(actions)
        if not tied:
            return None

        rng = getattr(self.env.state, 'random', random)
        return rng.choice(tied)

class SATSolveLiteralShortTie(_SATSolveLiteralTieMixin, SATSolveLiteral):
    """SATSolveLiteral with reduction/short-extension tie-breaking."""

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        tied = self._best_score_tied_actions(actions)
        return min(tied, key = self._connection_tie_key, default = None)

class SATSolveLiteralLookaheadTie(_SATSolveLiteralTieMixin, SATSolveLiteral):
    """SATSolveLiteral with CaDiCaL lookahead/EVSIDS tie-breaking.

    The SAT model value remains the primary score.  CaDiCaL data is only
    used inside the best model-value bucket, so this variant measures
    whether SAT branching data adds anything beyond SATCoP literal
    selection.
    """

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
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

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
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

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        tied = self._best_score_tied_actions(actions)
        return self._choose_by_propagation(tied)
