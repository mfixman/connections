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

class _StartClauseSignals(NamedTuple):
    mapped: int
    unmapped: int
    min_frequency: int
    total_frequency: int
    length: int
    model_false: int
    model_true: int
    model_unknown: int

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

class SATCoPRust(_SATScoreTracker, Strategy):
    """Approximate the upstream SATCoP Rust search policy on ``ClassicalSAT``.

    This follows the Rust control flow in ``packages/satcop/src/search.rs``
    as closely as the ``ClassicalSAT`` action API allows:

    - continuously refresh the shadow SAT state rather than only at resets;
    - commit successful reductions/extensions via ``cut`` to emulate Rust's
      restricted backtracking;
    - always prefer the first available ``Start`` at the root;
    - otherwise try the first available ``Reduction`` before any extension;
    - otherwise try the first available ``Extension`` in the calculus'
      existing randomized order;
    - within a chosen extension, pre-insert the extension's ground clause
      into the shadow SAT solver and re-solve (matching Rust's immediate
      clause insertion in ``unify()`` and ``sat::Solver::assert``), then
      reorder remaining subgoals: SAT-true first, SAT-false second (closed
      by paper-mode model lemmata), SAT-unknown last;
    - stay at the same depth when new ground clauses were discovered.

    Important approximations:
        The Rust prover only proves SAT-true promises of an extension
        clause (``search.rs:163-178``), then returns success without
        attempting the remaining (false/unknown) subgoals. Soundness
        comes from the ground SAT solver, not the tableau proof.
        ``ClassicalSAT`` always materializes the whole clause as tableau
        children, so this strategy cannot skip subgoals outright.  It
        mitigates the gap by:

        - Pre-inserting the extension clause so the model is as current
          as the Rust's ``sat::Solver::assert`` (which calls ``solve()``
          after every clause addition, ``sat.rs:154``).
        - Placing SAT-false promises before unknowns so paper-mode model
          lemmata close them instantly, matching the Rust's skip for
          those literals.
        - SAT-unknown subgoals remain: they must be proved in Python but
          are skipped entirely in the Rust prover.

        The Rust SAT solver is a custom WalkSAT/PicoSAT hybrid
        (``sat.rs``) that re-solves incrementally after every clause
        insertion.  The Python shadow uses CaDiCaL, which is also
        incremental but is only re-solved at synchronisation points.
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_sat_score_tracker()
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('SATCoPRust requires the ClassicalSAT calculus')
        self.env.state.enable_paper_mode(True)

    def _pre_insert_extension_clause(self, action: Extension) -> None:
        """Insert the extension clause's ground version into the shadow SAT
        solver and re-solve to update the model.

        The Rust prover inserts ground clauses inside ``unify()``
        (``search.rs:216-221``) immediately after every successful
        unification, then calls ``sat::Solver::solve`` (``sat.rs:154``).
        Doing this before the subgoal-ordering decision gives the SAT
        model the same information the Rust solver would have at the
        corresponding point in its recursive search.

        Requires: the substitution already includes ``action.sub_updates``.
        """
        state = self.env.state
        if not action.clause_copy:
            return
        try:
            ground_clause, expanded = state._ground_clause_data(
                action.clause_copy, apply_substitution = True,
            )
        except Exception:
            return
        if ground_clause in state.shadow.ground_clauses:
            return
        state.shadow.expanded_clauses.setdefault(ground_clause, expanded)
        state.shadow.new_ground_clauses.add(ground_clause)
        state._flush_new_ground_clauses()
        state._solve_ground_clauses()

    def _extension_with_true_promises_first(self, action: Extension) -> Extension:
        state = self.env.state
        if not action.clause_copy or not (0 <= action.lit_idx < len(action.clause_copy)):
            return action

        head_literal = action.clause_copy[action.lit_idx]
        remaining = [
            literal
            for idx, literal in enumerate(action.clause_copy)
            if idx != action.lit_idx
        ]

        state.substitution.update(action.sub_updates)
        try:
            self._pre_insert_extension_clause(action)

            true_promises = []
            false_promises = []
            unknown_promises = []
            for literal in remaining:
                value = state.sat_literal_value(literal)
                if value is True:
                    true_promises.append(literal)
                elif value is False:
                    false_promises.append(literal)
                else:
                    unknown_promises.append(literal)
        finally:
            state.substitution.backtrack()

        if not true_promises and not false_promises:
            return action

        # Rust order (search.rs:163-178): SAT-true promises are proved,
        # everything else is skipped.  We can't skip, but SAT-false
        # promises are auto-closed by paper-mode model lemmata, so
        # placing them before unknowns minimises the residual gap.
        action.clause_copy = (
            [head_literal] + true_promises + false_promises + unknown_promises
        )
        action.lit_idx = 0
        return action

    def step(self):
        actions = self.env.legal_actions()

        self._sync_shadow_solver()
        self._record_nonzero_atom_scores()

        if action := next((a for a in actions if isinstance(a, Start)), None):
            return action

        if action := next((a for a in actions if isinstance(a, Reduction)), None):
            action.cut = True
            return action

        if action := next((a for a in actions if isinstance(a, Extension)), None):
            action.cut = True
            return self._extension_with_true_promises_first(action)

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

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
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

class _SATSolveLiteralStartMixin:
    """Choose root start clauses using cheap statistics from the shadow CNF."""

    _start_mode = 'rare'

    def _init_start_guidance(self) -> None:
        self._start_frequency_clauses = set()
        self._start_atom_counts = {}
        self._start_signed_counts = {}
        self._start_decisions = 0
        self._start_candidates = 0
        self._start_mapped = 0
        self._start_unmapped = 0
        self._start_changed_default = 0
        self._start_agreed_default = 0
        self._start_unique_best = 0
        self._start_ties = 0
        self._start_total_min_frequency = 0
        self._start_model_false = 0
        self._start_model_true = 0
        self._start_model_unknown = 0

    def _sync_start_frequency_counts(self) -> None:
        current_clauses = set(self.env.state.shadow.ground_clauses)
        if not self._start_frequency_clauses.issubset(current_clauses):
            self._start_frequency_clauses = set()
            self._start_atom_counts = {}
            self._start_signed_counts = {}

        for clause in current_clauses - self._start_frequency_clauses:
            for sat_lit in clause:
                atom_id = abs(sat_lit)
                self._start_atom_counts[atom_id] = self._start_atom_counts.get(atom_id, 0) + 1
                self._start_signed_counts[sat_lit] = self._start_signed_counts.get(sat_lit, 0) + 1
            self._start_frequency_clauses.add(clause)

    def _start_sat_literals(self, action: Start) -> list[int]:
        if action.clause_copy is None:
            return []

        state = self.env.state
        if not hasattr(state, '_atom_key'):
            return []

        substitution = getattr(state, 'substitution', None)
        sat_literals = []
        seen = set()
        for literal in action.clause_copy:
            grounded_literal = substitution(literal) if callable(substitution) else literal
            atom_id = state.shadow.atom_map.get(state._atom_key(grounded_literal))
            if atom_id is None:
                continue
            sat_lit = -atom_id if grounded_literal.neg else atom_id
            if sat_lit not in seen:
                sat_literals.append(sat_lit)
                seen.add(sat_lit)
        return sat_literals

    def _start_clause_signals(self, action: Start) -> _StartClauseSignals:
        sat_literals = self._start_sat_literals(action)
        frequencies = [self._start_atom_counts.get(abs(sat_lit), 0) for sat_lit in sat_literals]
        model = getattr(self.env.state.shadow, 'model', {})

        model_false = 0
        model_true = 0
        model_unknown = 0
        for sat_lit in sat_literals:
            value = model.get(abs(sat_lit))
            if value is None:
                model_unknown += 1
                continue
            lit_value = value if sat_lit > 0 else (not value)
            if lit_value:
                model_true += 1
            else:
                model_false += 1

        length = len(action.clause_copy or [])
        return _StartClauseSignals(
            mapped = len(sat_literals),
            unmapped = max(0, length - len(sat_literals)),
            min_frequency = min(frequencies) if frequencies else 0,
            total_frequency = sum(frequencies),
            length = length,
            model_false = model_false,
            model_true = model_true,
            model_unknown = model_unknown,
        )

    def _start_clause_score(self, signals: _StartClauseSignals) -> tuple[int, ...]:
        mapped = int(signals.mapped > 0)
        rare = (-signals.min_frequency, -signals.total_frequency, -signals.length)
        if self._start_mode == 'model':
            return (
                mapped,
                signals.model_false,
                -signals.model_true,
                -signals.model_unknown,
                *rare,
            )
        return (mapped, *rare)

    def _choose_start_clause(self, starts: list[Start]) -> Start | None:
        if not starts:
            return None

        self._sync_start_frequency_counts()
        default_choice = starts[0]
        scored = [
            (action, self._start_clause_signals(action))
            for action in starts
        ]

        self._start_decisions += 1
        self._start_candidates += len(starts)
        self._start_mapped += sum(1 for _, signals in scored if signals.mapped > 0)
        self._start_unmapped += sum(1 for _, signals in scored if signals.mapped == 0)
        self._start_total_min_frequency += sum(signals.min_frequency for _, signals in scored)
        self._start_model_false += sum(signals.model_false for _, signals in scored)
        self._start_model_true += sum(signals.model_true for _, signals in scored)
        self._start_model_unknown += sum(signals.model_unknown for _, signals in scored)

        best_score = max(self._start_clause_score(signals) for _, signals in scored)
        best = [
            action
            for action, signals in scored
            if self._start_clause_score(signals) == best_score
        ]
        if len(best) == 1:
            self._start_unique_best += 1
        else:
            self._start_ties += 1

        chosen = best[0]
        if chosen is default_choice:
            self._start_agreed_default += 1
        else:
            self._start_changed_default += 1
        return chosen

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        starts = [action for action in actions if isinstance(action, Start)]
        if len(starts) > 1 and min(self.score(action) for action in actions) == self.score(starts[0]):
            return self._choose_start_clause(starts)
        return super()._choose_action(actions)

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        if self._start_decisions:
            avg_min_frequency = self._start_total_min_frequency / self._start_candidates
            parts.append(
                'start='
                f'mode:{self._start_mode},'
                f'decisions:{self._start_decisions},'
                f'candidates:{self._start_candidates},'
                f'mapped:{self._start_mapped},'
                f'unmapped:{self._start_unmapped},'
                f'unique:{self._start_unique_best},'
                f'ties:{self._start_ties},'
                f'avg_min_freq:{avg_min_frequency:.2f},'
                f'model_false:{self._start_model_false},'
                f'model_true:{self._start_model_true},'
                f'model_unknown:{self._start_model_unknown},'
                f'agreed_default:{self._start_agreed_default},'
                f'changed_default:{self._start_changed_default}'
            )
        return '; '.join(parts) if parts else None

class SATSolveLiteralRareStart(_SATSolveLiteralStartMixin, SATSolveLiteral):
    """SATSolveLiteral with rare-SAT-atom ordering for root start clauses.

    The strategy leaves Rawson-Reger literal selection unchanged after a
    start clause has been selected.  At the root, where SATSolveLiteral
    otherwise follows the randomised start order, it prefers start
    clauses containing rarer atoms in the current shadow CNF and falls
    back to shorter clauses.
    """

    _start_mode = 'rare'

    def __init__(self, env):
        super().__init__(env)
        self._init_start_guidance()

class SATSolveLiteralModelStart(_SATSolveLiteralStartMixin, SATSolveLiteral):
    """SATSolveLiteral with SAT-model-aware root start ordering.

    After the first SAT iteration this prefers start clauses with more
    literals currently false in the SAT model, because model-false goals
    are immediately closed by SATCoP's model lemmata.  Before a model is
    available, or inside equal model buckets, it uses the same rare-atom
    fallback as ``SATSolveLiteralRareStart``.
    """

    _start_mode = 'model'

    def __init__(self, env):
        super().__init__(env)
        self._init_start_guidance()

class SATSolveLiteralFirstStart(_SATSolveLiteralStartMixin, SATSolveLiteral):
    """SATSolveLiteral with one SAT-guided start choice per fresh root list.

    This is the conservative version of the Start-guidance experiment.
    When a reset creates a fresh root list containing all start clauses,
    the strategy scores only the first few randomized starts and uses
    SAT metadata once to choose the first start clause to try.  After
    backtracking returns to root, the remaining starts are left in the
    ordinary SATSolveLiteral/randomised order.  The score is model-aware
    when a SAT model exists, and otherwise falls back to the rare-atom
    signal from the initial shadow CNF.
    """

    _start_mode = 'model'
    _start_candidate_limit = 32

    def __init__(self, env):
        super().__init__(env)
        self._init_start_guidance()
        self._fresh_start_count = None

    def _root_start_count(self) -> int:
        state = self.env.state
        if hasattr(state, '_start_clause_indices'):
            return max(1, len(state._start_clause_indices()))
        if self._fresh_start_count is None:
            self._fresh_start_count = 0
        return self._fresh_start_count

    def _is_fresh_root_start_list(self, starts: list[Start]) -> bool:
        state = self.env.state
        goal = getattr(state, 'goal', None)
        if goal is None or getattr(goal, 'parent', None) is not None:
            return False

        expected = self._root_start_count()
        if expected == 0:
            self._fresh_start_count = len(starts)
            expected = self._fresh_start_count
        return len(starts) == expected

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        if not actions:
            return None

        starts = [action for action in actions if isinstance(action, Start)]
        best_score = min(self.score(action) for action in actions)
        if (
            len(starts) > 1
            and best_score == self.score(starts[0])
            and self._is_fresh_root_start_list(starts)
        ):
            return self._choose_start_clause(starts[:self._start_candidate_limit])

        return SATSolveLiteral._choose_action(self, actions)

class _SATSolveLiteralFridayMixin(_SATSolveLiteralStartMixin):
    """Root-control experiments guided by SAT start-clause metadata."""

    _start_mode = 'model'
    _friday_enable_start_guidance = True
    _friday_commit_on_confident_start = False
    _friday_enable_root_stagnation = False
    _friday_use_first_start = False
    _friday_root_visit_limit = 8
    _start_candidate_limit = 32

    def _init_friday_controls(self) -> None:
        self._init_start_guidance()
        self._fresh_start_count = None
        self._friday_committed_depth = None
        self._friday_root_depth = None
        self._friday_root_visits = 0
        self._friday_root_clause_count = None
        self._friday_confident_starts = 0
        self._friday_commit_resets = 0
        self._friday_stagnation_resets = 0

    def _root_start_count(self) -> int:
        state = self.env.state
        if hasattr(state, '_start_clause_indices'):
            return max(1, len(state._start_clause_indices()))
        if self._fresh_start_count is None:
            self._fresh_start_count = 0
        return self._fresh_start_count

    def _is_fresh_root_start_list(self, starts: list[Start]) -> bool:
        state = self.env.state
        goal = getattr(state, 'goal', None)
        if goal is None or getattr(goal, 'parent', None) is not None:
            return False

        expected = self._root_start_count()
        if expected == 0:
            self._fresh_start_count = len(starts)
            expected = self._fresh_start_count
        return len(starts) == expected

    def _choose_start_clause_with_confidence(self, starts: list[Start]) -> tuple[Start | None, bool]:
        if not starts:
            return None, False

        self._sync_start_frequency_counts()
        default_choice = starts[0]
        scored = [
            (action, self._start_clause_signals(action))
            for action in starts
        ]

        self._start_decisions += 1
        self._start_candidates += len(starts)
        self._start_mapped += sum(1 for _, signals in scored if signals.mapped > 0)
        self._start_unmapped += sum(1 for _, signals in scored if signals.mapped == 0)
        self._start_total_min_frequency += sum(signals.min_frequency for _, signals in scored)
        self._start_model_false += sum(signals.model_false for _, signals in scored)
        self._start_model_true += sum(signals.model_true for _, signals in scored)
        self._start_model_unknown += sum(signals.model_unknown for _, signals in scored)

        best_score = max(self._start_clause_score(signals) for _, signals in scored)
        best = [
            (action, signals)
            for action, signals in scored
            if self._start_clause_score(signals) == best_score
        ]
        if len(best) == 1:
            self._start_unique_best += 1
        else:
            self._start_ties += 1

        chosen, chosen_signals = best[0]
        if chosen is default_choice:
            self._start_agreed_default += 1
        else:
            self._start_changed_default += 1

        if len(best) != 1:
            return chosen, False

        other_signals = [
            signals
            for action, signals in scored
            if action is not chosen
        ]
        if not other_signals or chosen_signals.mapped == 0:
            return chosen, False

        strong_model = (
            chosen_signals.model_false > 0
            and chosen_signals.model_false > max(signals.model_false for signals in other_signals)
        )
        strong_rarity = chosen_signals.min_frequency < min(
            signals.min_frequency for signals in other_signals
        )
        return chosen, (strong_model or strong_rarity)

    def _maybe_friday_root_reset(self, starts: list[Start]) -> Reset | None:
        state = self.env.state
        expected = self._root_start_count()
        clause_count = len(getattr(state.shadow, 'ground_clauses', ()))

        if self._friday_root_depth != state.max_depth:
            self._friday_committed_depth = None
            self._friday_root_depth = state.max_depth
            self._friday_root_visits = 0
            self._friday_root_clause_count = clause_count

        if (
            self._friday_commit_on_confident_start
            and self._friday_committed_depth == state.max_depth
            and 0 < len(starts) < expected
        ):
            self._friday_committed_depth = None
            self._friday_commit_resets += 1
            self._friday_root_visits = 0
            self._friday_root_clause_count = clause_count
            return Reset(id = 'rs', depth = state.max_depth + 1, reason = 'friday committed start failed')

        if not self._friday_enable_root_stagnation:
            return None

        if len(starts) >= expected:
            self._friday_root_visits = 0
            self._friday_root_clause_count = clause_count
            return None

        if getattr(state.shadow, 'new_ground_clauses', set()) or clause_count > (self._friday_root_clause_count or 0):
            self._friday_root_visits = 0
            self._friday_root_clause_count = clause_count
            return None

        self._friday_root_visits += 1
        if self._friday_root_visits < self._friday_root_visit_limit:
            return None

        self._friday_stagnation_resets += 1
        self._friday_root_visits = 0
        self._friday_root_clause_count = clause_count
        return Reset(id = 'rs', depth = state.max_depth + 1, reason = 'friday root stagnation')

    def _choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        if not actions:
            return None

        state = self.env.state
        starts = [action for action in actions if isinstance(action, Start)]
        goal = getattr(state, 'goal', None)
        at_root = goal is not None and getattr(goal, 'parent', None) is None

        if at_root and starts:
            if reset := self._maybe_friday_root_reset(starts):
                return reset

            best_score = min(self.score(action) for action in actions)
            if (
                self._friday_enable_start_guidance
                and len(starts) > 1
                and best_score == self.score(starts[0])
            ):
                if self._friday_use_first_start and not self._is_fresh_root_start_list(starts):
                    return SATSolveLiteral._choose_action(self, actions)

                candidates = starts[:self._start_candidate_limit] if self._friday_use_first_start else starts
                chosen, confident = self._choose_start_clause_with_confidence(candidates)
                if chosen is not None:
                    if confident and self._friday_commit_on_confident_start:
                        self._friday_confident_starts += 1
                        self._friday_committed_depth = state.max_depth
                    return chosen

        return SATSolveLiteral._choose_action(self, actions)

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        if self._friday_confident_starts or self._friday_commit_resets or self._friday_stagnation_resets:
            parts.append(
                'friday='
                f'confident_starts:{self._friday_confident_starts},'
                f'commit_resets:{self._friday_commit_resets},'
                f'stagnation_resets:{self._friday_stagnation_resets}'
            )
        return '; '.join(parts) if parts else None

class SATSolveLiteralFridayStartCut(_SATSolveLiteralFridayMixin, SATSolveLiteral):
    """SAT-guided root-start choice with selective root commitment."""

    _friday_commit_on_confident_start = True

    def __init__(self, env):
        super().__init__(env)
        self._init_friday_controls()

class SATSolveLiteralFridayRootReset(_SATSolveLiteralFridayMixin, SATSolveLiteral):
    """SATSolveLiteral with early reset on repeated stale root revisits."""

    _friday_enable_start_guidance = False
    _friday_enable_root_stagnation = True

    def __init__(self, env):
        super().__init__(env)
        self._init_friday_controls()

class SATSolveLiteralFridayHybrid(_SATSolveLiteralFridayMixin, SATSolveLiteral):
    """Conservative Friday hybrid: first-start guidance, commitment, stagnation reset."""

    _friday_commit_on_confident_start = True
    _friday_enable_root_stagnation = True
    _friday_use_first_start = True

    def __init__(self, env):
        super().__init__(env)
        self._init_friday_controls()

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
