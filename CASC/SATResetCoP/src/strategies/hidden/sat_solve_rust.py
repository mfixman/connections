"""Rust-style SATCoP control approximations."""

from __future__ import annotations

from connections.calculi.classicalsat import Backtrack, Extension, Reduction, Reset, Start

from ..base import Strategy
from ..sat_solve import _SATScoreTracker


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
    - within a chosen extension, pre-insert the current tableau clauses and
      the extension's ground clause into the shadow SAT solver and re-solve
      (matching Rust's immediate clause insertion in ``unify()`` and
      ``sat::Solver::assert``), then keep only SAT-true remaining subgoals;
    - stay at the same depth when new ground clauses were discovered.

    Important approximations:
        The Rust prover only proves SAT-true promises of an extension
        clause (``search.rs:163-178``), then returns success without
        attempting the remaining (false/unknown) subgoals. Soundness
        comes from the ground SAT solver, not the tableau proof.
        ``ClassicalSAT`` normally materializes the whole clause as tableau
        children.  This strategy rewrites the selected extension action so
        only the connected head and SAT-true remaining promises are kept,
        which matches the Rust loop as closely as this action API allows.

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
        self._enable_rust_start_policy()
        self.env.state.enable_paper_mode(True)

    def _enable_rust_start_policy(self) -> None:
        """Use the closest available approximation to Rust's start clauses.

        The Rust loader starts from conjecture-derived clauses when those are
        identifiable, otherwise from positive clauses.  MyPyCoP's Prolog
        translation does not currently preserve per-clause TPTP roles, so the
        closest local setting is the positive-clause fallback rather than the
        default ClassicalSAT positive+negative pool.
        """
        for owner in (self.env, self.env.state):
            settings = getattr(owner, 'settings', None)
            if settings is not None:
                settings.rust_start_clauses = True
                settings.positive_start_clauses = True

    def _queue_ground_clause(self, clause, expanded) -> bool:
        state = self.env.state
        if clause in state.shadow.ground_clauses:
            return False
        state.shadow.expanded_clauses.setdefault(clause, expanded)
        state.shadow.new_ground_clauses.add(clause)
        state._iteration_new_ground_clause = True
        return True

    def _queue_current_tableau_clauses(self) -> bool:
        inserted = False
        state = self.env.state
        tableau_clauses = getattr(state, '_tableau_clauses', None)
        if not callable(tableau_clauses):
            return False
        for clause, expanded in tableau_clauses().items():
            inserted = self._queue_ground_clause(clause, expanded) or inserted
        return inserted

    def _sync_if_inserted(self, inserted: bool) -> None:
        if not inserted:
            return
        state = self.env.state
        state._flush_new_ground_clauses()
        state._solve_ground_clauses()

    def _pre_insert_current_clauses(self, action: Reduction | Extension) -> None:
        """Insert the ground clauses Rust would insert after unification.

        The Rust prover inserts ground clauses inside ``unify()``
        (``search.rs:216-221``) immediately after every successful
        unification, then calls ``sat::Solver::solve`` (``sat.rs:154``).
        Doing this before Python refreshes the next tableau goal gives
        model-based lemmata the closest available SAT model.

        Requires: the substitution already includes ``action.sub_updates``.
        """
        state = self.env.state
        inserted = self._queue_current_tableau_clauses()

        if isinstance(action, Extension) and action.clause_copy:
            try:
                ground_clause, expanded = state._ground_clause_data(
                    action.clause_copy, apply_substitution = True,
                )
            except Exception:
                ground_clause = None
            if ground_clause is not None:
                inserted = self._queue_ground_clause(ground_clause, expanded) or inserted

        self._sync_if_inserted(inserted)

    def _pre_insert_with_action_substitution(self, action: Reduction | Extension) -> None:
        state = self.env.state
        if not hasattr(state, 'substitution'):
            return
        state.substitution.update(action.sub_updates)
        try:
            self._pre_insert_current_clauses(action)
        finally:
            state.substitution.backtrack()

    def _extension_with_rust_promises(self, action: Extension) -> Extension:
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
            self._pre_insert_current_clauses(action)

            true_promises = []
            for literal in remaining:
                if state.sat_literal_value(literal) is True:
                    true_promises.append(literal)
        finally:
            state.substitution.backtrack()

        # Rust order (search.rs:163-178): prove SAT-true promises and skip
        # the rest.  The connected head remains first and marked proven by
        # the ClassicalSAT extension action.
        action.clause_copy = [head_literal] + true_promises
        action.lit_idx = 0
        action.skip_tableau_clause_recording = True
        return action

    def step(self):
        actions = self.env.legal_actions()

        self._sync_shadow_solver()
        self._record_nonzero_atom_scores()

        if action := next((a for a in actions if isinstance(a, Start)), None):
            return action

        if action := next((a for a in actions if isinstance(a, Reduction)), None):
            action.cut = True
            self._pre_insert_with_action_substitution(action)
            return action

        if action := next((a for a in actions if isinstance(a, Extension)), None):
            action.cut = True
            return self._extension_with_rust_promises(action)

        discovered_ground_clause = getattr(self.env.state, 'iteration_discovered_ground_clause', None)
        shadow_still_growing = (
            discovered_ground_clause()
            if callable(discovered_ground_clause)
            else bool(getattr(self.env.state.shadow, 'new_ground_clauses', ()))
        )
        if (
            self.env.state.goal.parent is None
            and any(isinstance(action, Reset) for action in actions)
            and shadow_still_growing
        ):
            return Reset(id = 'rs', depth = self.env.state.max_depth, reason = 'ground shadow still growing')

        if action := next((a for a in actions if isinstance(a, Backtrack)), None):
            return action

        if action := next((a for a in actions if isinstance(a, Reset)), None):
            return action

        raise LookupError('Expected at least one action to be Start, Reduction, Extension, Backtrack, or Reset.')
