"""Rust-style SATCoP control approximations."""

from __future__ import annotations

from connections.calculi.classicalsat import Backtrack, Extension, Reduction, Reset, Start

from .base import Strategy
from .sat_solve import _SATScoreTracker


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
