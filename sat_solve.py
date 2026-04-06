"""Strategy for the SAT-aware classical connection calculus."""

from __future__ import annotations

from typing import Optional

from connections.calculi.classicalcut import Backtrack, ConnectionAction, Extension, Reduction, Reset, Start
from connections.utils.primitives import Literal
from strategy_base import Strategy

def _backtrack_or_reset(actions: list[ConnectionAction]) -> ConnectionAction:
    for action in actions:
        if isinstance(action, Backtrack):
            return action
    for action in actions:
        if isinstance(action, Reset):
            return action
    raise LookupError('expected Backtrack or Reset action')

def _forward_actions(actions: list[ConnectionAction]) -> list[ConnectionAction]:
    return [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

class SATSolve(Strategy):
    """Use SAT state from ``connections.calculi.classicalsat`` to pick actions."""

    def __init__(self, env):
        super().__init__(env)
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('SATSolve requires the ClassicalSAT calculus')

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if not forward:
            return self._dead_end_action(actions)
        if len(forward) == 1:
            return forward[0]

        return max(forward, key = self._score)

    def _dead_end_action(self, actions: list[ConnectionAction]) -> ConnectionAction:
        if any(isinstance(action, Reset) for action in actions) and self.env.state.new_ground_clauses:
            return Reset(id = 'rs', depth = self.env.state.max_depth, reason = 'ground shadow still growing')
        return _backtrack_or_reset(actions)

    def _score(self, action: ConnectionAction) -> tuple[int, ...]:
        if isinstance(action, Reduction):
            truth = self._literal_truth(action.principle_node.literal)
            truth_rank = 2 if truth is True else 1 if truth is None else 0
            return (2, truth_rank)

        if isinstance(action, Start):
            clause = action.clause_copy
            residual = clause
            kind_rank = 0
        else:
            clause = action.clause_copy
            residual = [lit for i, lit in enumerate(clause) if i != action.lit_idx]
            kind_rank = 1

        grounded_clause = self.env.state.ground_clause(clause)
        novelty_rank = int(grounded_clause not in self.env.state.ground_clauses)
        true_count = 0
        false_count = 0
        for lit in residual:
            truth = self._literal_truth(lit)
            if truth is True:
                true_count += 1
            elif truth is False:
                false_count += 1
        return (kind_rank, novelty_rank, true_count, -false_count, -len(residual))

    def _literal_truth(self, literal: Literal) -> Optional[bool]:
        return self.env.state.sat_literal_value(literal)

class SATSolveLiteral(Strategy):
    """Baseline SATCoP strategy from Rawson & Reger 2021 (Algorithm 2).

    Implements the paper's key contributions without action scoring:
    - Random action selection (order shuffled by the ClassicalSAT
      calculus — Sect. 5 randomisation).
    - Depth control: stays at the same iterative-deepening level
      while new ground clauses are being discovered; only increases
      depth when they stop (Sect. 5).

    Grounding, UNSAT detection, model-based lemmata, and clause
    randomisation are handled by the ClassicalSAT calculus.

    Differences from the paper:
    - Restricted backtracking (Sect. 3, ref [32]) is a calculus-level
      property that cannot be faithfully reproduced at the strategy
      level because SATConnectionState regenerates forward actions
      on every goal visit.  Backtracking is therefore unrestricted.
    - Resets may occur mid-search at any dead end (not only after
      exhausting all start clauses), because start actions are
      regenerated on every return to root.

    Compared to SATSolve, the only change is that forward actions
    are selected randomly (first shuffled action) rather than scored
    by SAT truth values.
    """

    def __init__(self, env):
        super().__init__(env)
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('SATSolveLiteral requires the ClassicalSAT calculus')

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        # Forward action available: take the first one (randomised
        # by the calculus).  Reductions come before extensions.
        if forward:
            return forward[0]

        # Dead end: stay at the same depth while new ground clauses
        # are being discovered (Sect. 5 depth control).
        state = self.env.state
        if any(isinstance(a, Reset) for a in actions) and state.new_ground_clauses:
            return Reset(id = 'rs', depth = state.max_depth)
        return _backtrack_or_reset(actions)
