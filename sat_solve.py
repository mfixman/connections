"""Strategy for the SAT-aware classical connection calculus."""

from __future__ import annotations

from connections.calculi.classicalcut import Backtrack, ConnectionAction, Extension, Reduction, Reset, Start
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

def _paper_reset_allowed(actions: list[ConnectionAction], state) -> bool:
    return (
        state.goal.parent is None
        and any(isinstance(action, Reset) for action in actions)
        and bool(state.new_ground_clauses())
    )

def _paper_forward(action: ConnectionAction) -> ConnectionAction:
    if isinstance(action, (Reduction, Extension)):
        action.cut = True
    return action

class SATSolve(Strategy):
    """Paper-faithful SATCoP control for ``ClassicalSAT``.

    This mirrors Rawson and Reger (2021), Algorithm 2:
    randomised forward selection from the calculus, restricted
    backtracking on successful reduction/extension steps, depth control
    based on whether new grounded clauses were found, and model-based
    lemmata handled by the state machine.
    """

    def __init__(self, env):
        super().__init__(env)
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('SATSolve requires the ClassicalSAT calculus')
        if hasattr(self.env.state, 'enable_paper_mode'):
            self.env.state.enable_paper_mode(True)

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if not forward:
            if _paper_reset_allowed(actions, self.env.state):
                return Reset(id = 'rs', depth = self.env.state.max_depth, reason = 'ground shadow still growing')
            return _backtrack_or_reset(actions)
        return _paper_forward(forward[0])

class SATSolveLiteral(Strategy):
    """Randomised SAT-aware baseline for ``ClassicalSAT``.

    Implements the SAT-guided search control from Rawson and Reger
    (2021, Sects. 4–5), without the model-based lemmata of Sect. 6:
    - Random action selection (order shuffled by the ClassicalSAT
      calculus — Sect. 5 randomisation).
    - Restricted backtracking on successful reduction/extension steps,
      matching Algorithm 2.
    - Depth control: stays at the same iterative-deepening level
      while new ground clauses are being discovered; only increases
      depth when they stop (Sect. 5).

    Grounding, UNSAT detection, and clause randomisation are handled
    by the ClassicalSAT calculus.

    Remaining differences from the paper:
    - The Python port does not implement the model-based lemmata
      shortcut from Sect. 6.
    - The SAT backend is external rather than SATCoP's custom
      incremental CDCL implementation from Sect. 4.3.

    Compared to SATSolve, the only change is that forward actions
    are selected randomly (first shuffled action) rather than scored
    by SAT truth values.
    """

    def __init__(self, env):
        super().__init__(env)
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('SATSolveLiteral requires the ClassicalSAT calculus')
        if hasattr(self.env.state, 'enable_paper_mode'):
            self.env.state.enable_paper_mode(False)

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        # Forward action available: take the first one (randomised
        # by the calculus).  Reductions come before extensions.
        if forward:
            return _paper_forward(forward[0])

        # Dead end: stay at the same depth while new ground clauses
        # are being discovered, but only after exhausting the current
        # iteration's start clauses (Algorithm 2 / Sect. 5).
        state = self.env.state
        if _paper_reset_allowed(actions, state):
            return Reset(id = 'rs', depth = state.max_depth)
        return _backtrack_or_reset(actions)
