"""Strategy for the SAT-aware classical connection calculus."""

from __future__ import annotations

from connections.calculi.classicalcut import Backtrack, ConnectionAction, Extension, Reduction, Reset, Start
from .base import Strategy

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

def _action_head_literal(action: ConnectionAction):
    """Return the first-order literal a forward action is closing.

    Reductions close the current goal against a path literal; extensions
    unify the current goal with ``clause_copy[lit_idx]``; starts have no
    single head literal.
    """
    if isinstance(action, Reduction):
        return action.path_lit
    if isinstance(action, Extension):
        if 0 <= action.lit_idx < len(action.clause_copy):
            return action.clause_copy[action.lit_idx]
    return None

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
    - The SAT backend is pycosat/pysat rather than the custom CDCL
      from §4.3.
    - Start clauses are taken from the positive+negative pool rather
      than strictly "clauses derived from the conjecture".
    """

    def __init__(self, env):
        super().__init__(env)
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('SATSolveLiteral requires the ClassicalSAT calculus')
        if hasattr(self.env.state, 'enable_paper_mode'):
            self.env.state.enable_paper_mode(True)

    def _score(self, action: ConnectionAction) -> int:
        """Lower is better: pick the literal the SAT model says is True."""
        literal = _action_head_literal(action)
        if literal is None:
            return 2  # start actions — no head literal
        value = self.env.state.sat_literal_value(literal)
        if value is True:
            return 0  # literal selection wants this first
        if value is None:
            return 1  # unknown / non-ground: preserve calculus order
        return 3      # already assigned false: leave for last

    def _choose_forward(self, forward: list[ConnectionAction]) -> ConnectionAction:
        if len(forward) == 1:
            return forward[0]
        # ``min`` with a key is stable w.r.t. the input order, so ties
        # preserve the randomised order produced by the calculus.
        best_idx = min(range(len(forward)), key = lambda i: (self._score(forward[i]), i))
        return forward[best_idx]

    def step(self):
        actions = self.env.legal_actions()
        forward = _forward_actions(actions)

        if forward:
            return _paper_forward(self._choose_forward(forward))

        state = self.env.state
        if _paper_reset_allowed(actions, state):
            return Reset(id = 'rs', depth = state.max_depth)
        return _backtrack_or_reset(actions)
