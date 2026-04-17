"""Strategy for the SAT-aware classical connection calculus."""

from __future__ import annotations

from connections.calculi.classicalcut import Backtrack, Extension, Reduction, Reset, Start
from .base import Strategy

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

    def step(self):
        state = self.env.state
        actions = self.env.legal_actions()
        action = min(actions, key = self.score, default = None)

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
