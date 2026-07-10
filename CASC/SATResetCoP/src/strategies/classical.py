"""Public Classical/ClassicalCut strategy surface."""

from __future__ import annotations

from connections.calculi.classicalcut import Backtrack, Extension, Reduction, Reset

from .base import Strategy


class PyCoP(Strategy):
    def step(self):
        return self.env.legal_actions()[0]


class LeanCoP(Strategy):
    """Mirror leanCoP's ``cut`` option for forward proof steps."""

    max_cut_height: int = 7
    should_comp: bool

    def __init__(self, env):
        super().__init__(env)
        self.should_comp = True

    def step(self):
        action = self.env.legal_actions()[0]

        if not self.should_comp:
            return action

        match action:
            case Reduction() | Extension():
                action.cut = True

            case Reset(depth = depth):
                if depth > self.max_cut_height:
                    self.should_comp = False
                    return Reset(id = 'rs', depth = 1)

            case Backtrack():
                return action

        return action


class FirstAction(PyCoP):
    """Backward-compatible hidden name for the old public baseline."""


class LeanCopCut(LeanCoP):
    """Backward-compatible hidden name for the old public leanCoP-cut baseline."""
