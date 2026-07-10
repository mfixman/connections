"""Hidden SAT-reset strategy variants."""

from __future__ import annotations

from connections.calculi.classicalsat import Backtrack, ConnectionAction, Extension, Reduction, Reset, Start

from ..sat_reset import (
    _ControlBanditMixin,
    _SATResetBase,
    SATResetAtStaleBacktrack,
    SATResetAtTwo,
    SATResetShadow,
)


class SATResetBandit(_ControlBanditMixin, _SATResetBase):
    def __init__(self, env):
        super().__init__(env)
        self.init_control_bandit()

    def before_step(self, actions: list[ConnectionAction]) -> None:
        self.settle_control_feedback(actions)

    def dead_end_action(self, actions: list[ConnectionAction]) -> ConnectionAction:
        state = self.env.state
        if (
            state.goal.parent is None
            and any(isinstance(candidate, Reset) for candidate in actions)
            and state.shadow.new_ground_clauses
        ):
            return self.same_depth_reset('root_shadow')

        action = self.choose_control_action(actions)
        if action is None:
            raise LookupError('expected Backtrack or Reset action')
        if isinstance(action, Reset):
            return self.copy_reset(action, 'bandit_reset')
        return action

    def final_info(self) -> str | None:
        self.settle_control_feedback(self.env.legal_actions(), force = True)

        parts = []
        base_info = _SATResetBase.final_info(self)
        if base_info is not None:
            parts.append(base_info)
        control_info = self.control_final_info_piece()
        if control_info is not None:
            parts.append(control_info)
        return '; '.join(parts) if parts else None


class SATResetAtStaleBacktrackTableau(SATResetAtStaleBacktrack):
    def __init__(self, env):
        super().__init__(env)
        self.env.state.enable_paper_mode(False)


class SATResetAtTwoDepthStep(SATResetAtTwo):
    def step_budget(self) -> int:
        return max(8, 2 * max(1, self.env.state.max_depth))

    def proactive_reset(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> Reset | None:
        del chosen
        reset_action = self.available_reset(actions)
        if reset_action is None:
            return None
        if self._since_reset_steps < self.step_budget():
            return None
        if getattr(self.env.state.shadow, 'new_ground_clauses', ()):
            return self.same_depth_reset('depth_step_shadow')
        return self._zero_yield_reset(reset_action)


class SATResetAtTwoTableau(SATResetAtTwo):
    def __init__(self, env):
        super().__init__(env)
        self.env.state.enable_paper_mode(False)


class SATResetExtensionBudget(SATResetShadow):
    def extension_budget(self) -> int:
        return max(4, max(1, self.env.state.max_depth))

    def proactive_reset(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> Reset | None:
        if not isinstance(chosen, (Start, Reduction, Extension)):
            return None
        if self.available_reset(actions) is None:
            return None
        if self._since_reset_extensions < self.extension_budget():
            return None
        return self.same_depth_reset('extension_budget')
