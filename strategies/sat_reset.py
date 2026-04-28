"""Reset-oriented SAT-aware connection-tableau strategies."""

from __future__ import annotations

import math

from connections.calculi.classicalsat import (
    Backtrack,
    ConnectionAction,
    Extension,
    Reduction,
    Reset,
    Start,
)

from .sat_solve import SATSolveLiteral


class _BanditTableMixin:
    def _zero_bandit_table(self, contexts: tuple[str, ...], arms: tuple[str, ...], value):
        return {
            context: {arm: value() for arm in arms}
            for context in contexts
        }


class _ResetStatsMixin:
    """Compact per-run reset diagnostics for CSV inspection."""

    def _init_reset_stats(self) -> None:
        self._reset_policy = self.__class__.__name__
        self._reset_forward_cuts = bool(self._forward_cuts_enabled())

        self._reset_starts = 0
        self._reset_reductions = 0
        self._reset_extensions = 0
        self._reset_bt1 = 0
        self._reset_bt2 = 0
        self._reset_total = 0
        self._reset_cut_reductions = 0
        self._reset_cut_extensions = 0

        self._reset_same_depth = 0
        self._reset_deeper = 0
        self._reset_growing = 0
        self._reset_stagnant = 0

        self._reset_reason_counts: dict[str, int] = {}
        self._since_reset_steps = 0
        self._since_reset_extensions = 0

        self._reset_bursts = 0
        self._reset_burst_steps_total = 0
        self._reset_burst_steps_max = 0
        self._reset_burst_extensions_total = 0
        self._reset_burst_extensions_max = 0
        self._reset_pending_new_total = 0
        self._reset_pending_new_max = 0
        self._reset_ground_total = 0
        self._reset_ground_max = 0

    def _bump_reset_reason(self, reason: str) -> None:
        self._reset_reason_counts[reason] = self._reset_reason_counts.get(reason, 0) + 1

    def _record_emitted_action(self, action: ConnectionAction) -> ConnectionAction:
        if isinstance(action, Start):
            self._reset_starts += 1
            self._since_reset_steps += 1
            return action

        if isinstance(action, Reduction):
            self._reset_reductions += 1
            if action.cut:
                self._reset_cut_reductions += 1
            self._since_reset_steps += 1
            return action

        if isinstance(action, Extension):
            self._reset_extensions += 1
            if action.cut:
                self._reset_cut_extensions += 1
            self._since_reset_steps += 1
            self._since_reset_extensions += 1
            return action

        if isinstance(action, Backtrack):
            if action.steps >= 2:
                self._reset_bt2 += 1
            else:
                self._reset_bt1 += 1
            self._since_reset_steps += 1
            return action

        if isinstance(action, Reset):
            state = self.env.state
            current_depth = getattr(state, 'max_depth', action.depth)
            pending_new = len(getattr(state.shadow, 'new_ground_clauses', ()))
            ground_count = len(getattr(state.shadow, 'ground_clauses', ()))

            self._reset_total += 1
            if action.depth <= current_depth:
                self._reset_same_depth += 1
            else:
                self._reset_deeper += 1

            if pending_new:
                self._reset_growing += 1
            else:
                self._reset_stagnant += 1

            self._reset_pending_new_total += pending_new
            self._reset_pending_new_max = max(self._reset_pending_new_max, pending_new)
            self._reset_ground_total += ground_count
            self._reset_ground_max = max(self._reset_ground_max, ground_count)

            self._reset_bursts += 1
            self._reset_burst_steps_total += self._since_reset_steps
            self._reset_burst_steps_max = max(self._reset_burst_steps_max, self._since_reset_steps)
            self._reset_burst_extensions_total += self._since_reset_extensions
            self._reset_burst_extensions_max = max(self._reset_burst_extensions_max, self._since_reset_extensions)

            self._since_reset_steps = 0
            self._since_reset_extensions = 0
            self._bump_reset_reason(action.reason or 'reset')
            return action

        return action

    @staticmethod
    def _avg(total: float, count: int) -> float:
        return total / count if count else 0.0

    def _reset_final_info_piece(self) -> str:
        reasons = '|'.join(
            f'{reason}:{count}'
            for reason, count in sorted(self._reset_reason_counts.items())
        ) or 'none'
        return (
            'reset='
            f'policy:{self._reset_policy},'
            f'cuts:{"on" if self._reset_forward_cuts else "off"},'
            f's:{self._reset_starts},'
            f'r:{self._reset_reductions},'
            f'e:{self._reset_extensions},'
            f'b1:{self._reset_bt1},'
            f'b2:{self._reset_bt2},'
            f'rs:{self._reset_total},'
            f'same:{self._reset_same_depth},'
            f'deep:{self._reset_deeper},'
            f'grow:{self._reset_growing},'
            f'stall:{self._reset_stagnant},'
            f'avg_burst:{self._avg(self._reset_burst_steps_total, self._reset_bursts):.2f},'
            f'max_burst:{self._reset_burst_steps_max},'
            f'avg_ext:{self._avg(self._reset_burst_extensions_total, self._reset_bursts):.2f},'
            f'max_ext:{self._reset_burst_extensions_max},'
            f'avg_new:{self._avg(self._reset_pending_new_total, self._reset_total):.2f},'
            f'max_new:{self._reset_pending_new_max},'
            f'avg_ground:{self._avg(self._reset_ground_total, self._reset_total):.2f},'
            f'max_ground:{self._reset_ground_max},'
            f'reasons:{reasons}'
        )


class _SATResetBase(_ResetStatsMixin, SATSolveLiteral):
    """Shared step loop for reset-oriented SAT strategies."""

    _forward_cuts = True

    def __init__(self, env):
        super().__init__(env)
        self._init_reset_stats()

    def _forward_cuts_enabled(self) -> bool:
        return self._forward_cuts

    def _before_step(self, actions: list[ConnectionAction]) -> None:
        del actions

    def _same_depth_reset(self, reason: str) -> Reset:
        return Reset(id = 'rs', depth = self.env.state.max_depth, reason = reason)

    def _copy_reset(self, action: Reset, reason: str) -> Reset:
        return Reset(id = action.id, depth = action.depth, reason = reason)

    def _available_reset(self, actions: list[ConnectionAction]) -> Reset | None:
        return next((action for action in actions if isinstance(action, Reset)), None)

    def _proactive_reset(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> Reset | None:
        del actions, chosen
        return None

    def _dead_end_action(self, actions: list[ConnectionAction]) -> ConnectionAction:
        if action := next((candidate for candidate in actions if isinstance(candidate, Backtrack)), None):
            return action
        if action := self._available_reset(actions):
            return self._copy_reset(action, 'dead_reset')
        raise LookupError('expected Backtrack or Reset action')

    def step(self):
        actions = self.env.legal_actions()
        self._before_step(actions)
        self._record_nonzero_atom_scores()
        action = self.choose_action(actions)
        self._record_literal_choice_diagnostics(actions, action)

        if reset := self._proactive_reset(actions, action):
            return self._record_emitted_action(reset)

        if isinstance(action, (Reduction, Extension)) and self._forward_cuts_enabled():
            action.cut = True

        if isinstance(action, (Start, Reduction, Extension)):
            return self._record_emitted_action(action)

        return self._record_emitted_action(self._dead_end_action(actions))

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        parts.append(self._reset_final_info_piece())
        return '; '.join(parts) if parts else None


class _ControlBanditMixin(_BanditTableMixin):
    """Bandit over backtrack / small backjump / reset control actions."""

    _control_schema = 'sat_solve_learn_control_bandit/v1'
    _control_arms = ('bt1', 'bt2', 'reset')
    _control_contexts = ('shadow_growing', 'shadow_stagnant')
    _control_prior_count = 1.0
    _control_ucb_scale = 0.40
    _control_prior_reward_mass = {
        'shadow_growing': {'bt1': 1.0, 'bt2': 1.0, 'reset': 1.0},
        'shadow_stagnant': {'bt1': 1.0, 'bt2': 1.0, 'reset': 1.0},
    }
    _control_arm_alias = {'bt1': 'b1', 'bt2': 'b2', 'reset': 'rs'}
    _control_context_alias = {'shadow_growing': 'grow', 'shadow_stagnant': 'stall'}

    def _init_control_bandit(self) -> None:
        self._control_pulls = self._zero_bandit_table(self._control_contexts, self._control_arms, int)
        self._control_rewards = self._zero_bandit_table(self._control_contexts, self._control_arms, float)
        self._control_decisions = 0
        self._control_updates = 0
        self._control_agreed_default = 0
        self._control_changed_default = 0
        self._control_open = None

    def _control_context(self) -> str:
        if getattr(self.env.state.shadow, 'new_ground_clauses', ()):
            return 'shadow_growing'
        return 'shadow_stagnant'

    def _control_score(self, context: str, arm: str) -> float:
        count = self._control_prior_count + self._control_pulls[context][arm]
        reward = self._control_prior_reward_mass[context][arm] + self._control_rewards[context][arm]
        mean = reward / count
        total = len(self._control_arms) * self._control_prior_count + sum(self._control_pulls[context].values())
        bonus = self._control_ucb_scale * math.sqrt(math.log(max(total, 2.0)) / count)
        return mean + bonus

    def _control_arm_choices(self, actions: list[ConnectionAction]) -> dict[str, ConnectionAction]:
        choices = {}
        backtrack_action = next((action for action in actions if isinstance(action, Backtrack)), None)
        if backtrack_action is not None:
            choices['bt1'] = backtrack_action
            goal = getattr(self.env.state, 'goal', None)
            if (
                goal is not None
                and getattr(goal, 'parent', None) is not None
                and getattr(goal.parent, 'parent', None) is not None
            ):
                choices['bt2'] = Backtrack(id = 'bt2', steps = 2)

        reset_action = next((action for action in actions if isinstance(action, Reset)), None)
        if reset_action is not None:
            choices['reset'] = reset_action
        return choices

    def _finalize_control_record(self, reward: float, outcome: str, ground_gain: int) -> None:
        del outcome, ground_gain
        if self._control_open is None:
            return

        context = self._control_open['context']
        arm = self._control_open['arm']
        self._control_pulls[context][arm] += 1
        self._control_rewards[context][arm] += reward
        self._control_updates += 1
        self._control_open = None

    def _settle_control_feedback(self, actions: list[ConnectionAction], force: bool = False) -> None:
        if self._control_open is None:
            return

        state = self.env.state
        ground_gain = max(0, len(getattr(state.shadow, 'ground_clauses', ())) - self._control_open['ground_clause_count'])
        theorem = bool(
            getattr(state, 'is_terminal', False)
            and isinstance(getattr(state, 'info', None), str)
            and state.info.startswith('Theorem')
        )
        forward_available = any(isinstance(action, (Start, Reduction, Extension)) for action in actions)

        if theorem:
            self._finalize_control_record(1.0, 'theorem', ground_gain)
        elif ground_gain > 0:
            self._finalize_control_record(1.0, 'ground', ground_gain)
        elif force or not forward_available:
            self._finalize_control_record(0.0, 'forced' if force else 'stalled', ground_gain)

    def _choose_control_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        choices = self._control_arm_choices(actions)
        if not choices:
            return None
        if len(choices) == 1:
            return next(iter(choices.values()))

        context = self._control_context()
        arm_scores = {
            arm: self._control_score(context, arm)
            for arm in choices
        }
        chosen_arm = max(choices, key = lambda arm: arm_scores[arm])
        chosen = choices[chosen_arm]
        default_choice = choices['bt1'] if 'bt1' in choices else next(iter(choices.values()))

        self._control_decisions += 1
        if chosen is default_choice:
            self._control_agreed_default += 1
        else:
            self._control_changed_default += 1

        self._control_open = {
            'context': context,
            'arm': chosen_arm,
            'ground_clause_count': len(getattr(self.env.state.shadow, 'ground_clauses', ())),
        }
        return chosen

    def _control_context_summary(self, context: str) -> str:
        total = sum(self._control_pulls[context].values())
        arm_parts = []
        for arm in self._control_arms:
            pulls = self._control_pulls[context][arm]
            mean = self._control_rewards[context][arm] / pulls if pulls else 0.0
            arm_parts.append(f'{self._control_arm_alias[arm]}:{pulls}/{mean:.2f}')
        return f'{self._control_context_alias[context]}:{total}[{",".join(arm_parts)}]'

    def _control_nonzero_contexts(self) -> list[str]:
        return [
            context
            for context in self._control_contexts
            if sum(self._control_pulls[context].values()) > 0
        ]

    def _control_final_info_piece(self) -> str | None:
        if not self._control_decisions:
            return None
        used_contexts = self._control_nonzero_contexts()
        context_summary = ','.join(
            self._control_context_summary(context)
            for context in used_contexts
        )
        return (
            'control='
            f'schema:{self._control_schema},'
            f'decisions:{self._control_decisions},'
            f'updates:{self._control_updates},'
            f'agreed_default:{self._control_agreed_default},'
            f'changed_default:{self._control_changed_default},'
            f'used_contexts:{len(used_contexts)},'
            f'{context_summary}'
        )


class SATReset(_ControlBanditMixin, _SATResetBase):
    """Rename of the learned reset controller discovered from experiments."""

    def __init__(self, env):
        super().__init__(env)
        self._init_control_bandit()

    def _before_step(self, actions: list[ConnectionAction]) -> None:
        self._settle_control_feedback(actions)

    def _dead_end_action(self, actions: list[ConnectionAction]) -> ConnectionAction:
        state = self.env.state
        if (
            state.goal.parent is None
            and any(isinstance(candidate, Reset) for candidate in actions)
            and state.shadow.new_ground_clauses
        ):
            return self._same_depth_reset('root_shadow')

        action = self._choose_control_action(actions)
        if action is None:
            raise LookupError('expected Backtrack or Reset action')
        if isinstance(action, Reset):
            return self._copy_reset(action, 'bandit_reset')
        return action

    def final_info(self) -> str | None:
        self._settle_control_feedback(self.env.legal_actions(), force = True)

        parts = []
        base_info = _SATResetBase.final_info(self)
        if base_info is not None:
            parts.append(base_info)
        control_info = self._control_final_info_piece()
        if control_info is not None:
            parts.append(control_info)
        return '; '.join(parts) if parts else None


class SATSolveLearnControl(SATReset):
    """Backward-compatible alias for the old strategy name."""


class SATResetNoCut(SATReset):
    """SATReset with cuts disabled on reductions/extensions."""

    _forward_cuts = False


class SATResetShadow(_SATResetBase):
    """Simple reset policy: keep restarting at the same depth while shadow grows."""

    def _dead_end_action(self, actions: list[ConnectionAction]) -> ConnectionAction:
        reset_action = self._available_reset(actions)
        if reset_action is not None and getattr(self.env.state.shadow, 'new_ground_clauses', ()):
            return self._same_depth_reset('shadow_reset')

        if action := next((candidate for candidate in actions if isinstance(candidate, Backtrack)), None):
            return action

        if reset_action is not None:
            return self._copy_reset(reset_action, 'shadow_deepen')

        raise LookupError('expected Backtrack or Reset action')


class SATResetStepBudget(SATResetShadow):
    """Same-depth reset after a bounded number of emitted steps."""

    def _step_budget(self) -> int:
        return max(8, 2 * max(1, self.env.state.max_depth))

    def _proactive_reset(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> Reset | None:
        if not isinstance(chosen, (Start, Reduction, Extension)):
            return None
        if self._available_reset(actions) is None:
            return None
        if self._since_reset_steps < self._step_budget():
            return None
        return self._same_depth_reset('step_budget')


class SATResetExtensionBudget(SATResetShadow):
    """Same-depth reset after a bounded number of extensions."""

    def _extension_budget(self) -> int:
        return max(4, max(1, self.env.state.max_depth))

    def _proactive_reset(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> Reset | None:
        if not isinstance(chosen, (Start, Reduction, Extension)):
            return None
        if self._available_reset(actions) is None:
            return None
        if self._since_reset_extensions < self._extension_budget():
            return None
        return self._same_depth_reset('extension_budget')
