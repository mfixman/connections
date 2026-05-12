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

from .base import Strategy
from .sat_solve import SATSolveLiteral


class _BanditTableMixin:
    """Allocate small ``context -> arm -> value`` tables for bandit state.

    The reset learners keep a few pull/reward counters indexed by a tiny
    discrete context space.  This mixin centralises the bookkeeping shape so
    both the reset bandit and the older learning strategies can build the same
    kind of table without duplicating loops.
    """

    def zero_bandit_table(self, contexts: tuple[str, ...], arms: tuple[str, ...], value):
        return {
            context: {arm: value() for arm in arms}
            for context in contexts
        }

    # Compatibility shim for older mixins that still call the underscored name.
    def _zero_bandit_table(self, contexts: tuple[str, ...], arms: tuple[str, ...], value):
        return self.zero_bandit_table(contexts, arms, value)


class _ResetStatsMixin:
    """Record how a reset-style policy spent its search budget.

    These strategies are interesting primarily because of *when* they reset,
    how long they search between resets, and whether they are harvesting new
    shadow clauses or just thrashing.  This mixin turns that information into a
    compact ``reset=...`` summary suitable for CSV output, so later experiment
    analysis can compare policies without replaying traces.
    """

    def _init_reset_stats(self) -> None:
        self._reset_policy = self.__class__.__name__
        self._reset_forward_cuts = bool(self.forward_cuts_enabled())

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
    """Common SAT-reset control loop layered on top of ``SATSolveLiteral``.

    All concrete strategies here keep the same forward ranking as
    ``SATSolveLiteral``: when a ``Start``, ``Reduction``, or ``Extension`` is
    available, the chosen forward action is still the one preferred by the SAT
    model.  The variation point is what happens around resets:

    - subclasses can decide whether forward cuts stay on;
    - subclasses can trigger a proactive same-depth reset before a dead end;
    - subclasses can override the dead-end action selection policy.

    This keeps the reset experiments comparable: they differ in reset control,
    not in the literal-ordering baseline.
    """

    _forward_cuts = True

    def __init__(self, env):
        super().__init__(env)
        self._init_reset_stats()

    def forward_cuts_enabled(self) -> bool:
        return self._forward_cuts

    def before_step(self, actions: list[ConnectionAction]) -> None:
        del actions

    def same_depth_reset(self, reason: str) -> Reset:
        return Reset(id = 'rs', depth = self.env.state.max_depth, reason = reason)

    def copy_reset(self, action: Reset, reason: str) -> Reset:
        return Reset(id = action.id, depth = action.depth, reason = reason)

    def available_reset(self, actions: list[ConnectionAction]) -> Reset | None:
        return next((action for action in actions if isinstance(action, Reset)), None)

    def proactive_reset(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> Reset | None:
        del actions, chosen
        return None

    def dead_end_action(self, actions: list[ConnectionAction]) -> ConnectionAction:
        if action := next((candidate for candidate in actions if isinstance(candidate, Backtrack)), None):
            return action
        if action := self.available_reset(actions):
            return self.copy_reset(action, 'dead_reset')
        raise LookupError('expected Backtrack or Reset action')

    def step(self):
        actions = self.env.legal_actions()
        self.before_step(actions)
        self._record_nonzero_atom_scores()
        action = self.choose_action(actions)
        self._record_literal_choice_diagnostics(actions, action)

        if reset := self.proactive_reset(actions, action):
            return self._record_emitted_action(reset)

        if isinstance(action, (Reduction, Extension)) and self.forward_cuts_enabled():
            action.cut = True

        if isinstance(action, (Start, Reduction, Extension)):
            return self._record_emitted_action(action)

        return self._record_emitted_action(self.dead_end_action(actions))

    def final_info(self) -> str | None:
        parts = []
        base_info = super().final_info()
        if base_info is not None:
            parts.append(base_info)
        parts.append(self._reset_final_info_piece())
        return '; '.join(parts) if parts else None


class _ControlBanditMixin(_BanditTableMixin):
    """Online control learner for dead-end actions.

    When no forward step is available, the policy has to decide whether to:

    - backtrack one level,
    - backtrack two levels, or
    - reset.

    This mixin runs a tiny UCB bandit over those three arms.  The context is
    deliberately coarse: it only distinguishes whether the shadow CNF is still
    growing.  Reward is also coarse: a control choice is rewarded if it leads
    to a theorem or to more grounded shadow clauses before the next stall.
    """

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

    def init_control_bandit(self) -> None:
        self._control_pulls = self.zero_bandit_table(self._control_contexts, self._control_arms, int)
        self._control_rewards = self.zero_bandit_table(self._control_contexts, self._control_arms, float)
        self._control_decisions = 0
        self._control_updates = 0
        self._control_agreed_default = 0
        self._control_changed_default = 0
        self._control_open = None

    # Compatibility shim for older callers in sat_solve_learn.py.
    def _init_control_bandit(self) -> None:
        self.init_control_bandit()

    def control_context(self) -> str:
        if getattr(self.env.state.shadow, 'new_ground_clauses', ()):
            return 'shadow_growing'
        return 'shadow_stagnant'

    # Compatibility shim for older overrides in sat_solve_learn.py.
    def _control_context(self) -> str:
        return self.control_context()

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

    def settle_control_feedback(self, actions: list[ConnectionAction], force: bool = False) -> None:
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

    # Compatibility shim for older callers in sat_solve_learn.py.
    def _settle_control_feedback(self, actions: list[ConnectionAction], force: bool = False) -> None:
        self.settle_control_feedback(actions, force = force)

    def choose_control_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        choices = self._control_arm_choices(actions)
        if not choices:
            return None
        if len(choices) == 1:
            return next(iter(choices.values()))

        context = self.control_context()
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

    # Compatibility shim for older callers in sat_solve_learn.py.
    def _choose_control_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        return self.choose_control_action(actions)

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

    def control_final_info_piece(self) -> str | None:
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

    # Compatibility shim for older callers in sat_solve_learn.py.
    def _control_final_info_piece(self) -> str | None:
        return self.control_final_info_piece()


class SATResetNoCut(_ControlBanditMixin, _SATResetBase):
    """``SATResetBandit`` without Prolog-style forward cuts.

    The control logic is unchanged, including the same root-reset rule and
    the same bandit over dead-end actions. The only difference is that
    reductions and extensions do not commit a cut, so the tableau is allowed
    to revisit more local alternatives before reset or backtrack control takes
    over.

    This is intentionally simple: it isolates the interaction between
    reset-heavy control and restricted backtracking.
    """

    _forward_cuts = False

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


class SATResetAtStaleBacktrack(_SATResetBase):
    """Reset instead of backtracking once the shadow stops growing.

    This is the simplest deterministic approximation to the current
    ``SATResetBandit`` story:

    - if fresh shadow clauses exist at a dead end, restart at the same depth to
      exploit that new information;
    - otherwise, skip local backtracking and deepen immediately.

    The forward policy remains exactly ``SATSolveLiteral``.
    """

    def dead_end_action(self, actions: list[ConnectionAction]) -> ConnectionAction:
        reset_action = self.available_reset(actions)
        if reset_action is not None and getattr(self.env.state.shadow, 'new_ground_clauses', ()):
            return self.same_depth_reset('shadow_reset')

        if reset_action is not None:
            return self.copy_reset(reset_action, 'stale_deepen')

        if action := next((candidate for candidate in actions if isinstance(candidate, Backtrack)), None):
            return action

        raise LookupError('expected Backtrack or Reset action')


class ResetSATCoP(Strategy):
    """Standalone version of ``SATResetAtStaleBacktrack``.

    Forward actions are ranked exactly as in ``SATSolveLiteral``. At dead
    ends, the policy resets at the same depth while the shadow CNF is still
    growing; otherwise, it deepens immediately.
    """

    def __init__(self, env):
        super().__init__(env)
        if not hasattr(self.env.state, 'sat_literal_value'):
            raise TypeError('ResetSATCoP requires the ClassicalSAT calculus')
        self.env.state.enable_paper_mode(True)

    def score(self, action: ConnectionAction):
        match action:
            case Start():
                return 2

            case Reduction(path_lit = path_lit):
                value = self.env.state.sat_literal_value(path_lit)
                if value is True:
                    return 0
                if value is None:
                    return 1
                return 3

            case Extension(clause_copy = clause_copy, lit_idx = lit_idx):
                value = None
                if 0 <= lit_idx < len(clause_copy):
                    value = self.env.state.sat_literal_value(clause_copy[lit_idx])
                if value is True:
                    return 0
                if value is None:
                    return 1
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

        if state.shadow.new_ground_clauses:
            return Reset(id = 'rs', depth = state.max_depth, reason = 'shadow_reset')

        if any(isinstance(candidate, (Backtrack, Reset)) for candidate in actions):
            return Reset(id = 'rs', depth = state.max_depth + 1, reason = 'stale_deepen')

        raise LookupError('expected Backtrack or Reset action')


class SATResetAtStaleAction(SATResetAtStaleBacktrack):
    """Deepen after too many actions without new shadow clauses.

    This keeps the same deterministic dead-end rule as
    ``SATResetAtStaleBacktrack``, but also watches whether the shadow CNF has
    grown recently.  If not, it proactively resets before the next chosen
    action once the stale-action budget is exhausted.
    """

    def __init__(self, env):
        super().__init__(env)
        self._stale_action_count = 0
        self._last_shadow_clause_count = self._discovered_shadow_clause_count()

    def stale_action_budget(self) -> int:
        return max(8, 2 * max(1, self.env.state.max_depth))

    def _discovered_shadow_clause_count(self) -> int:
        shadow = self.env.state.shadow
        return len(getattr(shadow, 'ground_clauses', ())) + len(getattr(shadow, 'new_ground_clauses', ()))

    def before_step(self, actions: list[ConnectionAction]) -> None:
        del actions
        current = self._discovered_shadow_clause_count()
        if current > self._last_shadow_clause_count:
            self._stale_action_count = 0
            self._last_shadow_clause_count = current

    def proactive_reset(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> Reset | None:
        del chosen
        reset_action = self.available_reset(actions)
        if reset_action is None:
            return None
        if self._stale_action_count < self.stale_action_budget():
            return None
        if getattr(self.env.state.shadow, 'new_ground_clauses', ()):
            return self.same_depth_reset('stale_action_shadow')
        return self.copy_reset(reset_action, 'stale_action_reset')

    def _record_emitted_action(self, action: ConnectionAction) -> ConnectionAction:
        action = super()._record_emitted_action(action)
        if isinstance(action, Reset):
            self._stale_action_count = 0
            self._last_shadow_clause_count = self._discovered_shadow_clause_count()
        else:
            self._stale_action_count += 1
        return action


class SATResetAtTwo(SATResetAtStaleAction):
    """Allow one same-depth zero-yield burst before deepening.

    This is a two-strike variant of ``SATResetAtStaleAction``. When a burst at
    the current depth ends without harvesting any fresh shadow clauses, the
    first such strike triggers one more same-depth reset. Only after a second
    consecutive zero-yield burst at that depth does the strategy deepen.
    """

    def __init__(self, env):
        super().__init__(env)
        self._zero_yield_strikes = 0

    def _zero_yield_reset(self, reset_action: Reset) -> Reset:
        if self._zero_yield_strikes >= 1:
            return self.copy_reset(reset_action, 'two_strike_deepen')
        return self.same_depth_reset('two_strike_retry')

    def dead_end_action(self, actions: list[ConnectionAction]) -> ConnectionAction:
        reset_action = self.available_reset(actions)
        if reset_action is None:
            if action := next((candidate for candidate in actions if isinstance(candidate, Backtrack)), None):
                return action
            raise LookupError('expected Backtrack or Reset action')

        if getattr(self.env.state.shadow, 'new_ground_clauses', ()):
            return self.same_depth_reset('shadow_reset')

        return self._zero_yield_reset(reset_action)

    def proactive_reset(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> Reset | None:
        del chosen
        reset_action = self.available_reset(actions)
        if reset_action is None:
            return None
        if self._stale_action_count < self.stale_action_budget():
            return None
        if getattr(self.env.state.shadow, 'new_ground_clauses', ()):
            return self.same_depth_reset('stale_action_shadow')
        return self._zero_yield_reset(reset_action)

    def _record_emitted_action(self, action: ConnectionAction) -> ConnectionAction:
        state = self.env.state
        current_depth = getattr(state, 'max_depth', 0)
        pending_new = len(getattr(state.shadow, 'new_ground_clauses', ()))

        action = super()._record_emitted_action(action)
        if not isinstance(action, Reset):
            return action

        if action.depth > current_depth or pending_new > 0:
            self._zero_yield_strikes = 0
        else:
            self._zero_yield_strikes += 1

        return action


class SATResetAtTwoFixedStep(SATResetAtTwo):
    """``SATResetAtTwo`` with a constant proactive reset after ``N`` steps.

    This keeps the same dead-end/two-strike control rule as ``SATResetAtTwo``,
    but replaces the inherited stale-action trigger with a fixed budget on the
    number of emitted actions since the last reset. Once the budget is reached
    and a reset is available:

    - if fresh shadow clauses are pending, restart at the same depth;
    - otherwise use the ordinary two-strike zero-yield rule.

    In paper terms this is the closest "reset every ``N`` steps" variant of
    ``SATResetAtTwo``.
    """

    _fixed_step_budget = 8

    def step_budget(self) -> int:
        return self._fixed_step_budget

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
            return self.same_depth_reset('fixed_step_shadow')
        return self._zero_yield_reset(reset_action)


class SATResetStart(SATResetAtStaleAction):
    """Rotate the first root start after each same-depth reset.

    This keeps the ``SATResetAtStaleAction`` reset policy unchanged, but after
    every same-depth reset it forces the next root start choice to a different
    start clause in round-robin order. The goal is to diversify the early
    grounded clauses harvested at a fixed depth without changing the ordinary
    non-root ``SATSolveLiteral`` policy.
    """

    def __init__(self, env):
        super().__init__(env)
        self._start_cycle = 0
        self._force_root_start = False

    def choose_action(self, actions: list[ConnectionAction]) -> ConnectionAction | None:
        state = self.env.state
        if self._force_root_start and getattr(getattr(state, 'goal', None), 'parent', None) is None:
            starts = [action for action in actions if isinstance(action, Start)]
            if starts:
                ordered = sorted(starts, key = lambda action: action.id)
                chosen = ordered[self._start_cycle % len(ordered)]
                self._force_root_start = False
                return chosen

        return super().choose_action(actions)

    def _record_emitted_action(self, action: ConnectionAction) -> ConnectionAction:
        state = self.env.state
        current_depth = getattr(state, 'max_depth', 0)

        action = super()._record_emitted_action(action)
        if not isinstance(action, Reset):
            return action

        if action.depth <= current_depth:
            self._force_root_start = True
            self._start_cycle += 1
        else:
            self._force_root_start = False

        return action


class SATResetShadow(_SATResetBase):
    """Deterministic shadow-growth reset policy.

    This is the simplest “SAT-resetmaxxing” baseline in the file.  It keeps the
    ``SATSolveLiteral`` forward policy, but removes online learning entirely:

    - if the shadow CNF has accumulated fresh grounded clauses, reset at the
      same depth immediately;
    - otherwise prefer ordinary ``Backtrack`` when available;
    - only deepen via the calculus' ordinary ``Reset`` when there is nothing
      else to do.

    In effect it distils the strongest qualitative lesson from the learned
    controller into a fixed rule.
    """

    def dead_end_action(self, actions: list[ConnectionAction]) -> ConnectionAction:
        reset_action = self.available_reset(actions)
        if reset_action is not None and getattr(self.env.state.shadow, 'new_ground_clauses', ()):
            return self.same_depth_reset('shadow_reset')

        if action := next((candidate for candidate in actions if isinstance(candidate, Backtrack)), None):
            return action

        if reset_action is not None:
            return self.copy_reset(reset_action, 'shadow_deepen')

        raise LookupError('expected Backtrack or Reset action')


class SATResetStepBudget(SATResetShadow):
    """Shadow-reset policy with a simple step-count restart trigger.

    This inherits the deterministic shadow-growth policy, but also forces a
    same-depth reset after a bounded number of emitted proof actions since the
    last reset.  The budget is scaled by current depth so shallow iterations are
    reset quickly while deeper ones get a slightly longer burst.

    It is meant as a cheap analogue of “restart early and often”, without any
    learning and without inspecting detailed SAT features.
    """

    def step_budget(self) -> int:
        return max(8, 2 * max(1, self.env.state.max_depth))

    def proactive_reset(
        self,
        actions: list[ConnectionAction],
        chosen: ConnectionAction | None,
    ) -> Reset | None:
        if not isinstance(chosen, (Start, Reduction, Extension)):
            return None
        if self.available_reset(actions) is None:
            return None
        if self._since_reset_steps < self.step_budget():
            return None
        return self.same_depth_reset('step_budget')
