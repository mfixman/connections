import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeVar

from connections.env import ConnectionEnv
from connections.calculi.classicalcut import ConnectionAction, Start, Reduction, Extension, Backtrack, Reset

TAction = TypeVar('TAction', bound = ConnectionAction)

def backtrack_or_reset(actions: list[ConnectionAction]) -> ConnectionAction:
    backtracks = (a for a in actions if isinstance(a, Backtrack))
    resets = (a for a in actions if isinstance(a, Reset))
    return next(backtracks, None) or next(resets)

def _action_score(action: ConnectionAction) -> float:
    """Heuristic score for a forward action.  Higher is better.

    Source:
        Inspired by the fewest-solutions and shortest-clause principles
        from Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2.

    Design choices (not from the paper):
        The paper states qualitative ordering principles — prefer
        subgoals with fewer solutions, and use clause length as a proxy
        — but never proposes a numeric scoring function.  The specific
        values (Reduction = 2.0, Extension/Start = 1 / clause_length)
        are our adaptation.  Reductions score highest because they close
        a subgoal immediately (0 new subgoals); extensions score
        inversely with clause length as a rough proxy for the
        shortest-clause principle.
    """
    if isinstance(action, Reduction):
        return 2.0
    elif isinstance(action, (Extension, Start)):
        return 1.0 / max(len(action.clause_copy), 1)
    return 0.0


class Strategy:
    env: ConnectionEnv

    def __init__(self, env):
        self.env = env

    def step(self):
        raise NotImplemented('Abstract class')

    @classmethod
    def strategies(cls):
        result = {}
        queue = list(cls.__subclasses__())
        while queue:
            subcls = queue.pop(0)
            result[subcls.__name__] = subcls
            queue.extend(subcls.__subclasses__())
        return result

class FirstAction(Strategy):
    def step(self):
        return self.env.legal_actions()[0]

class LeanCopCut(Strategy):
    """Mirror leanCoP's ``cut`` option for forward proof steps.

    This strategy follows the baseline proof-search order of
    ``FirstAction``, but if the selected action is an ``Extension`` or a
    ``Reduction``, it sets ``cut=True`` before returning it.

    Source:
        Otten (2010), "Restricting backtracking in connection calculi",
        §4.3, Definition 8.  Restricted backtracking cuts off
        alternative rule applications once a literal has been solved.
        The Prolog implementation inserts a ``!`` (cut) after each
        successful proof step (line 14 of Figure 11).  The ``scut``
        variant additionally restricts start-clause backtracking
        (§5.4, line c).

    Consistency with leanCoP:
        The Prolog ``cut`` option in leanCoP applies restricted
        backtracking after successful reduction and extension steps.
        It does not force a cut on start steps; that is the separate
        ``scut`` option.  This strategy therefore leaves ``Start``
        actions unchanged.

    Design choices (not from the paper):
        leanCoP's ``cut`` is unconditional: every reduction/extension
        step gets a cut.  This strategy reproduces that behaviour
        exactly for the ``cut`` option; ``scut`` is not implemented
        here.  The ``comp(J)`` logic (``should_comp`` and
        ``max_cut_height``) follows Otten (2010), §5.4 verbatim:
        run with cut until iterative deepening exceeds depth J,
        then restart from depth 1 without cut.
    """

    max_cut_height: int = 7
    should_comp: bool
    cut_has: bool

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

        return action


class HeuristicSelection(Strategy):
    """Select the highest-scored forward action.

    Source:
        Inspired by Letz and Stenz (2001), "Model Elimination and
        Connection Tableau Procedures", especially the fewest-solutions
        and shortest-clause principles (§2.6.1–2.6.2).

    Design choice:
        The paper gives qualitative ordering principles (prefer
        reductions, then shorter clauses) applied to subgoal reordering
        within a clause.  We repurpose them as a numeric scoring
        function for action selection at choice points.  See
        ``_action_score`` for details.
    """

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        if not forward:
            return backtrack_or_reset(actions)

        if len(forward) == 1:
            return forward[0]

        scores = [_action_score(a) for a in forward]
        best_idx = max(range(len(forward)), key = lambda i: scores[i])
        return forward[best_idx]


class SubgoalReordering(Strategy):
    """Strict categorical ordering: reductions first, then extensions
    sorted by clause length (ascending), then starts by clause length.

    Source:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2.  The fewest-solutions
        principle states that subgoals producing fewer solutions should
        be preferred.  Reductions close the current subgoal immediately
        (0 new subgoals), so they are tried first.  Among extensions,
        the shortest-clause principle prefers clauses with fewer
        literals, since the number of solutions from a clause is the
        product of the solutions of its subgoals (footnote 3, p. 2031).

    Design choice:
        The paper describes subgoal alternation — interleaving choice
        points across multiple open subgoals — which our step-at-a-time
        framework cannot directly express.  We implement the core
        ranking: reductions > extensions sorted by number of new
        subgoals (len(clause_copy) - 1).  Ties preserve the default
        depth-first order, matching the paper's baseline (§2.6, "depth-
        first left-most/right-most selection").
    """

    def _action_key(self, action: ConnectionAction) -> tuple[int, int]:
        """Sort key: (category, new_subgoal_count).

        Category 0 = Reduction (closes branch, 0 new subgoals).
        Category 1 = Extension (sorted by clause length ascending).
        Category 2 = Start (sorted by clause length ascending).
        """
        if isinstance(action, Reduction):
            return (0, 0)
        elif isinstance(action, Extension):
            # clause_copy includes the head literal; new subgoals = len - 1
            return (1, len(action.clause_copy) - 1)
        elif isinstance(action, Start):
            return (2, len(action.clause_copy))
        return (3, 0)

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        if not forward:
            return backtrack_or_reset(actions)

        if len(forward) == 1:
            return forward[0]

        # Stable sort preserves depth-first order within same category+length.
        forward.sort(key=self._action_key)
        return forward[0]


class EntropyCut(Strategy):
    """Selective restricted backtracking based on action-distribution
    entropy.

    Analogous to leanCoP's ``cut`` option (Otten 2010, Definition 8),
    which cuts off non-essential backtracking after every successful
    proof step.  Rather than cutting unconditionally, this strategy
    cuts only when the heuristic score distribution has low entropy
    (i.e. one action type clearly dominates).

    Source:
        Otten (2010), "Restricting backtracking in connection calculi",
        §4.3, Definition 8: restricted backtracking cuts off
        alternative rule applications once a literal is solved.  The
        Prolog implementation inserts a ``!`` after each successful
        proof step (line 14 of Figure 11).

        Zombori et al. (2021), "The Role of Entropy in Guiding a
        Connection Prover": uses entropy of a learned neural policy to
        measure search confidence.  When entropy is low the prover is
        confident and can commit to a branch.

    Design choices (not from the papers):
        leanCoP's ``cut`` is unconditional — every forward step is
        cut.  Zombori et al. use a *learned* neural policy whose
        entropy is a natural confidence measure.  We combine the two
        ideas but replace the neural policy with hand-written heuristic
        scores (see ``_action_score``).  The specific mechanism —
        softmax over heuristic scores, then cut when Shannon entropy
        < 0.69 — is original to this project.

        The threshold 0.69 ≈ ln(2) is the minimum entropy for two
        equally-scored actions; a threshold just below this means we
        cut only when one action type clearly dominates (e.g. one
        reduction vs. one extension).

        This strategy does *not* reorder forward actions; it always
        takes the first available action (baseline depth-first order)
        and only decides whether to annotate it with ``cut=True``.
    """

    entropy_threshold: float = 0.69

    def _softmax_entropy(self, scores: list[float]) -> float:
        """Shannon entropy of the softmax distribution over scores."""
        max_s = max(scores)
        exp_scores = [math.exp(s - max_s) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        return -sum(p * math.log(p) for p in probs if p > 0)

    def _apply_entropy_cut(self, chosen: ConnectionAction, forward: list[ConnectionAction]) -> ConnectionAction:
        """Set ``cut=True`` on *chosen* if the action distribution has
        low entropy (high confidence)."""
        scores = [_action_score(a) for a in forward]
        entropy = self._softmax_entropy(scores)
        chosen.cut = (entropy < self.entropy_threshold)
        return chosen

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        if not forward:
            return backtrack_or_reset(actions)

        if len(forward) == 1:
            return forward[0]

        return self._apply_entropy_cut(forward[0], forward)


class AdaptiveReset(Strategy):
    """Adaptive restart policy: backtrack, reset(+1), or reset(+2).

    Analogous to leanCoP's ``comp(J)`` option (Otten 2010, §5.4),
    which restarts proof search with a complete (cut-free) strategy
    once iterative deepening exceeds a fixed depth limit J.  Rather
    than a fixed threshold, this strategy uses dynamic signals to
    decide when and how far to reset.

    Source:
        Otten (2010), "Restricting backtracking in connection calculi",
        §5.4: ``comp(Limit)`` restores completeness by restarting from
        depth 1 without cut once PathLim exceeds Limit.

        Holden (2021), "Machine Learning for Automated Theorem
        Proving", Chapter 3: surveys SAT restart strategies that adapt
        restart frequency based on solver state (Luby restarts, glucose-
        style dynamic restarts).

        Li et al. (2024), "A Reinforcement Learning based Reset Policy
        for CDCL SAT Solvers": proposes Thompson Sampling to choose
        between restart policies in SAT.  Does not transfer to
        connection calculi (see project memory for negative result).

    Design choices (not from the papers):
        leanCoP's ``comp(J)`` uses a fixed depth limit; we replace it
        with two dynamic triggers:

        1. Consecutive failure count ≥ 5: if the prover hits 5
           dead-ends in a row, the current search direction is
           unlikely to lead to a proof.  The threshold of 5 is
           hand-tuned.
        2. Constraint saturation: if the number of learned constraints
           exceeds 2 × num_lits, further search at the current depth
           is unpromising.  This test is original to this project.

        The depth increment (+1 or +2) is decided by per-depth failure
        rate: if more than 80% of steps at the current depth were
        dead-ends, skip a level (+2); otherwise increment normally
        (+1).  This heuristic has no paper basis.

        Like ``EntropyCut``, this strategy does *not* reorder forward
        actions; it only affects the dead-end handling (backtrack vs
        reset).
    """

    max_consecutive_failures: int = 5

    def __init__(self, env):
        super().__init__(env)
        self.consecutive_failures: int = 0
        self.steps_at_depth: dict[int, int] = defaultdict(int)
        self.failures_at_depth: dict[int, int] = defaultdict(int)

    def _track_step(self, has_forward: bool) -> None:
        """Update per-depth statistics and failure counter."""
        state = self.env.state
        self.steps_at_depth[state.max_depth] += 1
        if has_forward:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.failures_at_depth[state.max_depth] += 1

    def _should_reset(self) -> tuple[bool, str]:
        """Decide whether to reset instead of backtracking."""
        if self.consecutive_failures >= self.max_consecutive_failures:
            return True, f'{self.consecutive_failures} consecutive failures'
        state = self.env.state
        if len(state.constraints) > state.matrix.num_lits * 2:
            return True, f'{len(state.constraints)} constraints (> {state.matrix.num_lits * 2})'
        return False, ''

    def _choose_reset_depth(self) -> int:
        """Choose the depth for the next iteration after a reset."""
        state = self.env.state
        current = state.max_depth
        steps = self.steps_at_depth[current]
        failures = self.failures_at_depth[current]
        if steps > 0 and failures / steps > 0.8:
            return current + 2
        return current + 1

    def _handle_dead_end(self, actions: list[ConnectionAction]) -> ConnectionAction:
        """Return a Reset (with chosen depth) or fall back to Backtrack."""
        do_reset, reason = self._should_reset()
        if do_reset:
            state = self.env.state
            depth = self._choose_reset_depth()
            delta = depth - state.max_depth
            self.consecutive_failures = 0
            return Reset(id = 'rs', depth = depth, reason = f'{reason}, +{delta} depth')
        return backtrack_or_reset(actions)

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        self._track_step(bool(forward))

        if not forward:
            return self._handle_dead_end(actions)

        return forward[0]


class SmartCut(AdaptiveReset, EntropyCut):
    """Entropy-based cut combined with adaptive reset.

    Analogous to leanCoP's ``[cut, comp(J)]`` strategy combination
    (Otten 2010, §5.5): restricted backtracking is active during the
    search, with a restart mechanism to recover from dead-end regions.

    Combines:
        ``EntropyCut`` — selective restricted backtracking (≈ ``cut``).
        ``AdaptiveReset`` — adaptive restart policy (≈ ``comp(J)``).

    This class does *not* reorder forward actions.  It follows the
    baseline depth-first order and only adds cut annotations and
    reset decisions.
    """

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        self._track_step(bool(forward))

        if not forward:
            return self._handle_dead_end(actions)

        if len(forward) == 1:
            return forward[0]

        return self._apply_entropy_cut(forward[0], forward)


@dataclass(frozen=True)
class ScheduledStrategySlot:
    strategy_name: str
    step_budget: int


class InitialSchedule(Strategy):
    """Run a fixed multi-strategy schedule loaded from JSON.

    The schedule file specifies a sequence of strategies and absolute
    step budgets.  Each slot runs its underlying strategy from a fresh
    prover state.  When the current slot exhausts its step budget, the
    environment is reset and the next strategy takes over.

    This mirrors the execution model used by
    ``paramils_schedule_wrapper.py`` for static schedules, except that
    budgets here are expressed directly in proof steps.
    """

    schedule_path = Path(__file__).resolve().parent / 'schedules' / 'initial_schedule.json'

    def __init__(self, env):
        super().__init__(env)
        self._slots = self._load_slots()
        self._slot_index = 0
        self._slot_steps = 0
        self._active_strategy = self._instantiate_strategy(self._slots[0].strategy_name)

    @classmethod
    def _load_slots(cls) -> tuple[ScheduledStrategySlot, ...]:
        with open(cls.schedule_path, encoding = 'utf-8') as handle:
            raw = json.load(handle)

        raw_slots = raw.get('slots')
        if not isinstance(raw_slots, list) or not raw_slots:
            raise ValueError(f'{cls.schedule_path} must define a non-empty "slots" list')

        slots: list[ScheduledStrategySlot] = []
        for index, raw_slot in enumerate(raw_slots, start = 1):
            if not isinstance(raw_slot, dict):
                raise ValueError(f'slot {index} in {cls.schedule_path} must be an object')

            strategy_name = raw_slot.get('strategy')
            step_budget = raw_slot.get('step_budget')
            if not isinstance(strategy_name, str) or not strategy_name:
                raise ValueError(f'slot {index} in {cls.schedule_path} is missing a valid strategy name')
            if not isinstance(step_budget, int) or step_budget <= 0:
                raise ValueError(f'slot {index} in {cls.schedule_path} must have a positive integer step_budget')

            slots.append(ScheduledStrategySlot(strategy_name = strategy_name, step_budget = step_budget))

        return tuple(slots)

    def _instantiate_strategy(self, strategy_name: str) -> Strategy:
        strategies = Strategy.strategies()
        if strategy_name == type(self).__name__:
            raise ValueError(f'{type(self).__name__} cannot schedule itself')
        if strategy_name not in strategies:
            choices = ', '.join(sorted(strategies))
            raise ValueError(f'unknown scheduled strategy {strategy_name!r}; available: {choices}')
        return strategies[strategy_name](self.env)

    def _advance_slot(self) -> bool:
        if self._slot_index + 1 >= len(self._slots):
            return False
        self._slot_index += 1
        self._slot_steps = 0
        self.env.reset()
        self._active_strategy = self._instantiate_strategy(self._slots[self._slot_index].strategy_name)
        return True

    def step(self):
        current = self._slots[self._slot_index]
        if self._slot_steps >= current.step_budget:
            self._advance_slot()

        action = self._active_strategy.step()
        self._slot_steps += 1
        return action
