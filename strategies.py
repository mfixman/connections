import math
from collections import defaultdict
from typing import Optional, TypeVar

from connections.env import ConnectionEnv
from connections.calculi.classicalcut import ConnectionAction, Start, Reduction, Extension, Backtrack, Reset

TAction = TypeVar('TAction', bound = ConnectionAction)


def choose_action_of_type(actions: list[ConnectionAction], action_type: type[TAction]) -> Optional[TAction]:
    return next((a for a in actions if isinstance(a, action_type)), None)


class Strategy:
    env: ConnectionEnv

    def __init__(self, env):
        self.env = env

    def step(self):
        raise NotImplemented('Abstract class')

    @classmethod
    def strategies(cls):
        return {subcls.__name__: subcls for subcls in cls.__subclasses__()}

class FirstAction(Strategy):
    def step(self):
        return self.env.legal_actions()[0]

class LeanCopCut(Strategy):
    """Mirror leanCoP's `cut` option for forward proof steps.

    This strategy follows the baseline proof-search order of
    `FirstAction`, but if the selected action is an `Extension` or a
    `Reduction`, it sets `cut=True` before returning it.

    Consistency with leanCoP:
        The Prolog `cut` option in leanCoP applies restricted
        backtracking after successful reduction and extension steps.
        It does not force a cut on start steps; that is the separate
        `scut` option. This strategy therefore leaves `Start` actions
        unchanged.
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

class _SmartSupport:
    """Shared helpers for the SmartCut variants."""

    def _init_entropy_cut(self):
        # Entropy threshold below which we set cut=True.
        # Lower values = more conservative (cut less often).
        # With softmax over our heuristic scores, entropy of ln(2)
        # ≈ 0.693 is the minimum for 2 actions. A threshold just
        # below this means we only cut when there is a single dominant
        # action type (e.g., one reduction vs one extension).
        self.entropy_threshold = 0.69

    def _init_reset_policy(self):
        # How many consecutive dead-ends before considering a reset.
        self.max_consecutive_failures = 5

        # Tracks consecutive steps where no forward action was available.
        self.consecutive_failures = 0

        # Per-depth productivity tracking for adaptive reset depth.
        self.steps_at_depth: dict[int, int] = defaultdict(int)
        self.failures_at_depth: dict[int, int] = defaultdict(int)

    def score(self, action: ConnectionAction) -> float:
        """Heuristic score for a forward action. Higher is better."""
        if isinstance(action, Reduction):
            return 2.0
        elif isinstance(action, Extension):
            return 1.0 / max(len(action.clause_copy), 1)
        elif isinstance(action, Start):
            return 1.0 / max(len(action.clause_copy), 1)
        return 0.0

    def choose_forward_action(self, forward: list[ConnectionAction]) -> tuple[ConnectionAction, list[float]]:
        scores = [self.score(a) for a in forward]
        best_idx = max(range(len(forward)), key = lambda i: scores[i])
        return forward[best_idx], scores

    def softmax_entropy(self, scores: list[float]) -> tuple[list[float], float]:
        """Compute softmax probabilities and Shannon entropy."""
        max_s = max(scores)
        exp_scores = [math.exp(s - max_s) for s in scores]
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        return probs, entropy

    def apply_entropy_cut(self, chosen: ConnectionAction, scores: list[float]) -> ConnectionAction:
        """Annotate the chosen action with cut=True if confidence is high."""
        _, entropy = self.softmax_entropy(scores)
        chosen.cut = (entropy < self.entropy_threshold)
        return chosen

    def should_reset(self) -> tuple[bool, str]:
        """Decide whether to reset instead of backtracking."""
        if self.consecutive_failures >= self.max_consecutive_failures:
            return True, f'{self.consecutive_failures} consecutive failures'
        state = self.env.state
        if len(state.constraints) > state.matrix.num_lits * 2:
            return True, f'{len(state.constraints)} constraints (> {state.matrix.num_lits * 2})'
        return False, ''

    def choose_reset_depth(self) -> int:
        """Choose the depth for the next iteration after a reset."""
        state = self.env.state
        current = state.max_depth
        steps = self.steps_at_depth[current]
        failures = self.failures_at_depth[current]
        if steps > 0 and failures / steps > 0.8:
            return current + 2
        return current + 1


class HeuristicSelection(Strategy, _SmartSupport):
    """Prefer reductions, then short extensions, then long extensions.

    Source:
        Inspired by Letz and Stenz (2001), "Model Elimination and
        Connection Tableau Procedures", especially the fewest-solutions
        and shortest-clause principles.

    Design choice:
        The exact numeric scoring scheme in `score()` is an adaptation
        for this codebase. In particular, scoring reductions as 2.0 and
        extensions as 1 / clause_length is not taken verbatim from the
        paper.
    """

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        if not forward:
            return choose_action_of_type(actions, Backtrack) or choose_action_of_type(actions, Reset)

        if len(forward) == 1:
            return forward[0]

        chosen, _ = self.choose_forward_action(forward)
        return chosen


class SubgoalReordering(Strategy, _SmartSupport):
    """Strict categorical ordering: reductions first, then extensions
    sorted by clause length (ascending), then starts by clause length.

    Source:
        Letz and Stenz (2001), "Model Elimination and Connection
        Tableau Procedures", §2.6.1–2.6.2. The fewest-solutions
        principle states that subgoals producing fewer solutions should
        be preferred. Reductions close the current subgoal immediately
        (0 new subgoals), so they are tried first. Among extensions,
        the shortest-clause principle prefers clauses with fewer
        literals, since the number of solutions from a clause is the
        product of the solutions of its subgoals (footnote 3, p. 2031).

    Design choice:
        The paper describes subgoal alternation — interleaving choice
        points across multiple open subgoals — which our step-at-a-time
        framework cannot directly express. We implement the core
        ranking: reductions > extensions sorted by number of new
        subgoals (len(clause_copy) - 1). Ties preserve the default
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
            return choose_action_of_type(actions, Backtrack) or choose_action_of_type(actions, Reset)

        if len(forward) == 1:
            return forward[0]

        # Stable sort preserves depth-first order within same category+length.
        forward.sort(key=self._action_key)
        return forward[0]


class EntropyCut(Strategy, _SmartSupport):
    """Add a cut when the forward-action distribution has low entropy.

    Source:
        Inspired by Zombori et al. (2021), "The Role of Entropy in
        Guiding a Connection Prover".
        The underlying cut semantics come from Otten (2010) and Otten
        (2023) on restricted backtracking in leanCoP.

    Design choice:
        The runtime rule "softmax over hand-written action scores, then
        cut when entropy < 0.69" is an original adaptation here. This
        class preserves baseline action choice and only changes whether
        the chosen forward action is annotated with `cut=True`.
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_entropy_cut()

    def step(self):
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        if not forward:
            return choose_action_of_type(actions, Backtrack) or choose_action_of_type(actions, Reset)

        if len(forward) == 1:
            return forward[0]

        chosen = forward[0]
        scores = [self.score(a) for a in forward]
        return self.apply_entropy_cut(chosen, scores)


class AdaptiveReset(Strategy, _SmartSupport):
    """Choose between backtracking, Reset(+1), and Reset(+2).

    Source:
        Loosely inspired by Li et al. (2024), "A Reinforcement
        Learning based Reset Policy for CDCL SAT Solvers", and by the
        SAT restart literature summarized in Holden (2021).

    Design choice:
        The concrete policy here is original to this project: dead-end
        resets, the threshold of 5 consecutive failures, the
        constraint-saturation test, and the +1/+2 depth jump heuristic
        are not taken verbatim from those papers.
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_reset_policy()

    def step(self):
        state = self.env.state
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        # Track per-depth statistics.
        self.steps_at_depth[state.max_depth] += 1

        if not forward:
            self.consecutive_failures += 1
            self.failures_at_depth[state.max_depth] += 1

            do_reset, reason = self.should_reset()
            if do_reset:
                depth = self.choose_reset_depth()
                delta = depth - state.max_depth
                self.consecutive_failures = 0
                return Reset(id = 'rs', depth = depth, reason = f'{reason}, +{delta} depth')
            return choose_action_of_type(actions, Backtrack) or choose_action_of_type(actions, Reset)

        self.consecutive_failures = 0
        return forward[0]


class SmartCut(Strategy, _SmartSupport):
    """Combine the three SmartCut mechanisms.

    This strategy combines:
        1. heuristic forward-action selection,
        2. entropy-based cut annotation,
        3. adaptive backtrack/reset control.
    """

    def __init__(self, env):
        super().__init__(env)
        self._init_entropy_cut()
        self._init_reset_policy()

    def step(self):
        state = self.env.state
        actions = self.env.legal_actions()
        forward = [a for a in actions if isinstance(a, (Start, Reduction, Extension))]

        # Track per-depth statistics.
        self.steps_at_depth[state.max_depth] += 1

        if not forward:
            self.consecutive_failures += 1
            self.failures_at_depth[state.max_depth] += 1

            do_reset, reason = self.should_reset()
            if do_reset:
                depth = self.choose_reset_depth()
                delta = depth - state.max_depth
                self.consecutive_failures = 0
                return Reset(id = 'rs', depth = depth, reason = f'{reason}, +{delta} depth')
            return choose_action_of_type(actions, Backtrack) or choose_action_of_type(actions, Reset)

        self.consecutive_failures = 0

        if len(forward) == 1:
            return forward[0]

        chosen, scores = self.choose_forward_action(forward)
        return self.apply_entropy_cut(chosen, scores)
